from functools import partial
from time import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import dataclasses
from brax import envs

from models import PPOStochasticActor, PPOValueNetwork
from stats import RunningMeanStd
from losses import compute_ppo_loss
from eqx_utils import filter_scan
from plotting import rollout_and_render

# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_debug_nans", True)

class AgentModel(eqx.Module):
    actor_network: PPOStochasticActor
    value_network: PPOValueNetwork

class TrainingState(eqx.Module):
    opt_state: optax.OptState
    model: AgentModel
    obs_rms: RunningMeanStd
    env_steps: jnp.ndarray

def generate_rollout_v(key, env_state, training_state, env_v, unroll_length):
    
    # @eqx.filter_jit
    def step_fn(carry, _):
        state, key = carry
        obs = state.obs # this is one obs across the batch
        batch_size = obs.shape[0]
        _key, key = jr.split(key)
        obs_normalized = training_state.obs_rms.normalize(obs)
        action, raw_action = jax.vmap(training_state.model.actor_network)(jr.split(_key, batch_size), obs_normalized)
        # jax.debug.print("action: {}", action)
        next_state = env_v.step(state, action)
        log_prob = jax.vmap(training_state.model.actor_network.log_prob)(obs_normalized, action)
        value = jax.vmap(training_state.model.value_network)(obs_normalized)
        data = {
            'obs': obs,
            'action': action,
            'reward': next_state.reward,
            'discount': 1 - next_state.done, # how we decide if truncation is terminal or not
            'truncation': next_state.info['truncation'],
            'log_prob': log_prob,
            'value': value,
            'next_obs': next_state.obs,
            'raw_action': raw_action, # action without gaussian applied to it
        }
        return (next_state, key), data

    (final_state, _), data_seq = filter_scan(
        step_fn, (env_state, key), None, length=unroll_length
    )

    return data_seq, final_state

def train(
        env,  # The environment, assumed to be JAX-compatible
        num_timesteps: int,
        episode_length: int,
        num_envs: int = 128,
        action_repeat: int = 1,
        num_evals: int = 1,
        num_resets_per_eval: int = 0,
        learning_rate: float = 1e-4,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        seed: int = 0,
        unroll_length: int = 10,
        minibatch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        reward_scaling: float = 1.0,
        clipping_epsilon: float = 0.3,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
    ):

    key = jr.PRNGKey(seed)
    _key, key = jr.split(key)

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (minibatch_size * unroll_length * num_minibatches * action_repeat)
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = int(np.ceil(
            num_timesteps
            / (
                    num_evals_after_init
                    * env_step_per_training_step
                    * max(num_resets_per_eval, 1)
            )
    ))

    env_v = envs.training.wrap(env, episode_length=episode_length)
    actor_network = PPOStochasticActor(_key, layer_sizes=[env.observation_size, 64, 64, env.action_size]); _key, key = jr.split(key)
    value_network = PPOValueNetwork(_key, layer_sizes=[env.observation_size, 64, 64, 1]); _key, key = jr.split(key)
    model = AgentModel(actor_network=actor_network, value_network=value_network)
    opt = optax.adam(learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array)) # eqx.is_inexact_array
    obs_rms = RunningMeanStd(mean=jnp.zeros(env.observation_size), var=jnp.ones(env.observation_size), count=1e-4)
    training_state = TrainingState(opt_state, model, obs_rms, env_steps=jnp.array(0))
    generate_unroll_v = partial(generate_rollout_v, env_v=env_v, unroll_length=unroll_length)

    def minibatch_step(carry, minibatch_data):
        training_state, key = carry
        key_loss, new_key = jr.split(key)

        def loss_fn(model):
            loss, metrics = compute_ppo_loss(
                actor_network=model.actor_network,
                value_network=model.value_network,
                observation_rms=training_state.obs_rms,
                data=minibatch_data,
                rng=key_loss,
                entropy_cost=entropy_cost,
                discounting=discounting,
                reward_scaling=reward_scaling,
                gae_lambda=gae_lambda,
                clipping_epsilon=clipping_epsilon,
                normalize_advantage=normalize_advantage
            )
            return loss, metrics
        
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(training_state.model)
        updates, new_opt_state = opt.update(
            grads, training_state.opt_state, training_state.model
        )

        ### TESTING
        # abs_grads = jax.tree_util.tree_map(lambda x: jnp.abs(x).max(), grads)
        # max_grad = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, y), abs_grads)
        # jax.debug.print("DEBUG: grads max: {:.6f}", max_grad)
        # jax.debug.print("DEBUG: policy loss: {:.6f}", metrics["policy_loss"])
        # jax.debug.print("DEBUG: value loss: {:.6f}", metrics["v_loss"])
        # jax.debug.print("DEBUG: entropy loss: {:.6f}", metrics["entropy_loss"])

        new_model = eqx.apply_updates(training_state.model, updates)
        new_training_state = dataclasses.replace(
            training_state,
            opt_state=new_opt_state,
            model=new_model,
            env_steps=training_state.env_steps + unroll_length * num_envs
        )

        return (new_training_state, new_key), metrics

    def sgd_step(carry, _, data):
        training_state, key = carry
        key_perm, key_grad, new_key = jr.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (new_training_state, _), metrics = filter_scan(
                minibatch_step,
                (training_state, key_grad),
                shuffled_data,
                length=num_minibatches)
        
        # jax.debug.print("IMMEDIATELY AFTER SGD STEP SCAN")

        return (new_training_state, new_key), metrics
    
    def training_step(carry, _):

        training_state, env_state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        # jax.debug.print("DEBUG: key_generate_unroll: {}", key_generate_unroll)

        # jax.debug.print("DEBUG: actor std{}", training_state.model.actor_network.std)

        # gather data - we cut up a long trajectory into small trajectories from each iteration
        def f(carry, _):
            env_state, key = carry
            key, new_key = jr.split(key)
            data, next_state = generate_unroll_v(key, env_state, training_state)
            return (next_state, new_key), data
        
        (new_env_state, _), data = filter_scan(
            f, (env_state, key_generate_unroll), (),
            length=minibatch_size * num_minibatches // num_envs)
        
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

        # check that the different rollouts are actually doing different stuff - they are
        # jax.debug.print("DEBUG: minibatch rollout deltas: {:0.2f}", jnp.abs(data["obs"][:-1] - data["obs"][1:]).max())
        
        # jax.debug.print("IMMEDIATELY BEFORE TRAINING STEP SCAN")

        # sgd step
        (new_training_state, _), metrics = filter_scan(
            partial(sgd_step, data=data),
            (training_state, key_sgd), (),
            length=num_updates_per_batch)
        
        # jax.debug.print("IMMEDIATELY AFTER TRAINING STEP SCAN")

        return (new_training_state, new_env_state, new_key), metrics
        
    def training_epoch(key, training_state, env_state):

        # jax.debug.print("IMMEDIATELY BEFORE EPOCH SCAN")

        (new_training_state, new_env_state, _), loss_metrics = filter_scan(
            training_step, (training_state, env_state, key), (),
            length=num_training_steps_per_epoch)
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)

        # jax.debug.print("IMMEDIATELY AFTER EPOCH SCAN")

        return new_training_state, new_env_state, loss_metrics

    env_reset_jv = jax.jit(env_v.reset)
    env_state = env_reset_jv(jr.split(_key, num=num_envs)); _key, key = jr.split(key)

    # check generated trajectories - theyre legit
    # data, next_state = generate_unroll_v(key, env_state, training_state)
    # data2, next_state = generate_unroll_v(key, env_state, training_state)
    # for i in range(num_envs):
    #     x = data["obs"] - data["obs"]
    #     plt.plot(x[:,i,0], x[:,i,1])
    # plt.savefig('test.png', dpi=500)

    tic = time()

    for it in range(num_evals_after_init):

        print(it)

        for _ in range(max(num_resets_per_eval, 1)):
        
            # training_state, env_state = _strip_weak_type((training_state, env_state))

            training_state, env_state, training_metrics = training_epoch(_key, training_state, env_state); _key, key = jr.split(key)
        
            # training_state, env_state, training_metrics = _strip_weak_type(result)

            current_step = training_state.env_steps

            env_state = env_reset_jv(jr.split(_key, num=num_envs)) if num_resets_per_eval > 0 else env_state; _key, key = jr.split(key)

        # print(current_step)
    
    print(f"time taken: {time() - tic}")

    rollout_and_render(env, training_state.model.actor_network, training_state.obs_rms)

if __name__ == "__main__":

    training_state = train(
        env=envs.get_environment("inverted_pendulum", backend="positional"),
        num_timesteps=2_000_000, num_evals=20, reward_scaling=10, episode_length=1000, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, minibatch_size=1024, seed=1
    )

    # training_state = train(
    #     env=envs.get_environment("ant", backend="positional"),
    #     num_timesteps=50_000_000, num_evals=10, reward_scaling=10, episode_length=1000, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, minibatch_size=2048, seed=1
    # )
