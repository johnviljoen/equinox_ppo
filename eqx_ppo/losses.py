import jax
import jax.numpy as jnp
import jax.random as jr

from models import PPOStochasticActor, PPOValueNetwork
from stats import RunningMeanStd


def compute_gae(truncation: jnp.ndarray,
                termination: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float, # = 1.0,
                discount: float): # = 0.99
    """Calculates the Generalized Advantage Estimation (GAE).

    Args:
        truncation: A float32 tensor of shape [T, B] with truncation signal.
        termination: A float32 tensor of shape [T, B] with termination signal.
        rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
        values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
        bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
        lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
        lambda_=1.
        discount: TD discount.

    Returns:
        A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
        A float32 tensor of shape [T, B] of advantages.
    """
    # Used everywhere, mask = (1 - d_t)
    truncation_mask = 1 - truncation
    
    # TD residuals:
    # delta_t = r_t + gamma * (1-d_t) * V(s_{t+1}) - V(s_t)
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate([values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    # Recursive Advantage calculation
    # A_t = delta_t + gamma * lambda * (1-d_t) * A_{t+1}
    # when lambda = 1 all future steps are fully considered, when lambda = 0 only 1 step TD error is considered
    # acc is the accumulated advantage
    acc = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []
    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)
    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs, (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True)

    # Final Values and Advantages
    # advantage = (rewards + gamma * (1-termination) * V(s_{t+1})) - V(s_t)
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)
    vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount * (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_ppo_loss(
        actor_network: PPOStochasticActor, # Policy network (Equinox module)
        value_network: PPOValueNetwork,        # Value network (Equinox module)
        observation_rms: RunningMeanStd,        # Running mean std parameters
        data,      # Transition data
        rng: jnp.array,
        entropy_cost: float, # = 1e-4,
        discounting: float, # = 0.99,
        reward_scaling: float, # = 1.0,
        gae_lambda: float, # = 0.95,
        clipping_epsilon: float, # = 0.2,
        normalize_advantage: bool # = True
    ):

    # Put the time dimension first.
    # data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Normalize data
    obs_normalized = observation_rms.normalize(data["obs"])
    next_obs_normalized = observation_rms.normalize(data["next_obs"])

    # Calculate the estimated advantages and values
    baseline = jax.vmap(jax.vmap(value_network))(obs_normalized)
    bootstrap_value = jax.vmap(value_network)(next_obs_normalized[-1])
    rewards = data["reward"] * reward_scaling
    truncation = data["truncation"]
    termination = (1 - data["discount"]) * (1 - truncation)
    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting)
    
    # optional normalization
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Calculate the Policy Ratio rho_s
    # rho_s = policy(a_t|s_t) / policy_old(a_t|s_t) = exp(log policy(a_t|s_t) - log policy_old(a_t|s_t))
    # Compute the log probabilities under the current policy
    # Use deterministic outputs (mean) from the policy network
    target_action_log_probs = jax.vmap(jax.vmap(actor_network.log_prob))(obs_normalized, data['action'])
    behaviour_action_log_probs = data["log_prob"]
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    # jax.debug.print("rho_s: {}", rho_s.mean())

    # Surrogate Objective: we use a clipped version of the policy ratio to prevent large updates
    # L_surrogate = min(rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t)
    # where rho_t * A_t: the standard policy gradient
    # where clip(rho_t, 1-eps, 1+eps) * A_t: the clipped version which constrains ratio rho_t within [1-eps, 1+eps]
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages

    # The final policy loss is the negative of the mean of the clipped surrogate objective
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    # value_loss = 1/2 * 1/T * Sum_{t=0}^T (V_s - V(s_t)) ** 2
    # the loss is scaled y an additional factor 0.5 likely to balance the loss components
    v_error = vs - baseline
    v_loss = jnp.mean(v_error ** 2) * 0.5 * 0.5

    # Entropy reward
    # entropy = - Sum_a policy(a|s) log policy(a|s) * entropy_cost

    ### JOHN YOU ARE HERE - WE HAVE TO LOOK AT THIS TO ENSURE WE ARE VMAPPING AND AVERAGING CORRECTLY
    ### THIS ENTROPY GOES NEGATIVE WHEN THATS IMPOSSIBLE

    key_entropy = jr.split(rng, obs_normalized.shape[0:2])
    entropy = jnp.mean(jax.vmap(jax.vmap(actor_network.entropy))(key_entropy, obs_normalized))
    entropy_loss = entropy_cost * -entropy
    # entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))

    # jax.debug.print("DEBUG: entropy: {}", entropy)

    # Total loss = policy loss + value loss + entropy loss
    total_loss = policy_loss + v_loss + entropy_loss

    # jax.debug.print("v_loss: {v_loss}", v_loss = v_loss)
    # jax.debug.print("total_loss: {total_loss}", total_loss = total_loss)

    return total_loss, {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'v_loss': v_loss,
        'entropy_loss': entropy_loss
    }

if __name__ == "__main__": 

    seed = 0

    import jax.random as jr
    from brax import envs

    def unit_test_gae_loss(seed):

        rng = jr.PRNGKey(seed); _rng, rng = jr.split(rng)

        truncation = jnp.array([
            [0,0],
            [0,0],
            [1,0],
            [0,0],
            [0,1],
        ])

        termination = jnp.array([
            [0,1],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
        ])

        rewards = jr.normal(_rng, [5,2]); _rng, rng = jr.split(rng)
        values = jr.normal(_rng, [5,2]); _rng, rng = jr.split(rng)
        bootstrap_value = jr.normal(_rng, [2]); _rng, rng = jr.split(rng)

        gae_lambda = 1.0
        gae_discount = 0.99

        test_vs, test_advantages = compute_gae(truncation, termination, rewards, values, bootstrap_value, lambda_=gae_lambda, discount=gae_discount)

        truncation_mask = 1 - truncation

        # TD residuals:
        # delta_t = r_t + gamma * (1-d_t) * V(s_{t+1}) - V(s_t)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = jnp.concatenate([values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
        deltas = rewards + gae_discount * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        # Initialization
        acc = jnp.zeros_like(bootstrap_value)
        vs_minus_v_xs = []

        # Reverse loop for GAE accumulation
        for t in reversed(range(truncation_mask.shape[0])):
            delta = deltas[t]
            trunc_mask = truncation_mask[t]
            term = termination[t]
            # Accumulate advantage
            acc = delta + gae_discount * (1 - term) * trunc_mask * gae_lambda * acc
            # Store the accumulated value
            vs_minus_v_xs.insert(0, acc)  # Insert at the front to reverse the order since we are iterating in reverse
        vs_minus_v_xs = jnp.stack(vs_minus_v_xs)

        # Final Values and Advantages
        # advantage = (rewards + gamma * (1-termination) * V(s_{t+1})) - V(s_t)
        # Add V(x_s) to get v_s.
        vs = jnp.add(vs_minus_v_xs, values)
        vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
        advantages = (rewards + gae_discount * (1 - termination) * vs_t_plus_1 - values) * truncation_mask

    # unit_test_gae_loss(seed)

    def unit_test_ppo_loss(seed):

        import dataclasses
        import optax
        import equinox as eqx

        from train import AgentModel, TrainingState
        from brax import envs
        # from brax_example.losses import compute_ppo_loss as compute_ppo_loss_example

        env = envs.get_environment('ant', backend='positional')

        num_timesteps=50_000_000
        episode_length=1000
        num_envs=int(4096*1)
        learning_rate=3e-4
        entropy_cost=1e-2
        discounting=0.97
        seed=1
        unroll_length=5
        batch_size=2048
        num_minibatches=32
        num_updates_per_batch=4
        reward_scaling=10.
        clipping_epsilon=0.2
        gae_lambda=0.95
        normalize_advantage=True

        key = jr.PRNGKey(seed)
        _key, key = jr.split(key)

        env_v = envs.training.wrap(env, episode_length=episode_length)
        actor_network = PPOStochasticActor(_key, layer_sizes=[env.observation_size, 64, 64, env.action_size]); _key, key = jr.split(key)
        value_network = PPOValueNetwork(_key, layer_sizes=[env.observation_size, 64, 64, 1]); _key, key = jr.split(key)
        model = AgentModel(actor_network=actor_network, value_network=value_network)
        opt = optax.adam(learning_rate)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        obs_rms = RunningMeanStd(mean=jnp.zeros(env.observation_size), var=jnp.ones(env.observation_size), count=1e-4)
        training_state = TrainingState(opt_state, model, obs_rms, env_steps=0)

        def generate_rollout_v(key, env_state, training_state, env_v=env_v, unroll_length=unroll_length):
            
            def step_fn(carry, _):
                state, key = carry
                obs = state.obs  
                _key, key = jr.split(key)
                obs_normalized = training_state.observation_rms.normalize(obs)
                action, raw_action = jax.vmap(training_state.model.actor_network)(jr.split(_key, batch_size), obs_normalized)
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

            (final_state, _), data_seq = jax.lax.scan(
                step_fn, (env_state, key), None, length=unroll_length
            )

            return data_seq, final_state

        env_reset_jv = jax.jit(env_v.reset)
        generate_unroll_jv = eqx.filter_jit(generate_rollout_v)
        total_steps = 0
        env_state = env_reset_jv(jr.split(_key, num=batch_size)); _key, key = jr.split(key)

        # single unroll to test ppo_loss
        data, env_state = generate_unroll_jv(_key, env_state, training_state); _key, key = jr.split(key)
        new_observation_rms = training_state.observation_rms.update(data['obs'])
        training_state = dataclasses.replace(training_state, observation_rms=new_observation_rms)
        
        # compute ppo loss via my method and via the known to be correct method
        loss, all_losses = compute_ppo_loss(
            actor_network=model.actor_network,
            value_network=model.value_network,
            observation_rms=training_state.observation_rms,
            data=data,
            rng=key,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage
        )

        # loss_example = compute_ppo_loss_example(

        # )

        print('fin')

    unit_test_ppo_loss(seed)