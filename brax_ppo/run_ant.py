import functools
import jax

from datetime import datetime
import matplotlib.pyplot as plt

from IPython.display import HTML, clear_output

import brax

from brax import envs
from brax.io import model
from brax.io import html

import train as ppo

env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'positional'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

HTML(html.render(env.sys, [state.pipeline_state]))

# working default arguments
ant_ppo_train = functools.partial(
    ppo.train,  
    num_timesteps=50_000_000, 
    num_evals=10, 
    reward_scaling=10, 
    episode_length=1000, 
    normalize_observations=True, 
    action_repeat=1, 
    unroll_length=5, 
    num_minibatches=32, 
    num_updates_per_batch=4, 
    discounting=0.97, 
    learning_rate=3e-4, 
    entropy_cost=1e-2, 
    num_envs=4096, 
    batch_size=2048, 
    seed=1
)

max_y = 8000
min_y = 0

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    clear_output(wait=True)
    plt.xlim([0, ant_ppo_train.keywords['num_timesteps']])
    plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()

make_inference_fn, params, _ = ant_ppo_train(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

model.save_params('tmp/params', params)
params = model.load_params('tmp/params')
inference_fn = make_inference_fn(params)

# create an env with auto-reset
env = envs.create(env_name=env_name, backend=backend)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
for _ in range(1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)

with open(f'tmp/{env_name}.html', 'w') as f:
    f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))
