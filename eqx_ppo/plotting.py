import jax
import equinox as eqx
from tqdm import tqdm

# BRAX rendering
from IPython.display import HTML, clear_output
from brax.io import model
from brax.io import html

def rollout_and_render(env, actor, obs_rms, save_path="tmp/test.html"):
    print("rolling out policy to render")
    rollout = []
    rng = jax.random.PRNGKey(seed=0)
    env_state = env.reset(rng=rng)
    env_step = jax.jit(env.step)
    actor_call = eqx.filter_jit(actor.forward_deterministic) # we are running the deterministic policy!
    for _ in tqdm(range(1000)):
        rollout.append(env_state.pipeline_state)
        _rng, rng = jax.random.split(rng)
        obs = env_state.obs
        obs_normalized = obs_rms.normalize(obs)
        act = actor_call(obs_normalized)
        env_state = env_step(env_state, act)

    print("rendering")
    with open(save_path, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))
