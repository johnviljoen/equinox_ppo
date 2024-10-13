# Equinox PPO Implementation on Brax

This repo was based upon the [brax implementation](https://github.com/google/brax/tree/main/brax/training/agents/ppo) of PPO based on Flax so as to ensure interoperability with Brax environments. If you are searching for other JAX PPO repos - almost all are in Flax, which led me to create this one built in Equinox (although note Brax uses Flax internally).

### Jax debugging tips

- To find out recompiles: jax.config.update("jax_log_compiles", True)
- To find NaNs which would otherwise propogate silently: jax.config.update("jax_debug_nans", True)
- To prevent recompiles on unchanged code between runs: https://jax.readthedocs.io/en/latest/persistent_compilation_cache.html

# References

- [brax implementation](https://github.com/google/brax/tree/main/brax/training/agents/ppo)
- [cleanrl implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py)