"""
- RunningMeanStd: eqx.Module
- NormalDistribution: dataclass
- NormalTanhDistribution: dataclass
"""

import jax
import jax.numpy as jnp
import dataclasses
import equinox as eqx


@dataclasses.dataclass
class NormalDistribution:
    loc: jnp.array
    scale: jnp.array

    def sample(self, key):
        return jax.random.normal(key, shape=self.loc.shape) * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * jnp.ones_like(self.loc)


@dataclasses.dataclass
class NormalTanhDistribution:
    _min_std: float = 0.001
    _var_scale: float = 1.0

    def create_dist(self, loc, scale):
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return NormalDistribution(loc=loc, scale=scale)
    
    def sample_no_postprocess(self, key, loc, scale):
        return self.create_dist(loc, scale).sample(key=key)
    
    def sample(self, key, loc, scale):
        return jnp.tanh(self.sample_no_postprocess(key, loc, scale))

    def mode(self, loc, scale):
        return jnp.tanh(self.create_dist(loc, scale).mode())

    # the forward log det of the jacobian of the tanh bijector
    def tanh_log_det_jac(self, x):
        return 2. * (jnp.log(2.) - x - jax.nn.softplus(-2. * x))

    def log_prob(self, loc, scale, actions):
        dist = self.create_dist(loc, scale)
        log_probs = dist.log_prob(actions)
        log_probs -= self.tanh_log_det_jac(actions)
        # if self._event_ndims == 1:
        log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
        return log_probs

    def entropy(self, key, loc, scale):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(loc, scale)
        entropy = dist.entropy()
        entropy += self.tanh_log_det_jac(dist.sample(key=key))
        # if self._event_ndims == 1:
        entropy = jnp.sum(entropy, axis=-1)
        return entropy


class RunningMeanStd(eqx.Module):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    epsilon: float = 1e-4

    def update(self, arr) -> None:
        arr = jax.lax.stop_gradient(arr)
        batch_mean = jnp.mean(arr, axis=tuple(range(arr.ndim - 1)))
        batch_var = jnp.var(arr, axis=tuple(range(arr.ndim - 1)))
        batch_count = jnp.prod(jnp.array(arr.shape[:-1])) # arr.shape[0] # enforcing int is problematic in jax tracing
        return self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + delta ** 2
            * self.count
            * batch_count
            / tot_count
        )
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        return dataclasses.replace(
            self,
            mean=new_mean,
            var=new_var,
            count=new_count
        )
    
    def normalize(self, arr):
        return (arr - self.mean) / jnp.sqrt(self.var + 1e-5)

    def denormalize(self, arr):
        return arr * jnp.sqrt(self.var + 1e-5) + self.mean