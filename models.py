import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import stats

from typing import Callable, List
from dataclasses import dataclass


#### JOHN YOU ARE HERE - UNCERTAIN ABOUT THE RAW_STD COMPARED WITH BRAX - SEE _LOG_PROB_TEST.py

class PPOActorStochasticMLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable
    layer_sizes: List[int]
    action_distribution: dataclass = eqx.field(static=True)

    def __init__(self, key, layer_sizes, activation=jax.nn.silu, action_distribution=stats.NormalTanhDistribution()):
        jax.debug.print("INFO: initializing PPOActorStochasticMLP, ensure final layer size is 2x action dim for mean and std")
        keys = jr.split(key, num=len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features, out_features, key=keys[i])
            for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        ]
        self.activation = activation
        self.layer_sizes = layer_sizes
        self.action_distribution = action_distribution

    def __call__(self, key, x):
        logits = self.mlp_forward(x)
        mean, raw_std = jnp.split(logits, 2)
        action = self.action_distribution.sample(key, mean, raw_std)
        return action, mean

    def mlp_forward(self, x):
        for linear in self.layers[:-1]:
            x = self.activation(linear(x))
        logits = self.layers[-1](x)  # Output mean
        return logits
    
    def log_prob(self, x, action):
        logits = self.mlp_forward(x)
        mean, raw_std = jnp.split(logits, 2)
        return self.action_distribution.log_prob(
            loc=mean, 
            scale=raw_std,
            actions=action
        )

    def entropy(self, key, x):
        logits = self.mlp_forward(x)
        mean, raw_std = jnp.split(logits, 2)
        return self.action_distribution.entropy(
            key=key, 
            loc=mean, 
            scale=raw_std
        )
    

class PPOActorStochasticVec(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable
    raw_std: jnp.ndarray  # Standard deviation
    layer_sizes: List[int]
    action_distribution: dataclass = eqx.field(static=True)

    def __init__(self, key, layer_sizes, activation=jax.nn.relu, action_distribution=stats.NormalTanhDistribution()):
        keys = jr.split(key, num=len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features, out_features, key=keys[i])
            for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        ]
        self.activation = activation
        self.raw_std = jnp.zeros(layer_sizes[-1])
        self.layer_sizes = layer_sizes
        self.action_distribution = action_distribution

    def __call__(self, key, x):
        mean = self.forward_deterministic(x)
        action = self.action_distribution.sample(key, mean, self.raw_std)
        return action, mean
    
    def forward_deterministic(self, x):
        for linear in self.layers[:-1]:
            x = self.activation(linear(x))
        mean = self.layers[-1](x)  # Output mean
        return mean

    def log_prob(self, x, action):
        return self.action_distribution.log_prob(
            loc=self.forward_deterministic(x), 
            scale=self.raw_std, 
            actions=action
        )

    def entropy(self, key, x):
        return self.action_distribution.entropy(
            key=key, 
            loc=self.forward_deterministic(x), 
            scale=self.raw_std
        )
    

class PPOValueNetwork(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(self, key, layer_sizes, activation=jax.nn.relu):
        keys = jr.split(key, num=len(layer_sizes) - 1)
        self.layers = [
            eqx.nn.Linear(in_f, out_f, key=k)
            for in_f, out_f, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x.squeeze(-1)  # Output scalar value


if __name__ == "__main__":

    from jax import jit, vmap

    global_key = jr.PRNGKey(0)
    global_key, actor_key, critic_key, dataset_key, sample_key = jr.split(global_key, num=5)

    batch_size = 30
    num_actions = 2
    num_observations = 2

    actor = PPOStochasticActor(actor_key, [num_observations,10,10,num_actions])
    value = PPOValueNetwork(critic_key, [num_actions,10,10,1])

    x = jr.normal(dataset_key, shape=[batch_size,2])

    actor_call_jv = jit(vmap(actor.__call__))
    actor_log_prob_jv = jit(vmap(actor.log_prob))
    actor_entropy_jv = jit(vmap(actor.entropy))

    u = actor_call_jv(jr.split(sample_key, num=batch_size), x)
    lp = actor_log_prob_jv(x, u)
    e = actor_entropy_jv(jr.split(sample_key, num=batch_size), x)

    print('fin')

