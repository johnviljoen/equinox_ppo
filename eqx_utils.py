from jax import lax
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.internal as eqxi
from equinox._misc import default_floating_dtype

@eqx.filter_jit
def filter_scan(f, init, xs, length=None, reverse=False, unroll=1):
    # Partition the initial carry and sequence inputs into dynamic and static parts
    init_dynamic, init_static = eqx.partition(init, eqx.is_array)
    xs_dynamic, xs_static = eqx.partition(xs, eqx.is_array)

    # Define the scanned function, handling the combination and partitioning
    def scanned_fn(carry_dynamic, x_dynamic):
        # Combine dynamic and static parts for the carry and input
        carry = eqx.combine(carry_dynamic, init_static)
        x = eqx.combine(x_dynamic, xs_static)

        # Apply the original function
        out_carry, out_y = f(carry, x)

        # Partition the outputs into dynamic and static parts
        out_carry_dynamic, out_carry_static = eqx.partition(out_carry, eqx.is_array)
        out_y_dynamic, out_y_static = eqx.partition(out_y, eqx.is_array)

        # Return dynamic outputs and wrap static outputs using Static to prevent tracing
        return out_carry_dynamic, (out_y_dynamic, eqxi.Static((out_carry_static, out_y_static)))

    # Use lax.scan with the modified scanned function
    final_carry_dynamic, (ys_dynamic, static_out) = lax.scan(
        scanned_fn, init_dynamic, xs_dynamic, length=length, reverse=reverse, unroll=unroll
    )

    # Extract static outputs
    out_carry_static, ys_static = static_out.value

    # Combine dynamic and static parts of the outputs
    final_carry = eqx.combine(final_carry_dynamic, out_carry_static)
    ys = eqx.combine(ys_dynamic, ys_static)

    return final_carry, ys


def any_nan_in_pytree(tree):

    # Only look at leaves that are array like
    filter_array = eqx.filter(tree, eqx.is_array_like)

    # Apply jnp.isnan to each leaf of the pytree
    tree_isnan = jax.tree.map(jnp.isnan, filter_array)

    # Apply jnp.any to each leaf to reduce within each leaf
    tree_isnan_any = jax.tree.map(jnp.any, tree_isnan)

    # Now reduce the entire pytree using a logical OR to check if any leaf has a True value
    any_nan_in_pytree = jax.tree_util.tree_reduce(lambda x, y: x or y, tree_isnan_any, initializer=False)

    return any_nan_in_pytree


def lecun_normal_init(
    key, shape: tuple[int, ...], dtype, lim=None
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_dtype = jnp.finfo(dtype).dtype
        rkey, ikey = jr.split(key, 2)
        real = jr.normal(rkey, shape, real_dtype) * jnp.sqrt(1.0 / shape[0])
        imag = jr.normal(ikey, shape, real_dtype) * jnp.sqrt(1.0 / shape[0])
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jr.normal(key, shape, dtype) * jnp.sqrt(1.0 / shape[0])


class LecunNormalInitLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
            self,
            in_features,
            out_features,
            use_bias: bool = True,
            dtype=None,
            *,
            key
        ):
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jr.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / jnp.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = lecun_normal_init(wkey, wshape, dtype, lim)
        bshape = (out_features_,)
        self.bias = lecun_normal_init(bkey, bshape, dtype, lim) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    @jax.named_scope("LecunNormalInitLinear")
    def __call__(self, x, *, key = None):
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x
