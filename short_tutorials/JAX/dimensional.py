from typing import Union, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from dataclasses import replace

import quax


class Dimension:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return False


class Dimensional(quax.ArrayValue):
    array: ArrayLike = eqx.field(converter=jnp.asarray)
    dimensions: dict[Dimension, int] = eqx.field(
        static=True, converter=lambda x: {x: 1} if isinstance(x, Dimension) else x
    )

    def aval(self):
        return jax.core.ShapedArray(jnp.shape(self.array), jnp.result_type(self.array))

    def materialise(self):
        raise ValueError("Refusing to materialize Dimensional array.")

    def __mul__(self, other):
        return quax.quaxify(jnp.multiply)(self, other)

    def __rmul__(self, other):
        return quax.quaxify(jnp.multiply)(other, self)

    def __sub__(self, other):
        return quax.quaxify(jnp.subtract)(self, other)

    def __neg__(self):
        return quax.quaxify(jnp.negative)(self)

    def __truediv__(self, other):
        return quax.quaxify(jnp.divide)(self, other)

    def __pow__(self, other):
        return quax.quaxify(jnp.power)(self, other)

    def __add__(self, other):
        return quax.quaxify(jnp.add)(self, other)

    def __repr__(self):
        return f"Dimensional(array={self.array}, dimensions={self.dimensions})"


##############################################################################


@quax.register(jax.lax.add_p)
def _(x: Dimensional, y: Dimensional):
    if x.dimensions != y.dimensions:
        raise ValueError(
            f"Cannot add two arrays with dimensions {x.dimensions} and {y.dimensions}."
        )
    return Dimensional(x.array + y.array, x.dimensions)


@quax.register(jax.lax.mul_p)
def _(x: Dimensional, y: Dimensional):
    dimensions = x.dimensions.copy()
    for k, v in y.dimensions.items():
        if k in dimensions:
            dimensions[k] += v
        else:
            dimensions[k] = v
    return Dimensional(x.array * y.array, dimensions)


@quax.register(jax.lax.mul_p)
def _(x: ArrayLike, y: Dimensional):
    return Dimensional(x * y.array, y.dimensions)


@quax.register(jax.lax.mul_p)
def _(x: Dimensional, y: ArrayLike):
    return Dimensional(x.array * y, x.dimensions)


@quax.register(jax.lax.integer_pow_p)
def _(x: Dimensional, *, y: int):
    dimensions = {k: v * y for k, v in x.dimensions.items()}
    return Dimensional(x.array, dimensions)


@quax.register(jax.lax.square_p)
def _square_p(x: Dimensional) -> Dimensional:
    return replace(
        x,
        array=jnp.square(x.array),
        dimensions={k: v * 2 for k, v in x.dimensions.items()},
    )


@quax.register(jax.lax.sqrt_p)
def _sqrt_p(x: Dimensional) -> Dimensional:
    return replace(
        x,
        array=jnp.sqrt(x.array),
        dimensions={k: v / 2 for k, v in x.dimensions.items()},
    )


@quax.register(jax.lax.convert_element_type_p)
def _convert_element_type_p(
    operand: Dimensional,
    *,
    new_dtype: Any,
    weak_type: Any,
) -> Dimensional:
    """Convert the element type of a quantity."""
    # TODO: examples
    del weak_type
    return replace(
        operand, array=jax.lax.convert_element_type(operand.array, new_dtype)
    )


@quax.register(jax.lax.sub_p)
def _(x: Dimensional, y: Dimensional):
    if x.dimensions != y.dimensions:
        raise ValueError(
            f"Cannot subtract two arrays with dimensions {x.dimensions} and {y.dimensions}."
        )
    return Dimensional(x.array - y.array, x.dimensions)


@quax.register(jax.lax.add_p)
def _(x: Dimensional, y: ArrayLike) -> ArrayLike:
    if not all(v == 0 for v in x.dimensions.values()):
        raise ValueError(
            f"Cannot add a to a dimensionless array to an array with dimensions {y.dimensions} "
        )
    return x.array + y


@quax.register(jax.lax.div_p)
def _(x: Dimensional, y: Dimensional):
    dimensions = x.dimensions.copy()
    for k, v in y.dimensions.items():
        if k in dimensions:
            dimensions[k] -= v
        else:
            dimensions[k] = v
    return Dimensional(x.array * y.array, dimensions)


from jax._src.ad_util import add_any_p


@quax.register(add_any_p)
def _add_any_p(x: Dimensional, y: Dimensional) -> Dimensional:
    """Add two Dimensional using the ``jax._src.ad_util.add_any_p``."""
    return Dimensional(add_any_p.bind(x.array, y.array), x.dimensions)


@quax.register(jax.lax.neg_p)
def _neg_p(x: Dimensional) -> Dimensional:
    return replace(x, array=jax.lax.neg(x.array))


@quax.register(jax.lax.concatenate_p)
def _concatenate_p_aq(*operands: Dimensional, dimension: Any) -> Dimensional:
    operand0 = operands[0]
    if not all(op.dimensions == operand0.dimensions for op in operands):
        msg = f"Cannot concatenate arrays with different dimensions: {[op.dimensions for op in operands]}"
        raise ValueError(msg)
    return replace(
        operand0,
        array=jax.lax.concatenate([op.array for op in operands], dimension=dimension),
    )
