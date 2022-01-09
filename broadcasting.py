from typing import Any, Callable

import numpy as np

# Type aliases
Key = Callable[[Any], Any]
Array = np.ndarray
Shape = tuple[int, ...]


def minmax(arg: Any, arg2: Any, *, key: Key) -> tuple[Any, Any]:
    """Return the smaller and larger item of a pair of items."""
    return min((arg, arg2), key=key), max((arg, arg2), key=key)


def match_dimensions(arr: Array, arr2: Array) -> tuple[Array, Array]:
    """
    Return resized arrays with their dimensions matched for broadcasting.

    1's are prepended to the shape of the array with fewer axes until it
    has the same amount of axes as the other array.
    """
    if len(arr.shape) == len(arr2.shape):
        return arr, arr2  # leave them unchanged

    to_prepend, larger = minmax(arr, arr2, key=lambda x: len(x.shape))
    ts, ls = to_prepend.shape, larger.shape

    difference = len(ls) - len(ts)
    return np.resize(to_prepend, (1,) * difference + ts), larger


def broadcasted_shape(shape: Shape, shape2: Shape) -> Shape:
    """
    Return the shape to which two shapes will be broadcasted.

    Two shapes match if all their dimensions match. Two dimensions match
    if they're equal OR if one of them is 1. If the input shapes match,
    the broadcasted shape will consist of axes that are not 1 along each
    axis of the inputs. Otherwise, a ValueError exception is raised.
    """
    new_shape = []

    # Checking axes starting from the rightmost one to be consistent
    # with actual numpy implementation
    for dim, dim2 in zip(shape[::-1], shape2[::-1]):
        if dim == dim2 or dim2 == 1:
            new_shape.insert(0, dim)
        elif dim == 1:
            new_shape.insert(0, dim2)
        else:
            raise ValueError(
                f"operands could not be broadcast together with"
                f"shapes {shape} and {shape2}"
            )

    return tuple(new_shape)


def get_item(arr: Array, indices: tuple[int, ...]) -> Any:
    """
    Get an item of a ndarray from a tuple of indices.

    Used for the broadcasting implementation. The indices are modified
    such that an index in the corresponding position of an axis that is 1
    is turned to 0, thereby simulating the "stretching" effect.
    """
    modified_indices = tuple(
        0 if dim == 1 else index
        for dim, index in zip(arr.shape, indices)
    )
    return arr[modified_indices]


def add(arr: Array, arr2: Array) -> Array:
    """Add two arrays element-wise."""
    arr, arr2 = match_dimensions(arr, arr2)

    output_shape = broadcasted_shape(arr.shape, arr2.shape)
    out = np.empty(output_shape, dtype=int)

    for indices in np.ndindex(*output_shape):
        # The "stretching" effect is simulated by re-using items on
        # specific axes. For example, with a 3-D array and three nested for
        # loops, arr[0, j, k] (no i) would mean to repeat the j-k plane
        # regardless of the i axis (stretching it over the whole 3-D array).
        out[indices] = get_item(arr, indices) + get_item(arr2, indices)

    return out


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])[:, np.newaxis]
    b = np.array([[1, 2, 3, 4, 5, 6]])
    c = np.array([1, 2, 3, 4, 5, 6])

    print(add(add(a, b), c))
