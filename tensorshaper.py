
import numpy as np
import tensorflow as tf

__doc__ = """

Set of utilities to perform common array reshaping tasks on tf.Tensors and np.ndarrays
Author: Nicholas Rhinehart

Primary functions:

swap_axes(...): swap the axes of an array
popinsert_axes(...): move one axis before another
pack_to_axis(...): reshape the array to pack one axis into another
unpack_axis(...): expand the array to unpack data from an axis
"""

def tf_shape(inp):
    """Get the shape of a Tensor

    :param inp: tf.Tensor
    :returns: shape of tensor
    :rtype: tuple
    """

    return tuple(inp.get_shape().as_list())

def np_tf_get_shape(inp):
    """Get the shape of the array

    :param inp: ndarray or tf.Tensor
    :returns: shape of input
    :rtype: tuple 
    """
    if isinstance(inp, np.ndarray):
        return inp.shape
    else:
        return tf_shape(inp)

def rank(inp):
    """Returns a python int of the rank / number of dimensions of a tf.Tensor.
    Note tf.rank is similar, except it returns a tensor, which we need to
    evaluate if we want to use its value...

    :param inp: tf.Tensor or np.ndarray
    :returns: r \in \mathbb N_0
    :rtype: int

    """
    return len(np_tf_get_shape(inp))

def get_swapping_permutation(r, idx0, idx1):
    """Build a permutation that swaps two indices

    :param r: rank of the permutation
    :param idx0: source index
    :param idx1: target index
    :returns: sequence of permutation indices
    :rtype: list

    """
    assert(0 <= idx0)
    assert(0 <= idx1)
    permute = list(range(r))
    permute[idx0] = idx1
    permute[idx1] = idx0
    return permute

def pidx(r, idx):
    """Get a nonnegative index for the given rank and index

    :param r: maximum rank int \in N_+
    :param idx: idx \in [-r, r-1]
    :returns: idx \in [0, r-1]
    """
    
    if r < 1: raise ValueError("R is not positive")
    if not (-r <= idx <= r - 1): raise IndexError("Index is OOB")
    if idx < 0: return r + idx
    else: return idx

def get_popinsert_permutation(r, idx0, idx1):
    """Build a permutation that cycles a subcycle to put idx0 at idx

    :param r: rank of array
    :param idx0: int \in [-r, r - 1], 
    :param idx1: int \in [-r, r - 1], 
    :returns: sequence of permutation indices
    :rtype: list

    """
    idx0 = pidx(r, idx0)
    idx1 = pidx(r, idx1)
    permute = list(range(r))
    permute.insert(idx1, permute.pop(idx0))
    return permute

def swap_axes(arr, axis0, axis1, lib=tf):
    """Transpose an array to swap axes. Doesn't require full knowledge of the dimensionality of array.

    :param arr: array object (e.g. tf.Tensor)
    :param axis0: int \in [-rank, rank - 1], axis to swap with axis1
    :param axis1: int \in [-rank, rank - 1], axis to swap with axis0
    :param lib: library to call transpose with (e.g. tf or np)
    :returns: array object
    """
    r = rank(arr)
    axis0 = pidx(r, axis0)
    axis1 = pidx(r, axis1)
    assert(axis0 <= r - 1), "rank oob"
    assert(axis1 <= r - 1), "rank oob"
    return lib.transpose(arr, get_swapping_permutation(r, axis0, axis1))

def popinsert_axes(arr, axis0, axis1, lib=tf):
    """Move axis from one index to before another. 

    e.g. 
     popinsert_axes((A, B, C, D, ..., ?), 1, -1) -> (

    :param arr: 
    :param axis0: int \in [-rank, rank - 1], source axis
    :param axis1: int \in [-rank, rank - 1], axis before which axis0 will be placed
    :returns: popinserted array
    """
    r = rank(arr)
    assert(-r <= axis0 <= r - 1), "rank oob"
    assert(-r <= axis1 <= r - 1), "rank oob"
    return lib.transpose(arr, get_popinsert_permutation(r, axis0, axis1))

def pack_to_axis(arr, axis_source, axis_target, lib=tf):
    """Pack the source axis with the target axis with the source serving as the outer
    axis (rows) and target serving as inner (cols)

    e.g.
     pack_to_axis((A, B, C, D), 1, 3)) -> (A, C, BD)
     pack_to_axis((A, B, C, D), 2, 0)) -> (CA, B, D)

    :param arr: array
    :param axis_source: index of outer (slower raveling) axis
    :param axis_target: index of inner (faster raveling) axis
    :returns: packed array
    """
    
    r = rank(arr)
    s = np_tf_get_shape(arr)
    assert(-r <= axis_source <= r - 1), "axis source oob"
    assert(axis_target != axis_source), "identical axes"
    assert(-r + 1 <= axis_target <= r - 1), "axis target oob"
    if axis_target < 0: axis_target = r - abs(axis_target)
    fwd = axis_source < axis_target
    pretarget = axis_target - fwd
    cycled = popinsert_axes(arr, axis_source, pretarget, lib=lib)
    permuter = get_popinsert_permutation(r, axis_source, pretarget)
    s_permute = np.asarray(s)[permuter].tolist()
    del s_permute[pretarget:pretarget + 2]
    s_permute.insert(pretarget, -1)
    return lib.reshape(cycled, s_permute)

def unpack_axis(arr, axis_source, axis1_size, lib=tf):
    """Increases dimensionality of array by unpacking an axis to a size 
    
    e.g.
     unpack_axis((A, B, CD, E), 2, D) -> (A, B, C, D, E)

    :param arr: array
    :param axis_source: index of source axis
    :param axis1_size: size of the latter axis (in the example, D)
    :returns: array unpacked
    """
    r = rank(arr)
    axis_source = pidx(r, axis_source)
    s = list(np_tf_get_shape(arr))
    assert(0 <= axis_source <= r - 1), "axis source oob"
    del s[axis_source]
    s_unpack = s[:axis_source] + [-1, axis1_size] + s[axis_source:]
    return lib.reshape(arr, s_unpack)

def frontpack(arr, lib=tf):
    """ (A, B, ...) -> (AB, ...) """
    return pack_to_axis(arr, 0, 1, lib=lib)

def frontswap(arr, lib=tf):
    """ (A, B, ...) -> (B, A, ...)"""
    return swap_axes(arr, 0, 1, lib=lib)

def backpack(arr, lib=tf):
    """ (..., A, B) -> (..., AB)"""
    return pack_to_axis(arr, -2, -1, lib=lib)

def backswap(arr, lib=tf):
    """ (..., A, B) -> (..., B, A)"""
    return swap_axes(arr, -2, -1, lib=lib)
