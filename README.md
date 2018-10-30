# tensorshaper
This is set of utilities to perform very common array reshaping tasks on tf.Tensors and np.ndarrays. It obviates the need for lots of hardcoded or programmatic dimensionality and shape reasoning. It's frustrating that reshaping and transpose operations (e.g. `tf.reshape`, `np.reshape`, `tf.transpose`, and `np.transpose`) generally require full knowledge of the number of dimensions, and size of each dimension in an array. This set of utilities (in a single file) removes that requirement -- you only need to know things about the axes you care about.

The core functionality this provides is <strong>Dimensionality-agnostic axis permutation and reshaping</strong>

# Axis swapping
Say you want to permute an array `arr` of size `(A, B, C, D)` by swapping the `B` axis with the `D` axis. Doing this with `tf.reshape` or `np.reshape` in one call is generally not possible (a `reshape` generally would not preseve axis identities), and you need to either hardcode or programmatically extract the values of `A, B, C` and `D` in order to do so. Instead, use `ts.swap_axis(arr, 1, 3)`. 

# Axis packing 
Say you want to reshape `arr` (still of shape `(A, B, C, D)`) to `(AB, C, D)`. `tf.reshape(arr, (A*B, C, d)` would work, or you can call `ts.pack_to_axis(arr, 0, 1)`, which requires you only to know which axis you want to pack, instead of the size of each dimension and the dimensioality of the array. Let's say the array is now `(A, B, C, D, ...)`, where `...` indicates some unknown number (e.g. it's dynamic) of other axes. Reshaping this is going to be annoying. Or, you can call `ts.pack_to_axis(arr, 0, 1)`, the same call as before. Or even just call `ts.frontpack(arr)`. 

Let's take another example, where I need to reshape my inputs `(?_1, ?_2, ?_3, D)` to some function `func` that requires a matrix (`2d-array`), for instance, an MLP, that expects features of size `D`. I'll do this: `ts.frontpack(ts.frontpack(arr))`, and receive an array of size `(?_1*?_2*?_3, D)`, and never have to retrieve the values of `?_1, ?_2`, or `?_3`.

# Axis pop-insertion
Let's now say you have an array of dimensionality `>=3`: `arr.shape = (A, B, ..., ?)`. How can we reshape it so axis `B` is at the end, `(A, ..., B, ?)`? Just do this: `ts.popinsert_axes(arr, 1, -1)`. 

# Library agnostic
The only assumptions is that the library (an argument that defaults to `tensorflow`) provides the `.reshape` and `.transpose` functions. So it works with both `numpy` and `tensorflow`, and can be extended to other libraries if you add a line to the function that extracts the shape of an array.

With these utilities, I almost never use `{tf,np}.{transpose,reshape}` anymore. N.B. that I extracted these utilities from a set of larger utilities, which were tested within the context of those, but I have not tested them after moving them into this new standalone module.
