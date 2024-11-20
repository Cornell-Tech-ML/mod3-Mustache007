# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## Task 3.1 & 3.2
### Diagnostics Output

```console
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py (164) 
-----------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                            | 
        out: Storage,                                                                    | 
        out_shape: Shape,                                                                | 
        out_strides: Strides,                                                            | 
        in_storage: Storage,                                                             | 
        in_shape: Shape,                                                                 | 
        in_strides: Strides,                                                             | 
    ) -> None:                                                                           | 
        # TODO: Implement for Task 3.1.                                                  | 
        # Check if input and output tensors have same shape and strides for fast path    | 
        if (np.array_equal(out_strides, in_strides)                                      | 
            and np.array_equal(out_shape, in_shape)):                                    | 
            # Fast path - directly map elements                                          | 
            for idx in prange(out.size):-------------------------------------------------| #0
                out[idx] = fn(in_storage[idx])                                           | 
        else:                                                                            | 
            # Slow path - handle broadcasting                                            | 
            for elem_idx in prange(out.size):--------------------------------------------| #1
                output_coords = np.empty(MAX_DIMS, np.int32)                             | 
                input_coords = np.empty(MAX_DIMS, np.int32)                              | 
                                                                                         | 
                to_index(elem_idx, out_shape, output_coords)                             | 
                                                                                         | 
                broadcast_index(output_coords, out_shape, in_shape, input_coords)        | 
                                                                                         | 
                input_pos = index_to_position(input_coords, in_strides)                  | 
                out_pos = index_to_position(output_coords, out_strides)                  | 
                                                                                         | 
                out[out_pos] = fn(in_storage[input_pos])                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(182) is hoisted out of the parallel loop labelled #1 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: output_coords = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(183) is hoisted out of the parallel loop labelled #1 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: input_coords = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(220)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py (220) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        if (                                                               | 
            np.array_equal(out_strides, a_strides)                         | 
            and np.array_equal(out_strides, b_strides)                     | 
            and np.array_equal(out_shape, a_shape)                         | 
            and np.array_equal(out_shape, b_shape)                         | 
        ):                                                                 | 
            for i in prange(out.size):-------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(out.size):-------------------------------------| #3
                out_index = np.empty(MAX_DIMS, np.int32)                   | 
                a_index = np.empty(MAX_DIMS, np.int32)                     | 
                b_index = np.empty(MAX_DIMS, np.int32)                     | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                a_pos = index_to_position(a_index, a_strides)              | 
                b_pos = index_to_position(b_index, b_strides)              | 
                out_pos = index_to_position(out_index, out_strides)        | 
                out[out_pos] = fn(                                         | 
                    a_storage[a_pos],                                      | 
                    b_storage[b_pos],                                      | 
                )                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(242) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(243) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(244) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(279)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py (279) 
---------------------------------------------------------------|loop #ID
    def _reduce(                                               | 
        out: Storage,                                          | 
        out_shape: Shape,                                      | 
        out_strides: Strides,                                  | 
        a_storage: Storage,                                    | 
        a_shape: Shape,                                        | 
        a_strides: Strides,                                    | 
        reduce_dim: int,                                       | 
    ) -> None:                                                 | 
        # TODO: Implement for Task 3.1.                        | 
        reduce_size = a_shape[reduce_dim]                      | 
        for i in prange(out.size):-----------------------------| #4
            index = np.empty(MAX_DIMS, np.int32)               | 
            to_index(i, out_shape, index)                      | 
            out_pos = index_to_position(index, out_strides)    | 
            index[reduce_dim] = 0                              | 
            pos = index_to_position(index, a_strides)          | 
            acc = a_storage[pos]                               | 
            for j in range(1, reduce_size):                    | 
                index[reduce_dim] = j                          | 
                pos = index_to_position(index, a_strides)      | 
                acc = fn(acc, a_storage[pos])                  | 
            out[out_pos] = acc                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(291) is hoisted out of the parallel loop labelled #4 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py 
(306)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/huziyao/Desktop/MLE/workspace/mod3-Mustache007/minitorch/fast_ops.py (306) 
----------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                        | 
    out: Storage,                                                                                   | 
    out_shape: Shape,                                                                               | 
    out_strides: Strides,                                                                           | 
    a_storage: Storage,                                                                             | 
    a_shape: Shape,                                                                                 | 
    a_strides: Strides,                                                                             | 
    b_storage: Storage,                                                                             | 
    b_shape: Shape,                                                                                 | 
    b_strides: Strides,                                                                             | 
) -> None:                                                                                          | 
    """NUMBA tensor matrix multiply function.                                                       | 
                                                                                                    | 
    Should work for any tensor shapes that broadcast as long as                                     | 
                                                                                                    | 
    ```                                                                                             | 
    assert a_shape[-1] == b_shape[-2]                                                               | 
    ```                                                                                             | 
                                                                                                    | 
    Optimizations:                                                                                  | 
                                                                                                    | 
    * Outer loop in parallel                                                                        | 
    * No index buffers or function calls                                                            | 
    * Inner loop should have no global writes, 1 multiply.                                          | 
                                                                                                    | 
                                                                                                    | 
    Args:                                                                                           | 
    ----                                                                                            | 
        out (Storage): storage for `out` tensor                                                     | 
        out_shape (Shape): shape for `out` tensor                                                   | 
        out_strides (Strides): strides for `out` tensor                                             | 
        a_storage (Storage): storage for `a` tensor                                                 | 
        a_shape (Shape): shape for `a` tensor                                                       | 
        a_strides (Strides): strides for `a` tensor                                                 | 
        b_storage (Storage): storage for `b` tensor                                                 | 
        b_shape (Shape): shape for `b` tensor                                                       | 
        b_strides (Strides): strides for `b` tensor                                                 | 
                                                                                                    | 
    Returns:                                                                                        | 
    -------                                                                                         | 
        None : Fills in `out`                                                                       | 
                                                                                                    | 
    """                                                                                             | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                          | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                          | 
                                                                                                    | 
    # TODO: Implement for Task 3.2.                                                                 | 
    # Initialize local stride variables to avoid global reads                                       | 
    batch_stride_out = out_strides[0] if out_shape[0] > 1 else 0                                    | 
    row_stride_out = out_strides[-2]                                                                | 
    col_stride_out = out_strides[-1]                                                                | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                          | 
    row_stride_a = a_strides[-2]                                                                    | 
    col_stride_a = a_strides[-1]                                                                    | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                          | 
    col_stride_b = b_strides[-1]                                                                    | 
    row_stride_b = b_strides[-2]                                                                    | 
                                                                                                    | 
    # Perform the matrix multiplication                                                             | 
    for batch in prange(out_shape[0]):--------------------------------------------------------------| #7
        for row in prange(out_shape[-2]):-----------------------------------------------------------| #6
            for col in prange(out_shape[-1]):-------------------------------------------------------| #5
                # Calculate linear index for out tensor                                             | 
                out_idx = batch * batch_stride_out + row * row_stride_out + col * col_stride_out    | 
                                                                                                    | 
                # Initialize starting indices for a and b tensors                                   | 
                a_idx = batch * a_batch_stride + row * row_stride_a                                 | 
                b_idx = batch * b_batch_stride + col * col_stride_b                                 | 
                                                                                                    | 
                # Accumulate the result for out tensor                                              | 
                result = 0                                                                          | 
                for k in range(a_shape[-1]):                                                        | 
                    result += a_storage[a_idx] * b_storage[b_idx]                                   | 
                    a_idx += col_stride_a  # move to next column in a                               | 
                    b_idx += row_stride_b  # move to next row in b                                  | 
                                                                                                    | 
                out[out_idx] = result                                                               | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #7, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--7 is a parallel loop
   +--6 --> rewritten as a serial loop
      +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--7 (parallel)
   +--6 (parallel)
      +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--7 (parallel)
   +--6 (serial)
      +--5 (serial)


 
Parallel region 0 (loop #7) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#7).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
