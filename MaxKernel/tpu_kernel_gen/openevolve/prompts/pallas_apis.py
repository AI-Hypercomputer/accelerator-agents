PROMPT = """
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.pallas_call

**Signature**:
`jax.experimental.pallas.pallas_call(kernel: 'Callable[..., None]', out_shape: 'Any', *, grid_spec: 'GridSpec | None' = None, grid: 'TupleGrid' = (), in_specs: 'BlockSpecTree' = NoBlockSpec, out_specs: 'BlockSpecTree' = NoBlockSpec, scratch_shapes: 'ScratchShapeTree' = (), input_output_aliases: 'Mapping[int, int]' = {}, debug: 'bool' = False, interpret: 'Any' = False, name: 'str | None' = None, compiler_params: 'Mapping[Backend, pallas_core.CompilerParams] | pallas_core.CompilerParams | None' = None, cost_estimate: 'CostEstimate | None' = None, backend: 'Backend | None' = None, metadata: 'dict[str, str] | None' = None) -> 'Callable[..., Any]'`

**Description**:
Invokes a Pallas kernel on some inputs.  See `Pallas Quickstart
<https://docs.jax.dev/en/latest/pallas/quickstart.html>`_.  Args:   kernel: the
kernel function, that receives a Ref for each input and output.     The shape of
the Refs are given by the ``block_shape`` in the     corresponding ``in_specs``
and ``out_specs``.   out_shape: a PyTree of :class:`jax.ShapeDtypeStruct`
describing the shape     and dtypes of the outputs.   grid_spec: An alternative
way to specify ``grid``, ``in_specs``,     ``out_specs`` and ``scratch_shapes``.
If given, those other parameters     must not be also given.   grid: the
iteration space, as a tuple of integers. The kernel is executed     as many
times as ``prod(grid)``.     See details at :ref:`pallas_grid`.   in_specs: a
PyTree of :class:`jax.experimental.pallas.BlockSpec` with     a structure
matching that of the positional arguments.     The default value for
``in_specs`` specifies the whole array for all     inputs, e.g., as
``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.     See details at
:ref:`pallas_blockspec`.   out_specs: a PyTree of
:class:`jax.experimental.pallas.BlockSpec` with     a structure matching that of
the outputs.     The default value for ``out_specs`` specifies the whole array,
e.g., as ``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.     See
details at :ref:`pallas_blockspec`.   scratch_shapes: a PyTree of backend-
specific temporary objects required     by the kernel, such as temporary
buffers, synchronization primitives,     etc.   input_output_aliases: a
dictionary mapping the index of some inputs to     the index of the output that
aliases them. These indices are in the     flattened inputs and outputs.
debug: if True, Pallas prints various intermediate forms of the kernel     as it
is being processed.   interpret: runs the ``pallas_call`` as a ``jax.jit`` of a
scan over the     grid whose body is the kernel lowered as a JAX function. This
does not     require a TPU or a GPU, and is the only way to run Pallas kernels
on CPU.     This is useful for debugging.   name: if present, specifies the name
to use for this kernel call in     debugging and error messages. To this name we
append the file and line     where the kernel function is defined, .e.g: `{name}
for kernel function     {kernel_name} at {file}:{line}`. If missing, then we use
`{kernel_name} at     {file}:{line}`.   compiler_params: Optional compiler
parameters. The value should either be a     backend-specific dataclass
(:class:`jax.experimental.pallas.tpu.CompilerParams`,
:class:`jax.experimental.pallas.triton.CompilerParams`,
:class:`jax.experimental.pallas.mosaic_gpu.CompilerParams`) or a dict
mapping backend name to the corresponding platform-specific dataclass.
backend: Optional string literal one of  ``"mosaic_tpu"``, ``"triton"`` or
``"mosaic_gpu"`` determining the backend to be used. None means let Pallas
decide.   metadata: Optional dictionary of information about the kernel that
will be     serialized as JSON in the HLO. Can be used for debugging and
analysis.  Returns:   A function that can be called on a number of positional
array arguments to   invoke the Pallas kernel.

**Parameters**:
  - **kernel**: the kernel function, that receives a Ref for each input and output.
The shape of the Refs are given by the ``block_shape`` in the
corresponding ``in_specs`` and ``out_specs``.
  - **out_shape**: a PyTree of :class:`jax.ShapeDtypeStruct` describing the shape
and dtypes of the outputs.
  - **grid_spec**: An alternative way to specify ``grid``, ``in_specs``,
``out_specs`` and ``scratch_shapes``. If given, those other parameters
must not be also given.
  - **grid**: the iteration space, as a tuple of integers. The kernel is executed
as many times as ``prod(grid)``.
See details at :ref:`pallas_grid`.
  - **in_specs**: a PyTree of :class:`jax.experimental.pallas.BlockSpec` with
a structure matching that of the positional arguments.
The default value for ``in_specs`` specifies the whole array for all
inputs, e.g., as ``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.
See details at :ref:`pallas_blockspec`.
  - **out_specs**: a PyTree of :class:`jax.experimental.pallas.BlockSpec` with
a structure matching that of the outputs.
The default value for ``out_specs`` specifies the whole array,
e.g., as ``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.
See details at :ref:`pallas_blockspec`.
  - **scratch_shapes**: a PyTree of backend-specific temporary objects required
by the kernel, such as temporary buffers, synchronization primitives,
etc.
  - **input_output_aliases**: a dictionary mapping the index of some inputs to
the index of the output that aliases them. These indices are in the
flattened inputs and outputs.
  - **debug**: if True, Pallas prints various intermediate forms of the kernel
as it is being processed.
  - **interpret**: runs the ``pallas_call`` as a ``jax.jit`` of a scan over the
grid whose body is the kernel lowered as a JAX function. This does not
require a TPU or a GPU, and is the only way to run Pallas kernels on CPU.
This is useful for debugging.
  - **name**: if present, specifies the name to use for this kernel call in
debugging and error messages. To this name we append the file and line
where the kernel function is defined, .e.g: `{name} for kernel function
{kernel_name} at {file}:{line}`. If missing, then we use `{kernel_name} at
{file}:{line}`.
  - **compiler_params**: Optional compiler parameters. The value should either be a
backend-specific dataclass
(:class:`jax.experimental.pallas.tpu.CompilerParams`,
:class:`jax.experimental.pallas.triton.CompilerParams`,
:class:`jax.experimental.pallas.mosaic_gpu.CompilerParams`) or a dict
mapping backend name to the corresponding platform-specific dataclass.
  - **backend**: Optional string literal one of  ``"mosaic_tpu"``, ``"triton"`` or
``"mosaic_gpu"`` determining the backend to be used. None means let Pallas
decide.
  - **metadata**: Optional dictionary of information about the kernel that will be
serialized as JSON in the HLO. Can be used for debugging and analysis.

**Returns**:
  A function that can be called on a number of positional array arguments to
invoke the Pallas kernel.
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.BlockSpec

**Signature**:
`jax.experimental.pallas.BlockSpec(block_shape: 'Sequence[BlockDim | int | None] | None' = None, index_map: 'Callable[..., Any] | None' = None, indexing_mode: 'Any | None' = None, pipeline_mode: 'Buffered | None' = None, *, memory_space: 'Any | None' = None) -> None`

**Description**:
Specifies how an array should be sliced for each invocation of a kernel.  The
`block_shape` is a sequence of `int | None`s, or `BlockDim` types (e.g.
`pl.Element`, `pl.Squeezed`, `pl.Blocked`, `pl.BoundedSlice`). Each of these
types specify the size of the block dimension. `None` is used to specify a
dimension that is squeezed out of the kernel. The `BlockDim` types allow for
more fine-grained control over the indexing of the dimension. The `index_map`
needs to return a tuple of the same length as `block_shape`, which each entry
depending on the type of `BlockDim`.  See :ref:`pallas_blockspec` and the
individual `BlockDim` type docstrings for more details.

**Attributes**:
  - `block_shape`
  - `index_map`
  - `indexing_mode`
  - `memory_space`
  - `pipeline_mode`

**Methods**:
  - `__eq__`
  - `__init__`
  - `__post_init__`
  - `__replace__`
  - `__repr__`
  - `replace`
  - `to_block_mapping`
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.program_id

**Signature**:
`jax.experimental.pallas.program_id(axis: 'int') -> 'jax.Array'`

**Description**:
Returns the kernel execution position along the given axis of the grid.  For
example, with a 2D `grid` in the kernel execution corresponding to the grid
coordinates `(1, 2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)`
returns `2`.  The returned value is an array of shape `()` and dtype `int32`.
Args:   axis: the axis of the grid along which to count the program.

**Parameters**:
  - **axis**: the axis of the grid along which to count the program.
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.when

**Signature**:
`jax.experimental.pallas.when(condition)`
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.num_programs

**Signature**:
`jax.experimental.pallas.num_programs(axis: 'int') -> 'int | jax.Array'`

**Description**:
Returns the size of the grid along the given axis.
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.Element

**Signature**:
`jax.experimental.pallas.Element(block_size: 'int', padding: 'tuple[int, int]' = (0, 0)) -> None`

**Description**:
Use to index an array using an elementwise start index.

**Attributes**:
  - `padding`

**Methods**:
  - `__delattr__`
  - `__eq__`
  - `__hash__`
  - `__init__`
  - `__replace__`
  - `__repr__`
  - `__setattr__`
  - `__str__`
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.Blocked

**Signature**:
`jax.experimental.pallas.Blocked(block_size: 'int') -> None`

**Description**:
The default BlockShape type.

**Methods**:
  - `__delattr__`
  - `__eq__`
  - `__hash__`
  - `__init__`
  - `__replace__`
  - `__repr__`
  - `__setattr__`
  - `__str__`
  --------------------------------------------------------------------------------
### API: jax.experimental.pallas.tpu.CompilerParams

**Signature**:
`jax.experimental.pallas.tpu.CompilerParams(dimension_semantics: 'Sequence[DimensionSemantics] | None' = None, allow_input_fusion: 'Sequence[bool] | None' = None, vmem_limit_bytes: 'int | None' = None, collective_id: 'int | None' = None, has_side_effects: 'bool' = False, flags: 'Mapping[str, Any] | None' = None, internal_scratch_in_bytes: 'int | None' = None, serialization_format: 'int' = 1, kernel_type: 'KernelType' = <KernelType.TC: 0>, disable_bounds_checks: 'bool' = False)`

**Description**:
Mosaic TPU compiler parameters.  Attributes:   dimension_semantics: A list of
dimension semantics for each grid dimension     of the kernel. Either "parallel"
for dimensions that can execute in any     order, or "arbitrary" for dimensions
that must be executed sequentially.   allow_input_fusion: A list of booleans
indicating whether input fusion is     allowed for each argument.
vmem_limit_bytes: Overrides the default VMEM limit for a kernel. Note that
this must be used in conjunction with the     --xla_tpu_scoped_vmem_limit_kib=N
flag with N*1kib > vmem_limit_bytes.   collective_id: Indicates which barrier
semaphore to use for the kernel. Note     that using the same collective_id does
not guarantee that the same barrier     semaphore will be allocated between
kernels.   has_side_effects: Set to True to prevent kernel being CSEd by XLA.
flags: A dictionary of command line flags for the kernel.
internal_scratch_in_bytes: The size of the internal scratch space used by
Mosaic.   serialization_format: The serialization format for the kernel body.
kernel_type: Specify if the kernel is meant to run on TensorCore or one of
the SparseCores   disable_bounds_checks: Disable bounds checks in the kernel.

**Parameters**:
  - **dimension_semantics**: A list of dimension semantics for each grid dimension
of the kernel. Either "parallel" for dimensions that can execute in any
order, or "arbitrary" for dimensions that must be executed sequentially.
  - **allow_input_fusion**: A list of booleans indicating whether input fusion is
allowed for each argument.
  - **vmem_limit_bytes**: Overrides the default VMEM limit for a kernel. Note that
this must be used in conjunction with the
--xla_tpu_scoped_vmem_limit_kib=N flag with N*1kib > vmem_limit_bytes.
  - **collective_id**: Indicates which barrier semaphore to use for the kernel. Note
that using the same collective_id does not guarantee that the same barrier
semaphore will be allocated between kernels.
  - **has_side_effects**: Set to True to prevent kernel being CSEd by XLA.
  - **flags**: A dictionary of command line flags for the kernel.
  - **internal_scratch_in_bytes**: The size of the internal scratch space used by
Mosaic.
  - **serialization_format**: The serialization format for the kernel body.
  - **kernel_type**: Specify if the kernel is meant to run on TensorCore or one of
the SparseCores
  - **disable_bounds_checks**: Disable bounds checks in the kernel.

**Attributes**:
  - `BACKEND`
  - `allow_input_fusion`
  - `collective_id`
  - `dimension_semantics`
  - `disable_bounds_checks`
  - `flags`
  - `has_side_effects`
  - `internal_scratch_in_bytes`
  - `kernel_type`
  - `serialization_format`
  - `vmem_limit_bytes`

**Methods**:
  - `__delattr__`
  - `__eq__`
  - `__hash__`
  - `__init__`
  - `__replace__`
  - `__repr__`
  - `__setattr__`
  - `replace`
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.tpu.PrefetchScalarGridSpec

**Signature**:
`jax.experimental.pallas.tpu.PrefetchScalarGridSpec(num_scalar_prefetch: 'int', grid: 'Grid' = (), in_specs: 'BlockSpecTree' = NoBlockSpec, out_specs: 'BlockSpecTree' = NoBlockSpec, scratch_shapes: 'ScratchShapeTree' = ())`

**Description**:
PrefetchScalarGridSpec(num_scalar_prefetch: 'int', grid: 'Grid' = (), in_specs:
'BlockSpecTree' = NoBlockSpec, out_specs: 'BlockSpecTree' = NoBlockSpec,
scratch_shapes: 'ScratchShapeTree' = ())

**Attributes**:
  - `scratch_shapes`

**Methods**:
  - `__eq__`
  - `__hash__`
  - `__init__`
  - `__replace__`
  - `__repr__`
  - `_make_scalar_ref_aval`
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.tpu.MemorySpace

**Signature**:
`jax.experimental.pallas.tpu.MemorySpace(*values)`

**Attributes**:
  - `ANY`
  - `CMEM`
  - `HBM`
  - `HOST`
  - `SEMAPHORE`
  - `SMEM`
  - `VMEM`
  - `VMEM_SHARED`
  - `name`
  - `value`
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.tpu.sync_copy

**Signature**:
`jax.experimental.pallas.tpu.sync_copy(src_ref, dst_ref)`

**Description**:
Copies a PyTree of Refs to another PyTree of Refs.  Args:   src_ref: A Pytree of
source Refs/TransformedRefs.   dst_ref: A Pytree of destination
Refs/TransformedRefs.

**Parameters**:
  - **src_ref**: A Pytree of source Refs/TransformedRefs.
  - **dst_ref**: A Pytree of destination Refs/TransformedRefs.
--------------------------------------------------------------------------------
### API: jax.experimental.pallas.tpu.async_copy

**Signature**:
`jax.experimental.pallas.tpu.async_copy(src_ref, dst_ref, sem, *, priority: 'int' = 0) -> 'AsyncCopyDescriptor'`

**Description**:
Issues a DMA copying from src_ref to dst_ref.
--------------------------------------------------------------------------------





"""
