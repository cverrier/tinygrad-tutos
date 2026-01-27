# Kernel Code Rendering in tinygrad

This guide explains how tinygrad transforms kernel UOps into device-specific source code.

## Where Rendering Fits in the Pipeline

```
Tensor → UOp Graph → Schedule → Codegen → Runtime
                                   ↑
                            This guide
```

The rendering system takes kernel UOps (the intermediate representation after scheduling) and produces executable code for specific devices (Metal, CUDA, OpenCL, etc.).

## Kernel UOp Types

Kernel UOps represent low-level operations that map directly to generated code. Here are the key categories:

| Category | UOps | Purpose |
|----------|------|---------|
| **Definitions** | `DEFINE_GLOBAL`, `DEFINE_LOCAL`, `DEFINE_VAR`, `DEFINE_REG` | Declare buffers and variables |
| **Control Flow** | `RANGE`, `END`, `IF`, `ENDIF`, `BARRIER` | Loops, conditionals, synchronization |
| **Memory** | `INDEX`, `LOAD`, `STORE` | Memory addressing and access |
| **Arithmetic** | `ADD`, `MUL`, `SUB`, etc. (GroupOp.ALU) | Computations |
| **Constants** | `CONST`, `VCONST` | Literal values |
| **GPU** | `SPECIAL` | Thread/block indices |
| **Vector** | `VECTORIZE`, `GEP` | Vector creation and element access |
| **Type** | `CAST`, `BITCAST` | Type conversions |

Each UOp has the structure:
```python
UOp(op=Ops.ADD, dtype=dtypes.float, src=(a, b), arg=None)
```

- `op`: The operation type from the `Ops` enum
- `dtype`: Data type of the result
- `src`: Tuple of input UOps
- `arg`: Operation-specific argument (e.g., buffer index for `DEFINE_GLOBAL`)

## The Rendering Pipeline

Rendering happens in three steps, orchestrated by `get_program()` in `codegen/__init__.py`:

### Step 1: Optimization (`full_rewrite_to_sink`)

The kernel UOp graph goes through 13+ pattern matcher passes:

```python
# codegen/__init__.py:29-110
def full_rewrite_to_sink(sink:UOp, ren:Renderer|None=None, optimize:bool=True) -> UOp:
  # ... 13+ graph_rewrite passes including:
  # - pm_mops: movement ops
  # - pm_load_collapse: collapse loads
  # - pm_split_ranges: split ranges
  # - sym: symbolic simplification
  # - pm_simplify_ranges: range optimization
  # - expander: expand operations
  # - pm_add_gpudims: add GPU dimensions
  # - pm_add_loads: add load operations
  # - devectorize: handle vectorization
  # - pm_decomp: decompose unsupported ops
  # - pm_add_control_flow: add control flow
```

### Step 2: Linearization (`do_linearize`)

The DAG is converted to a linear list via topological sort with priorities:

```python
# codegen/__init__.py:132-135
def do_linearize(prg:UOp, sink:UOp) -> UOp:
  lst = line_rewrite(linearize(sink), pm_linearize_cleanups)
  return prg.replace(src=prg.src + (UOp(Ops.LINEAR, src=tuple(lst)),))
```

The linearizer (`codegen/late/linearizer.py`) uses a priority-based topological sort:

```python
# Priority rules (smaller = earlier):
# DEFINE_GLOBAL: -20  (first, with buffer index ordering)
# DEFINE_VAR: -19
# DEFINE_LOCAL: -18
# DEFINE_REG: -17
# LOAD: -1           (place loads early)
# default: 0
# STORE: 1           (place stores late)
# END: -5            (close loops as late as possible)
# RANGE: 5           (open loops)
```

### Step 3: Rendering (`do_render`)

The linearized UOps are converted to source code:

```python
# codegen/__init__.py:137-139
def do_render(ctx:Renderer, prg:UOp, lin:UOp) -> UOp:
  src = ctx.render(list(lin.src))
  return prg.replace(src=prg.src + (UOp(Ops.SOURCE, arg=src),))
```

## The PatternMatcher System

tinygrad uses pattern matching extensively for transformations. The key components:

### UPat (UOp Pattern)

Patterns that match UOps:

```python
# Match any ADD operation
UPat(Ops.ADD)

# Match ADD with named sources
UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("b")))

# Match LOAD from any buffer
UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True)
```

### PatternMatcher

A collection of patterns with transformation functions:

```python
# renderer/cstyle.py:12-61
base_rewrite = PatternMatcher([
  # RANGE → for loop
  (UPat(Ops.RANGE, name="x"),
   lambda ctx,x: f"for ({ctx.render_dtype(x.dtype)} {ctx[x]} = 0; {ctx[x]} < {ctx[x.src[0]]}; {ctx[x]}++) {{"),

  # LOAD → pointer dereference
  (UPat(Ops.LOAD, src=(UPat.var('bidx'),), allow_any_len=True),
   lambda ctx,bidx: f"(*{ctx[bidx]})"),

  # STORE → assignment
  (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var")), allow_any_len=True),
   lambda ctx,bidx,var: f"*{ctx[bidx]} = {ctx[var]};"),
])
```

### graph_rewrite

Applies patterns recursively until no more matches:

```python
result = graph_rewrite(input_uop, pattern_matcher, ctx=context)
```

## The code_for_op Dictionary

ALU operations are rendered using a dictionary of lambdas:

```python
# renderer/cstyle.py:123-131
code_for_op: dict = {
  Ops.SQRT: lambda x,dtype: f"sqrt({x})",
  Ops.RECIPROCAL: lambda x,dtype: f"(1/{x})",
  Ops.NEG: lambda x,dtype: f"-{x}",
  Ops.ADD: lambda a,b,dtype: f"({a}+{b})",
  Ops.MUL: lambda a,b,dtype: f"({a}*{b})",
  Ops.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})",
  # ...
}
```

The pattern matcher uses this to render ALU ops:

```python
# renderer/cstyle.py:55-56
(UPat(GroupOp.ALU, name="x"), lambda ctx,x: ctx.code_for_op[x.op](
  *([strip_parens(ctx[v]) if v.op == x.op and x.op in {Ops.ADD, Ops.MUL, ...} else ctx[v] for v in x.src]), x.dtype)),
```

## The CStyleLanguage._render() Method

The core rendering algorithm (`renderer/cstyle.py:161-217`):

```python
def _render(self, uops:list[UOp]) -> tuple[str, list[str], list[tuple[str,tuple[DType,bool]]]]:
  r: dict[UOp, str] = {}  # Maps UOps to variable names
  child_count = Counter(v for ru in uops for v in ru.src)
  kernel = []
  depth = 1  # Indentation depth
  c: defaultdict[str, int] = defaultdict(int)  # Name counters

  for u in uops:
    # Skip NOOPs
    if u.op in {Ops.NOOP, Ops.GROUP}: continue

    # Handle definitions
    if u.op is Ops.DEFINE_GLOBAL:
      r[u] = f"data{u.arg}"
      continue

    # Generate variable name based on op type
    prefix = {Ops.LOAD: "val", Ops.INDEX: "bidx", Ops.CONST: "const", ...}.get(u.op, "alu")
    r[u] = f"{prefix}{c[prefix]}"

    # Apply string_rewrite PatternMatcher to get code
    l = self.string_rewrite.rewrite(u, ctx=self)

    # Track depth for braces
    if u.op in {Ops.ENDIF, Ops.END}: depth -= 1

    # Decide: inline or emit as statement
    if should_inline(u):
      r[u] = l  # Store expression directly
    else:
      if needs_declaration(u):
        l = f"{self.render_dtype(u.dtype)} {r[u]} = {l};"
      kernel.append("  "*depth + l)
      c[prefix] += 1

    if u.op in {Ops.IF, Ops.RANGE}: depth += 1
```

Key decisions:
- **Inlining**: Constants, single-use ALU ops, and index calculations are inlined
- **Statements**: LOADs, STOREs, and multi-use values get their own lines
- **Depth tracking**: `{` and `}` are managed by RANGE/END and IF/ENDIF

## End-to-End Example

Here's what happens when you run:

```python
from tinygrad import Tensor
(Tensor([1,2,3]) + 1).realize()
```

### 1. Kernel UOp Graph

```python
c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(3), (), 0)  # output buffer
c4 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(3), (), 1)  # input buffer
c2 = UOp.range(3, 0, AxisType.LOOP)                     # loop i in 0..3
c7 = c4.index(c2) + UOp.const(dtypes.int, 1)           # input[i] + 1
c9 = c0.index(c2, ptr=True).store(c7).end(c2)          # output[i] = ...
ast = c9.sink(...)
```

### 2. After Optimization

GPU dimensions are added, the loop becomes a thread index:

```
DEFINE_GLOBAL data0 (output)
DEFINE_GLOBAL data1 (input)
SPECIAL lidx0 (local thread id)
LOAD val0 = *(data1 + lidx0)
ADD alu0 = val0 + 1
STORE *(data0 + lidx0) = alu0
```

### 3. Rendered Metal Code

```c
kernel void E_3(device int* data0_3, device int* data1_3,
                uint3 gid [[threadgroup_position_in_grid]],
                uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 3 */
  int val0 = (*(data1_3+lidx0));
  *(data0_3+lidx0) = (val0+1);
}
```

## Device-Specific Rendering

Different devices extend `CStyleLanguage` or `Renderer` with their own:

| Device | Renderer | Key Differences |
|--------|----------|-----------------|
| **Metal** | `MetalRenderer` | `lid.x`, `gid.x`, `threadgroup_barrier`, `precise::sin` |
| **CUDA** | `CUDARenderer` | `threadIdx.x`, `blockIdx.x`, `__syncthreads()` |
| **OpenCL** | `OpenCLRenderer` | `get_local_id(0)`, `get_group_id(0)`, `barrier()` |
| **PTX** | `PTXRenderer` | Assembly instructions: `add.s32`, `ld.global.f32` |
| **CPU/Clang** | `ClangRenderer` | No GPU primitives, vector intrinsics |

### Example: Same Operation on Different Devices

| UOp | Metal | CUDA | PTX |
|-----|-------|------|-----|
| `SPECIAL(lidx0)` | `lid.x` | `threadIdx.x` | `mov.u32 %lidx0, %tid.x;` |
| `ADD(a, b)` | `(a+b)` | `(a+b)` | `add.s32 %r, %a, %b;` |
| `BARRIER` | `threadgroup_barrier(...)` | `__syncthreads()` | `bar.sync 0;` |

Device-specific `code_for_op` overrides:

```python
# CUDA: half-precision intrinsics
Ops.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype == dtypes.half else f"sqrt({x})"

# AMD: OCML library calls
Ops.SQRT: lambda x,dtype: f"__ocml_sqrt_f32({x})"
```

## Key Source Files

| File | Purpose |
|------|---------|
| `tinygrad/codegen/__init__.py` | Pipeline orchestration, `get_program()` |
| `tinygrad/renderer/__init__.py` | Base `Renderer` class, `ProgramSpec` |
| `tinygrad/renderer/cstyle.py` | C-style rendering (CUDA, Metal, OpenCL, Clang) |
| `tinygrad/renderer/ptx.py` | NVIDIA PTX assembly |
| `tinygrad/codegen/late/linearizer.py` | Topological sort with priorities |
| `tinygrad/uop/__init__.py` | `Ops` enum, `GroupOp` collections |

## Debugging

View rendered code with `DEBUG=5`:

```bash
DEBUG=5 python -c "from tinygrad import Tensor; (Tensor([1,2,3])+1).realize()"
```

View the UOp graph before rendering:

```bash
VIZ=1 python -c "from tinygrad import Tensor; (Tensor([1,2,3])+1).realize()"
```

Print linearized UOps:

```bash
DEBUG=6 python -c "from tinygrad import Tensor; (Tensor([1,2,3])+1).realize()"
```
