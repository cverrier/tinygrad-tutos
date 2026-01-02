# tinygrad JIT System: A Concise Practical Guide

> [!NOTE]
> This tutorial has been built around tinygrad's commit [526fd4e](https://github.com/tinygrad/tinygrad/tree/526fd4ec7104eda1ef8114e64d99b2788910a8fd) with a Python 3.11 virtual environment. Since tinygrad is evolving rapidly, if you're following along in code, make sure to checkout that commit in order to avoid any discrepancies.

This tutorial explains tinygrad's [JIT (Just-In-Time) compilation](https://grokipedia.com/page/Just-in-time_compilation) system. You'll learn what it does, how it works internally, and how to use it effectively.

## Table of Contents

1. [The Problem JIT Solves](#the-problem-jit-solves)
2. [The Three Phases of TinyJit](#the-three-phases-of-tinyjit)
3. [What Gets Cached](#what-gets-cached)
4. [Dynamic Shapes with Variables](#dynamic-shapes-with-variables)
5. [Common Pitfalls](#common-pitfalls)
6. [Practical Patterns](#practical-patterns)
7. [Graph Execution](#graph-execution)

---

## The Problem JIT Solves

In general, the first time you call `.realize()` on a tensor, tinygrad must:
1. Build/verify the `UOp` graph
2. Run the scheduler to create kernels
3. Compile kernels into bytecode, and cache the result
4. Execute kernels (create a GPU-executable object from the bytecode -- one executable object per kernel)

Steps 1-3 are *overhead* — they only need to happen once per unique computation structure. Step 4 is the actual work of running the computation: after all, during model training or inference, you only run the *same computation* repeatedly with *different inputs*. This is where tinygrad's JIT system comes in. Let's see how.

As mentioned, in step 3, tinygrad always caches kernel compilation to bytecode, so subsequent calls with the same computation structure skip that step, which is already a big win:

```python
from tinygrad import Tensor
from tinygrad.helpers import Timing

def do_ops(a, b):
  x1 = (a + b).realize()
  x2 = (x1 * x1).realize()
  x3 = (2 * x2 - 1).realize()
  return x3

for i in range(5):
  t1, t2 = Tensor.empty(4, 4).realize(), Tensor.empty(4, 4).realize()
  with Timing(f"Call {i+1}: "):
    do_ops(t1, t2)
```

Each time `do_ops()` is called, it generates three kernels (I forced this behavior using `.realize()` three times). Running this code (with `DEBUG=2` to measure kernel execution timing accurately) produces the following output on METAL backend:

```
*** METAL      1 E_4_4                                          arg  3 mem   0.00 GB tm     12.25us/     0.01ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL      2 E_4_4n1                                        arg  2 mem   0.00 GB tm     11.87us/     0.02ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL      3 E_4_4n2                                        arg  2 mem   0.00 GB tm     11.75us/     0.04ms (      0 GFLOPS    0|0      GB/s) ['__rmul__', '__sub__']
Call 1:  27.56 ms
*** METAL      4 E_4_4                                          arg  3 mem   0.00 GB tm      9.08us/     0.04ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL      5 E_4_4n1                                        arg  2 mem   0.00 GB tm      9.08us/     0.05ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL      6 E_4_4n2                                        arg  2 mem   0.00 GB tm      8.75us/     0.06ms (      0 GFLOPS    0|0      GB/s) ['__rmul__', '__sub__']
Call 2:   1.72 ms
*** METAL      7 E_4_4                                          arg  3 mem   0.00 GB tm      8.88us/     0.07ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL      8 E_4_4n1                                        arg  2 mem   0.00 GB tm      8.88us/     0.08ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL      9 E_4_4n2                                        arg  2 mem   0.00 GB tm      8.75us/     0.09ms (      0 GFLOPS    0|0      GB/s) ['__rmul__', '__sub__']
Call 3:   1.56 ms
*** METAL     10 E_4_4                                          arg  3 mem   0.00 GB tm      8.88us/     0.10ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL     11 E_4_4n1                                        arg  2 mem   0.00 GB tm      8.83us/     0.11ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL     12 E_4_4n2                                        arg  2 mem   0.00 GB tm      8.75us/     0.12ms (      0 GFLOPS    0|0      GB/s) ['__rmul__', '__sub__']
Call 4:   1.42 ms
*** METAL     13 E_4_4                                          arg  3 mem   0.00 GB tm      8.88us/     0.12ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL     14 E_4_4n1                                        arg  2 mem   0.00 GB tm      8.87us/     0.13ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL     15 E_4_4n2                                        arg  2 mem   0.00 GB tm      8.75us/     0.14ms (      0 GFLOPS    0|0      GB/s) ['__rmul__', '__sub__']
Call 5:   1.33 ms
```

As you can see, there are three kernels (`E_4_4`, `E_4_4n1` and `E_4_4n2`) per call. The first call takes much longer due to kernel compilation. The subsequent calls are much faster, reusing the cached bytecode. More precisely, during call 1, tinygrad generates C-like source code for the three kernels emanating from `do_ops()` (run the Python code snippet with `DEBUG=4` to see this source code). For example, the first kernel's source code looks like this on METAL backend:
```cpp
#include <metal_stdlib>
using namespace metal;
kernel void E_4_4(device float* data0_16, device float* data1_16, device float* data2_16, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int lidx0 = lid.x; /* 4 */
  int alu0 = (lidx0<<2);
  float4 val0 = (*((device float4*)((data1_16+alu0))));
  float4 val1 = (*((device float4*)((data2_16+alu0))));
  *((device float4*)((data0_16+alu0))) = float4((val0.x+val1.x),(val0.y+val1.y),(val0.z+val1.z),(val0.w+val1.w));
}
```
Then tinygrad compiles this source code (like a C compiler) and caches the compilation result for future reuse (*i.e.*, in subsequent calls) — this is why calls 2-5 are much faster.

However, it's possible to do even better: even though the kernels' source code has been compiled and cached, this has to be converted into GPU-executable objects (one per kernel) and executed on the GPU. This also creates some overhead that can be significant when many kernels need to be executed. This is where tinygrad's JIT system comes in.

Let's decorate our `do_ops()` function with `TinyJit` and run the script (with `DEBUG=2`):

```python
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Timing

@TinyJit
def do_ops(a, b):
  x1 = (a + b).realize()
  x2 = (x1 * x1).realize()
  x3 = (2 * x2 - 1).realize()
  return x3

for i in range(5):
  t1, t2 = Tensor.empty(4, 4).realize(), Tensor.empty(4, 4).realize()
  with Timing(f"Call {i+1}: "):
    do_ops(t1, t2)
```

```
*** METAL      1 E_4_4                                          arg  3 mem   0.00 GB tm      9.00us/     0.01ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL      2 E_4_4n1                                        arg  2 mem   0.00 GB tm      9.33us/     0.02ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL      3 E_4_4n2                                        arg  2 mem   0.00 GB tm      8.87us/     0.03ms (      0 GFLOPS    0|0      GB/s) ['__rmul__', '__sub__']
Call 1:  29.96 ms
*** METAL      4 E_4_4                                          arg  3 mem   0.00 GB tm      5.62us/     0.03ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL      5 E_4_4n1                                        arg  2 mem   0.00 GB tm      5.71us/     0.04ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL      6 E_4_4n2                                        arg  2 mem   0.00 GB tm      5.71us/     0.04ms (      0 GFLOPS    0|0      GB/s) ['__rmul__', '__sub__']
JIT captured 3 kernels with 0 inputs
Call 2:   1.70 ms
JIT GRAPHing batch with 3 kernels on device <tinygrad.runtime.ops_metal.MetalDevice object at 0x107c85490>
*** METAL      7 <batched 3>                                    arg  0 mem   0.00 GB tm     35.58us/     0.08ms (      0 GFLOPS    0|0      GB/s)
Call 3:  14.52 ms
*** METAL      8 <batched 3>                                    arg  0 mem   0.00 GB tm     28.46us/     0.11ms (      0 GFLOPS    0|0      GB/s)
Call 4:   0.70 ms
*** METAL      9 <batched 3>                                    arg  0 mem   0.00 GB tm     29.04us/     0.14ms (      0 GFLOPS    0|0      GB/s)
Call 5:   0.45 ms
```

The first call is similar to what we observed before (kernels are compiled and cached). Call 2 is also very similar to before, but now JIT also captures the three kernels generated during this call (these are the GPU-executable objects), and records them for future replay (`JIT captured 3 kernels with 0 inputs`). Now, instead of telling the GPU to execute each kernel one after one, JIT can batch them into a single GPU command so as to avoid CPU-GPU submission overhead: this is exactly what happens during call 3, where tinygrad builds a GPU graph that batches the three kernels into one command (`JIT GRAPHing batch with 3 kernels on device ...`), and then executes this batched command (`<batched 3> ...`). As a consequence, call 3 takes longer because of the graph building and batching overhead, but subsequent calls (4 and 5) are very fast because they simply reuse the already built GPU graph and execute the batched command. If you compare calls 4 and 5 between the non-JIT and JIT versions, you can see a significant speedup!

In summary, tinygrad's JIT system helps skip the overhead of scheduling and dispatching multiple kernels by capturing the kernel sequence during the second call, and replaying it as a single batched GPU command in subsequent calls. For more detail on how JIT's graph execution works, see the [Graph Execution](#graph-execution) section below.

---

## The Three Phases of TinyJit

The `TinyJit` decorator has exactly three phases, controlled by an internal counter:

| Call | Phase | What happens |
|------|-------|--------------|
| 1 | Warmup | Function runs normally |
| 2 | Capture | Function runs, kernels are *recorded* |
| 3+ | Replay | Function body *skipped*, recorded kernels replayed |

Let's observe this directly:

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def add(a, b):
    print("  Function body running")
    return a + b

for i in range(4):
    print(f"Call {i+1}:")
    add(Tensor.empty(4, 4), Tensor.empty(4, 4))
```

The output is:
```
Call 1:
  Function body running
Call 2:
  Function body running
Call 3:
Call 4:
```

As you can see, the Python function body only runs twice. On calls 3+, JIT replays the captured kernels with new input buffers — your Python code never executes.

### Why Warmup Exists

The warmup phase (call 1) allows lazy initialization to happen before capture:

```python
from tinygrad import Tensor, TinyJit

class Model:
    def __init__(self):
        self.weights = None

    @TinyJit
    def forward(self, x):
        if self.weights is None:
            print("  Initializing weights")
            self.weights = Tensor.randn(100, 100)
        return x @ self.weights

model = Model()
for i in range(3):
    print(f"Call {i+1}:")
    model.forward(Tensor.randn(32, 100))
```

The output is:
```
Call 1:
  Initializing weights
Call 2:
Call 3:
```

If JIT captured on call 1, it would record the weight initialization kernel, and on every replay, it would try to re-initialize weights. By waiting until call 2, warmup handles one-time setup, and capture only records the "steady state" computation.

---

## What Gets Cached

JIT tracks two kinds of buffers differently:

1. Input buffers (function arguments) → swapped on each call
2. Closure buffers (accessed from outside, like weights) → same buffer reused

```python
from tinygrad import Tensor, TinyJit

weights = Tensor([1, 2, 3])

@TinyJit
def f(x):
    return x + weights

# Warmup + capture
f(Tensor([10, 10, 10]))
f(Tensor([10, 10, 10]))

# Replay with different input
print("With x=[100, 100, 100]:", f(Tensor([100, 100, 100])).numpy())

# Change weights and try again
weights = Tensor([999, 999, 999])
print("After changing weights:", f(Tensor([100, 100, 100])).numpy())
```

Output:
```
With x=[100, 100, 100]: [101 102 103]
After changing weights: [101 102 103]
```

Changing `weights` has no effect — the original buffer was captured during call 2 and reused on replay.

> [!IMPORTANT]
> After capture, JIT replays kernels with the captured buffers. Only input arguments (function parameters) are swapped; everything else is baked in.

### Shape Validation

JIT validates that input shapes match what was captured:

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def add(a, b):
    return a + b

# Warmup + capture with shape (100, 100)
add(Tensor.randn(100, 100), Tensor.randn(100, 100))
add(Tensor.randn(100, 100), Tensor.randn(100, 100))

# Try computation with shape (200, 200) and fail
add(Tensor.randn(200, 200), Tensor.randn(200, 200))
```

Running this code raises the following error on METAL backend:
```
AssertionError: args mismatch in JIT: self.captured.expected_st_vars_dtype_device=[(UOp(Ops.RESHAPE, dtypes.float, arg=None, src=(
  UOp(Ops.NOOP, dtypes.void, arg=None, src=()),
  UOp(Ops.CONST, dtypes.index.vec(2), arg=100, src=()),)), (), dtypes.float, 'METAL'), (UOp(Ops.RESHAPE, dtypes.float, arg=None, src=(
  UOp(Ops.NOOP, dtypes.void, arg=None, src=()),
  UOp(Ops.CONST, dtypes.index.vec(2), arg=100, src=()),)), (), dtypes.float, 'METAL')] != st_vars_dtype_device=[(UOp(Ops.RESHAPE, dtypes.float, arg=None, src=(
  UOp(Ops.NOOP, dtypes.void, arg=None, src=()),
  UOp(Ops.CONST, dtypes.index.vec(2), arg=200, src=()),)), (), dtypes.float, 'METAL'), (UOp(Ops.RESHAPE, dtypes.float, arg=None, src=(
  UOp(Ops.NOOP, dtypes.void, arg=None, src=()),
  UOp(Ops.CONST, dtypes.index.vec(2), arg=200, src=()),)), (), dtypes.float, 'METAL')]
```

Indeed, the kernel was captured with shape `(100, 100)`, so JIT rejects inputs of shape `(200, 200)`.

---

## Dynamic Shapes with Variables

What if you need different sizes? Fortunately, tinygrad supports symbolic shapes using `Variable`:

```python
from tinygrad import Tensor, TinyJit, Variable

@TinyJit
def sum_first_n(x):
    return x.sum()

base = Tensor.arange(10)
for i in range(1, 6):
    # Create a Variable: name="n", range=[1,10], current value=i
    n = Variable("n", 1, 10).bind(i)
    result = sum_first_n(base[:n])
    print(f"sum of first {i}: {result.item()}")
```

which prints:
```
Sum of first 1: 0
Sum of first 2: 1
Sum of first 3: 3
Sum of first 4: 6
Sum of first 5: 10
```

One kernel handles all sizes. During capture, the kernel is compiled with a symbolic loop bound (run with `DEBUG=4` to see the generated source code):
```cpp
#include <metal_stdlib>
using namespace metal;
kernel void r_n(device int* data0_1, device int* data1_10, constant int& n, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int acc0[1];
  *(acc0+0) = 0;
  for (int Ridx0 = 0; Ridx0 < n; Ridx0++) {
    int val0 = (*(data1_10+Ridx0));
    *(acc0+0) = ((*(acc0+0))+val0);
  }
  *(data0_1+0) = (*(acc0+0));
}
```
Look at the loop bound: `for (int Ridx0 = 0; Ridx0 < n; Ridx0++)` — `n` is a symbolic variable passed as a kernel argument (`constant int& n`). On each replay, tinygrad passes the concrete value of `n` to the kernel.

> [!TIP]
> Use `Variable` for dimensions that change within a known range (like sequence length in LLMs). The kernel is compiled once with symbolic bounds and reused for all sizes.

---

## Common Pitfalls

Each pitfall stems from one principle:
> After capture, Python doesn't run — JIT just replays kernels with swapped input buffers.

### Pitfall 1: Output Buffer Reuse

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(x):
    return x.sum()

r1 = f(Tensor([1, 1]))  # warmup
r2 = f(Tensor([2, 2]))  # capture
r3 = f(Tensor([3, 3]))  # replay

print(f"r1: {r1.item()}")  # 2 (warmup, independent)
print(f"r2: {r2.item()}")  # 6 (should be 4! overwritten by r3)
print(f"r3: {r3.item()}")  # 6
```

Output:
```
r1: 2
r2: 6
r3: 6
```

During capture, JIT records the output buffer. On replay, it writes to the same buffer, overwriting previous results.

> [!TIP]
> Clone and realize outputs if you need to keep them:
>
>```python
>r2 = f(Tensor([2, 2])).clone().realize()
>```

### Pitfall 2: Python Values are Frozen

```python
from tinygrad import Tensor, TinyJit

multiplier = 10

@TinyJit
def f(x):
    return x * multiplier

f(Tensor([5]))  # warmup
f(Tensor([5]))  # capture

multiplier = 99
print(f(Tensor([5])).item())  # Prints 50, not 99*5=495
```

During capture, `multiplier` was `10`, so the kernel was compiled with that value. To avoid this behavior, make sure to pass changing values as `Tensor` arguments:

```python
from tinygrad import Tensor

@TinyJit
def f(x: Tensor, multiplier: Tensor):  # pass multiplier as a tinygrad Tensor
    return x * multiplier
```

### Pitfall 3: Conditional Branches are Frozen

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(x, use_square):
    if use_square:
        return (x * x).realize()
    else:
        return (x * 2).realize()

f(Tensor([3]), True)   # warmup - takes True branch
f(Tensor([3]), False)  # capture - takes False branch

print(f(Tensor([3]), True).item())  # Prints 6, not 3*3=9
```

During capture, `use_square=False`, so only the `x * 2` kernel was recorded. If you need to have multiple branches, use separate JIT functions for each branch, or make the condition part of the computation:

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(x, square_flag):  # square_flag is a tinygrad Tensor
    squared = x * x
    doubled = x * 2
    return squared * square_flag + doubled * (1 - square_flag)

f(Tensor([3]), Tensor([True]))
f(Tensor([3]), Tensor([False]))

print(f(Tensor([3]), Tensor([True])).item())  # Prints 9
```

### Pitfall 4: Tensors in Containers Aren't Tracked

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(a, tensor_list):
    return (a + tensor_list[0]).realize()

a = Tensor([1, 1, 1])

f(a, [Tensor([10, 10, 10])])  # warmup
f(a, [Tensor([20, 20, 20])])  # capture - tensor_list[0] baked in
print(f(a, [Tensor([99, 99, 99])]).numpy())  # Prints [21 21 21], not [100 100 100]
```

JIT only tracks top-level Tensor arguments. Objects such as lists and dictionaries are not scanned, so, as always, make sure to pass tensors directly as arguments.

### Pitfall 5: Duplicate Inputs Fail

This one can be very subtle and tricky:

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(a, b):
    return a + b

t = Tensor([1, 2, 3])
f(t, t)  # AssertionError: duplicate inputs to JIT
```

I guess that JIT needs to know which input argument to use for each buffer slot in the cached kernels (if you are curious, see [tinygrad/engine/jit.py](https://github.com/tinygrad/tinygrad/blob/526fd4ec7104eda1ef8114e64d99b2788910a8fd/tinygrad/engine/jit.py#L67-L73)). So when calling the `TinyJit`-decorated function, each input argument must correspond to a unique buffer: tinygrad first calls `TinyJit.__call__` with the same input buffers `(t, t)`, which then calls `_prepare_jit_inputs` (see [reference](https://github.com/tinygrad/tinygrad/blob/526fd4ec7104eda1ef8114e64d99b2788910a8fd/tinygrad/engine/jit.py#L273-L274))

```python
  def __call__(self, *args, **kwargs) -> ReturnType:
    # Here, in our case, args = (t, t) (same buffers)
    input_buffers, var_vals, names, st_vars_dtype_device = _prepare_jit_inputs(args, kwargs)
```

which then processes the input arguments and checks for duplicates (see [reference](https://github.com/tinygrad/tinygrad/blob/526fd4ec7104eda1ef8114e64d99b2788910a8fd/tinygrad/engine/jit.py#L220-L228)):

```python
def _prepare_jit_inputs(args, kwargs):
  # args = (t, t) where t is the same Tensor object

  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  # input_tensors = [(0, t), (1, t)]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # names = [0, 1]
  # tensors = [t, t]

  # Then realize unrealized tensors...

  lbs: list[UOp] = flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  # lbs = [t.uop, t.uop]  (same UOp twice)

  input_buffers: list[Buffer] = flatten([rb.bufs if isinstance(rb:=lb.base.realized, MultiBuffer) else [rb]
                                         for lb in lbs if lb.base.realized is not None])
  # input_buffers = [buf, buf]  (same Buffer twice)

  # and fails here
  assert len(set(input_buffers)) == len(input_buffers), "duplicate inputs to JIT"
```

**Fix**: Clone if needed:
```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(a, b):
    return a + b

x = Tensor([1, 2, 3])
print(f(x, x.clone()).numpy())  # No error, prints [2 4 6]
```

> [!WARNING]
> All pitfalls stem from one principle: after JIT capture, Python code doesn't run. Only captured kernels replay with swapped input buffers.

---

## Practical Patterns

### Pattern 1: Basic Inference Loop with Accurate Timing

Sometimes, it's interesting to do some timing measurements to see how fast your model runs with JIT. However, due to GPU asynchronicity, naive timing can be misleading. Let's see an example:

```python
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Timing

class Model:
    def __init__(self):
        self.w1 = Tensor.randn(100, 50)
        self.w2 = Tensor.randn(50, 10)

    @TinyJit
    def forward(self, x):
        return (x @ self.w1).relu() @ self.w2

model = Model()
for i in range(6):
    x = Tensor.randn(32, 100)
    with Timing(f"Call {i+1}: "):
        out = model.forward(x)
```

Running this code snippet without any `DEBUG` flags produces the following output:
```
Call 1: 252.57 ms
Call 2:  20.12 ms
Call 3:   9.35 ms
Call 4:  69.04 ms
Call 5:   1.13 ms
Call 6:   1.04 ms
```

Something seems odd here: from what you've learned so far, you should expect that call 3 should take a bit longer than call 2 (due to GPU computation graph building overhead), and calls 4-6 should be way faster. However, call 4 is much slower than call 3! Why is that? This is because GPU execution is asynchronous: when you call `model.forward(x)`, the CPU submits work to the GPU and immediately returns, without waiting for the GPU to finish. More precisely:
1. Call 3 starts: the CPU submits a command to the GPU, and returns immediately: the timing for call 3 ends here, and seems fast (9.35 ms): but this time only measures how long it took for the CPU to submit the command (and not how long the GPU took to execute it).
2. The GPU starts executing the command and is working in the background.
3. Call 4 starts: the CPU tries to submit another command to the GPU, but the GPU is stil busy with the previous command (from call 3). Therefore, the CPU has to wait for the GPU to finish before it can submit the new command. This waiting time is included in the timing of call 4, making it appear much slower.

Now, if you run the same code snippet with `DEBUG=2`, you get:
```
scheduled    6 kernels in    14.28 ms | CACHE MISS d5429687 | 743 uops in cache
*** METAL      1 copy        4,   METAL <- PYTHON               arg  2 mem   0.00 GB tm     41.12us/     0.04ms (      0 GFLOPS    0|0      GB/s)
*** METAL      2 copy        8,   METAL <- PYTHON               arg  2 mem   0.00 GB tm     18.33us/     0.06ms (      0 GFLOPS    0|0      GB/s)
*** METAL      3 E                                              arg  2 mem   0.00 GB tm     15.46us/     0.07ms (      0 GFLOPS    0|0      GB/s) ['randn']
*** METAL      4 En1                                            arg  2 mem   0.00 GB tm      9.38us/     0.08ms (      0 GFLOPS    0|0      GB/s) ['randn']
*** METAL      5 E_50_32_4                                      arg  3 mem   0.00 GB tm     25.17us/     0.11ms (     63 GFLOPS    1|1      GB/s) ['randn']
*** METAL      6 E_25_32_4                                      arg  2 mem   0.00 GB tm     24.25us/     0.13ms (      1 GFLOPS    2|2      GB/s) ['randn']
scheduled    4 kernels in    26.12 ms | CACHE MISS 24a7b763 | 2498 uops in cache
*** METAL      7 E_625_4_4                                      arg  3 mem   0.00 GB tm     34.88us/     0.17ms (     71 GFLOPS    1|1      GB/s)
*** METAL      8 E_125_2_4                                      arg  3 mem   0.00 GB tm     25.38us/     0.19ms (     10 GFLOPS    0|0      GB/s)
*** METAL      9 r_25_8_2_4_25_4                                arg  3 mem   0.00 GB tm    138.75us/     0.33ms (      5 GFLOPS    0|7      GB/s) ['__matmul__', 'relu']
*** METAL     10 r_5_32_2_50                                    arg  3 mem   0.00 GB tm     91.33us/     0.42ms (      2 GFLOPS    0|2      GB/s) ['__matmul__']
Call 1: 244.27 ms
scheduled    3 kernels in     9.45 ms | CACHE MISS 032a593c | 4871 uops in cache
*** METAL     11 En2                                            arg  1 mem   0.00 GB tm     12.71us/     0.44ms (      0 GFLOPS    0|0      GB/s) ['randn']
*** METAL     12 E_50_32_4                                      arg  3 mem   0.00 GB tm     24.62us/     0.46ms (     65 GFLOPS    1|1      GB/s) ['randn']
*** METAL     13 E_25_32_4                                      arg  2 mem   0.00 GB tm     19.75us/     0.48ms (      1 GFLOPS    2|2      GB/s) ['randn']
scheduled    2 kernels in     8.32 ms | CACHE MISS 738de194 | 4910 uops in cache
*** METAL     14 r_25_8_2_4_25_4                                arg  3 mem   0.00 GB tm    138.17us/     0.62ms (      5 GFLOPS    0|7      GB/s) ['__matmul__', 'relu']
*** METAL     15 r_5_32_2_50                                    arg  3 mem   0.00 GB tm     88.04us/     0.71ms (      2 GFLOPS    0|2      GB/s) ['__matmul__']
JIT captured 2 kernels with 1 inputs
JIT memory reduced from 0.01 MB -> 0.01 MB, 1 -> 1 bufs
Call 2:  22.88 ms
scheduled    3 kernels in     9.05 ms | CACHE MISS 21afb088 | 5004 uops in cache
*** METAL     16 En2                                            arg  1 mem   0.00 GB tm     12.63us/     0.72ms (      0 GFLOPS    0|0      GB/s) ['randn']
*** METAL     17 E_50_32_4                                      arg  3 mem   0.00 GB tm     23.29us/     0.74ms (     68 GFLOPS    1|1      GB/s) ['randn']
*** METAL     18 E_25_32_4                                      arg  2 mem   0.00 GB tm     19.79us/     0.76ms (      1 GFLOPS    2|2      GB/s) ['randn']
JIT GRAPHing batch with 2 kernels on device <tinygrad.runtime.ops_metal.MetalDevice object at 0x109ddcad0>
*** METAL     19 <batched 2>                                    arg  1 mem   0.00 GB tm    252.50us/     1.02ms (      3 GFLOPS    0|5      GB/s)
Call 3:  27.10 ms
scheduled    3 kernels in     0.65 ms |  cache hit 21afb088 | 5002 uops in cache
*** METAL     20 En2                                            arg  1 mem   0.00 GB tm     12.50us/     1.03ms (      0 GFLOPS    0|0      GB/s) ['randn']
*** METAL     21 E_50_32_4                                      arg  3 mem   0.00 GB tm     22.62us/     1.05ms (     70 GFLOPS    1|1      GB/s) ['randn']
*** METAL     22 E_25_32_4                                      arg  2 mem   0.00 GB tm     19.75us/     1.07ms (      1 GFLOPS    2|2      GB/s) ['randn']
*** METAL     23 <batched 2>                                    arg  1 mem   0.00 GB tm    239.96us/     1.31ms (      3 GFLOPS    0|5      GB/s)
Call 4:   2.94 ms
scheduled    3 kernels in     0.55 ms |  cache hit 21afb088 | 5002 uops in cache
*** METAL     24 En2                                            arg  1 mem   0.00 GB tm     11.83us/     1.32ms (      0 GFLOPS    0|0      GB/s) ['randn']
*** METAL     25 E_50_32_4                                      arg  3 mem   0.00 GB tm     22.21us/     1.34ms (     72 GFLOPS    1|1      GB/s) ['randn']
*** METAL     26 E_25_32_4                                      arg  2 mem   0.00 GB tm     19.71us/     1.36ms (      1 GFLOPS    2|2      GB/s) ['randn']
*** METAL     27 <batched 2>                                    arg  1 mem   0.00 GB tm    235.08us/     1.60ms (      3 GFLOPS    0|5      GB/s)
Call 5:   2.57 ms
scheduled    3 kernels in     0.55 ms |  cache hit 21afb088 | 5002 uops in cache
*** METAL     28 En2                                            arg  1 mem   0.00 GB tm     12.37us/     1.61ms (      0 GFLOPS    0|0      GB/s) ['randn']
*** METAL     29 E_50_32_4                                      arg  3 mem   0.00 GB tm     23.08us/     1.63ms (     69 GFLOPS    1|1      GB/s) ['randn']
*** METAL     30 E_25_32_4                                      arg  2 mem   0.00 GB tm     19.63us/     1.65ms (      1 GFLOPS    2|2      GB/s) ['randn']
*** METAL     31 <batched 2>                                    arg  1 mem   0.00 GB tm    240.50us/     1.89ms (      3 GFLOPS    0|5      GB/s)
Call 6:   2.48 ms
```

Let's declutter this output just a little bit, so we can focus our attention on the timings:
```
Call 1: 244.27 ms
Call 2:  22.88 ms
Call 3:  27.10 ms
Call 4:   2.94 ms
Call 5:   2.57 ms
Call 6:   2.48 ms
```

Now, the timings make sense: call 3 is slightly slower than call 2 (again, due to GPU graph building overhead), and calls 4-6 are very fast, as expected. This is because `DEBUG` mode forces synchronization after each kernel to measure accurate timing.

> [!NOTE]
> For benchmarking with JIT, always run several warmup calls before measuring. The steady-state performance (calls 5+) is what matters.

### Pattern 2: Mutable State with Assign

Sometimes, you need state that updates each call — like a counter, running average, or [KV cache for transformers](https://huggingface.co/blog/not-lain/kv-caching). You learned that closure tensors are baked in during capture. So how do you update them? Use `.assign()` to update state in place: it writes to the same buffer:

```python
from tinygrad import Tensor, TinyJit

counter = Tensor([0])

@TinyJit
def increment():
  counter.assign(counter + 1)
  return counter

for i in range(5):
  result = increment()
  print(f"Call {i+1}: counter = {result.item()}")
```

Output:
```
Call 1: counter = 1
Call 2: counter = 2
Call 3: counter = 3
Call 4: counter = 4
Call 5: counter = 5
```

This works because `.assign()` modifies the buffer in place. Python assignment (`counter = counter + 1`) would create a new tensor that JIT doesn't know about.

### Pattern 3: Resetting JIT

When you need to change input shapes or model architecture, you can reset JIT to forget captured kernels and recapture on the next call:

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(x):
  return x @ x.T

# Use with shape (10, 5)
for _ in range(4):
  out = f(Tensor.randn(10, 5))
print(f"Shape after first capture: {out.shape}")

# Reset and use with shape (20, 8)
f.reset()
for _ in range(4):
  out = f(Tensor.randn(20, 8))
print(f"Shape after reset: {out.shape}")
```

Output:
```
Shape after first capture: (10, 10)
Shape after reset: (20, 20)
```

### Pattern 4: Variables for Dynamic Shapes

For dimensions that vary within a known range:

```python
from tinygrad import Tensor, TinyJit, Variable

@TinyJit
def process_sequence(x):
  return x.sum(axis=1)

data = Tensor.randn(2, 100, 8)  # batch=2, max_seq=100, features=8

for seq_len in [10, 25, 50, 75]:
  s = Variable("s", 1, 100).bind(seq_len)
  result = process_sequence(data[:, :s, :])
  print(f"seq_len={seq_len}: output shape = {result.shape}")

print(f"\nKernels captured: {len(process_sequence.captured.jit_cache)}")
```

Output:
```
seq_len=10: output shape = (2, 8)
seq_len=25: output shape = (2, 8)
seq_len=50: output shape = (2, 8)
seq_len=75: output shape = (2, 8)

Kernels captured: 1
```

One kernel handles all sequence lengths, which is crucial for LLM inference where the KV cache grows with each token.

---

## Graph Execution

### GPU Commands and Overhead

Your CPU and GPU are separate processors communicating through a command queue:

```
CPU                          GPU
 │                            │
 ├─── "run kernel A" ────────►│ (execute A)
 │    (wait for GPU)          │
 ├─── "run kernel B" ────────►│ (execute B)
 │    (wait for GPU)          │
 ├─── "run kernel C" ────────►│ (execute C)
```

Each submission has some overhead. With many small kernels, this overhead dominates.

### GPU Graph Batching

As discussed earlier, `TinyJit` can batch multiple kernels into a single GPU command:

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(a, b):
  x = (a + b).realize()  # Kernel 1
  y = (x * a).realize()  # Kernel 2
  z = (y - b).realize()  # Kernel 3
  return z

a = Tensor.empty(4, 4)
b = Tensor.empty(4, 4)

for i in range(4):
  print(f"--- Run {i+1} ---")
  f(a, b)
```

Output (with `DEBUG=2` on METAL backend):
```
--- Run 1 ---
*** METAL      1 E_4_4                                          arg  3 mem   0.00 GB tm     12.50us/     0.01ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL      2 E_4_4n1                                        arg  3 mem   0.00 GB tm     11.87us/     0.02ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL      3 E_4_4n2                                        arg  3 mem   0.00 GB tm     11.88us/     0.04ms (      0 GFLOPS    0|0      GB/s) ['__sub__']
--- Run 2 ---
*** METAL      4 E_4_4                                          arg  3 mem   0.00 GB tm      8.75us/     0.04ms (      0 GFLOPS    0|0      GB/s) ['__add__']
*** METAL      5 E_4_4n1                                        arg  3 mem   0.00 GB tm      8.87us/     0.05ms (      0 GFLOPS    0|0      GB/s) ['__mul__']
*** METAL      6 E_4_4n2                                        arg  3 mem   0.00 GB tm      8.79us/     0.06ms (      0 GFLOPS    0|0      GB/s) ['__sub__']
JIT captured 3 kernels with 2 inputs
--- Run 3 ---
JIT GRAPHing batch with 3 kernels on device <tinygrad.runtime.ops_metal.MetalDevice object at 0x10bf14550>
*** METAL      7 <batched 3>                                    arg  2 mem   0.00 GB tm     40.00us/     0.10ms (      0 GFLOPS    0|0      GB/s)
--- Run 4 ---
*** METAL      8 <batched 3>                                    arg  2 mem   0.00 GB tm     36.46us/     0.14ms (      0 GFLOPS    0|0      GB/s)
```

Three separate kernels become one `<batched 3>` command.

### Performance Impact

```python
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import Timing

def make_chain(n):
  @TinyJit
  def f(x):
    # Create a chain of n simple kernels
    for _ in range(n):
      x = (x + 1).realize()
    return x
  return f

a = Tensor.empty(4, 4).realize()

for n_kernels in [10, 30, 50, 100]:
  f = make_chain(n_kernels)
  # Warmup + capture + graph build
  for _ in range(5):
    f(a)
  # Measure steady state
  with Timing(f"{n_kernels} kernels: "):
    f(a)
```

Output:
```
10 kernels:   0.33 ms
30 kernels:   0.46 ms
50 kernels:   0.49 ms
100 kernels:   0.61 ms
```

Going from 10 to 50 kernels (5x more), time barely increases (1.5x): the submission overhead is paid once regardless of kernel count.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Purpose** | Skip scheduling overhead by replaying captured kernel sequence |
| **Three phases** | Warmup (call 1), capture (call 2), replay (call 3+) |
| **What's swapped** | Input buffers (function arguments) |
| **What's baked in** | Closures, Python values, conditional branches |
| **Dynamic shapes** | Use `Variable("name", min, max).bind(value)` |
| **Output reuse** | Use `.clone().realize()` to keep old results |
| **Mutable state** | Use `.assign()` to update buffers in place |
| **Shape change** | Use `.reset()` to clear captured state |
| **GPU graph** | Batches multiple kernels into single GPU command |
