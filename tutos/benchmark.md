# Understanding GPU Timing in tinygrad

This guide explains how to accurately measure GPU execution time in tinygrad. We start from first principles and build up to practical benchmarking patterns.

## Table of Contents

1. [Why GPU Timing is Different](#why-gpu-timing-is-different)
2. [Core Concepts](#core-concepts)
3. [Two Ways to Measure Time](#two-ways-to-measure-time)
4. [Practical Benchmarking Patterns](#practical-benchmarking-patterns)
5. [Common Pitfalls](#common-pitfalls)
6. [Complete Example](#complete-example)

---

## Why GPU Timing is Different

### The CPU Model: Synchronous Execution

When you run code on the CPU, each line executes and completes before the next one begins:

```python
import time

st = time.perf_counter()
result = expensive_cpu_function()  # Blocks until done
et = time.perf_counter() - st      # Correct timing
```

This works because `expensive_cpu_function()` is *synchronous*—it doesn't return until the work is finished.

### The GPU Model: Asynchronous Execution

GPUs work differently. When you ask the GPU to do something, the CPU doesn't wait around. Instead:

1. CPU *submits* work to a command queue
2. CPU *immediately returns* and continues executing
3. GPU *processes the queue* in the background

```
CPU Timeline:  [Submit work to GPU] → [Continue other work] → [Eventually sync]
                     ↓
GPU Timeline:        ~~~~~~~~~~~~[Actually doing the work]~~~~~~~~~~~~
```

This means naive timing doesn't work:

```python
import time

st = time.perf_counter()
result = expensive_gpu_function()  # Returns IMMEDIATELY after submitting!
et = time.perf_counter() - st      # WRONG! Only measures submission time
```

You're measuring how long it took to put work in the queue, not how long the work actually took.

---

## Core Concepts

### `Tensor.realize()` — Triggering Computation

In tinygrad, tensors are *lazy*. Operations build up a computation graph without executing anything:

```python
from tinygrad import Tensor

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b  # Nothing computed yet! Just builds a graph.
```

The `realize()` method triggers actual computation:

```python
c.realize()  # NOW the addition happens on the GPU
```

**Critical understanding**: `realize()` is *asynchronous*. It:
1. Builds a schedule of operations
2. Compiles kernels (if not cached)
3. Submits kernels to the GPU command queue
4. *Returns immediately* — does NOT wait for GPU to finish

```python
c.realize()  # Submits work, returns instantly
# GPU is still computing in the background!
```

### `Device.synchronize()` — The Synchronization Barrier

To wait for the GPU to finish all submitted work, you need `synchronize()`:

```python
from tinygrad import Device

c.realize()                           # Submit work
Device[Device.DEFAULT].synchronize()  # Block until GPU is done
# NOW the result is guaranteed to be ready
```

Think of it as a barrier: the CPU stops and waits until the GPU catches up.

```
CPU:  [realize()] → [synchronize() -------- waiting --------] → [continue]
                          ↓                                ↑
GPU:                 [kernel 1] → [kernel 2] → [kernel 3] ─┘
```

### How the GPU Tracks Progress: Signals and Timelines

Under the hood, tinygrad uses *signals* (also called semaphores) to track GPU progress. Each device maintains a `timeline_signal` that increments after each operation:

```
Timeline value:  0 → 1 → 2 → 3 → ...
                 ↑   ↑   ↑   ↑
              start  k1  k2  k3  (kernel completions)
```

When you call `synchronize()`, it waits until the signal reaches the expected value:

```python
# Simplified version of what synchronize() does:
self.timeline_signal.wait(self.timeline_value - 1)
```

This is more efficient than older "flush and wait" approaches because the GPU can signal completion asynchronously.

---

## Two Ways to Measure Time

### Method 1: Wall-Clock Time

Use Python's `time.perf_counter()` with proper synchronization:

```python
import time
from tinygrad import Tensor, Device

a = Tensor.randn(1024, 1024).realize()
b = Tensor.randn(1024, 1024).realize()
Device[Device.DEFAULT].synchronize()  # Ensure inputs are ready

st = time.perf_counter()
c = (a @ b).realize()                  # Submit work
Device[Device.DEFAULT].synchronize()   # Wait for completion
et = time.perf_counter() - st

print(f"Wall-clock time: {et*1000:.2f} ms")
```

*What this measures:*
```
|←──────────────────────────────── Wall-clock time ────────────────────────────────→|

[Schedule] → [Compile*] → [Submit K1] → [K1 runs] → [Submit K2] → [K2 runs] → [Sync]
|←──────────── CPU work ───────────→|   |← GPU →|  |←── CPU ──→|  |← GPU →|   |←CPU→|
```
*Compile only on first run if not cached

This includes:
- Time to build the execution schedule (CPU)
- Time to submit kernels to the queue (CPU)
- Actual kernel execution time (GPU)
- Gaps between kernel submissions (CPU overhead)
- Synchronization overhead (CPU)

### Method 2: Hardware Timestamps

The GPU has its own clock. With `DEBUG=2`, tinygrad uses hardware timestamps to measure only kernel execution:

```python
from tinygrad import Context, Device, Tensor
from tinygrad.helpers import GlobalCounters

a = Tensor.randn(1024, 1024).realize()
b = Tensor.randn(1024, 1024).realize()
Device[Device.DEFAULT].synchronize()

GlobalCounters.reset()

with Context(DEBUG=2):
  c = (a @ b).realize()
  Device[Device.DEFAULT].synchronize()

print(f"GPU kernel time: {GlobalCounters.time_sum_s*1000:.2f} ms")
```

```
*** METAL      1 r_32_8_32_4_2_4_4_128                          arg  3 mem   0.01 GB tm   2013.04us/     2.01ms (   1067 GFLOPS    6|135    GB/s) ['__matmul__']
GPU kernel time: 2.01 ms
```

*What this measures:*
```
[Schedule] → [Compile*] → [Submit K1] → [K1 runs] → [Submit K2] → [K2 runs] → [Sync]
                                        |← timed →|              |← timed →|

                                        |←──── GlobalCounters.time_sum_s ────→|
```

The timestamps are recorded *on the GPU itself*, measuring only the time the GPU spent actively executing kernels.

### Comparing the Two Methods

```python
import time

from tinygrad import Context, Device, Tensor
from tinygrad.helpers import GlobalCounters

# Setup
a = Tensor.randn(2048, 2048).realize()
b = Tensor.randn(2048, 2048).realize()
Device[Device.DEFAULT].synchronize()

# Warmup to avoid startup overhead
for i in range(5):
  c = (a @ b).realize()
  Device[Device.DEFAULT].synchronize()

# Measure both
GlobalCounters.reset()
st = time.perf_counter()
with Context(DEBUG=2):
  c = (a @ b).realize()
  Device[Device.DEFAULT].synchronize()
wall_time = time.perf_counter() - st
gpu_time = GlobalCounters.time_sum_s

print(f"Wall-clock time: {wall_time*1000:.3f} ms")
print(f"GPU kernel time: {gpu_time*1000:.3f} ms")
print(f"Overhead:        {(wall_time - gpu_time)*1000:.3f} ms")
```

Sample output:
```
*** METAL      6 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.07 GB tm   5022.92us/     5.02ms (   3420 GFLOPS   10|431    GB/s) ['__matmul__']
Wall-clock time: 6.456 ms
GPU kernel time: 5.023 ms
Overhead:        1.433 ms
```

### When to Use Each Method

| Method | Use When |
|--------|----------|
| Wall-clock | You care about *end-to-end latency* (user-facing response time) |
| Hardware timestamps | You want to measure *GPU efficiency* (comparing to theoretical peak) |

The *gap* between them tells you about CPU overhead:
- Large gap → kernel launch overhead is significant; consider kernel fusion or `TinyJit`
- Small gap → GPU-bound workload; optimize the kernels themselves

---

## Practical Benchmarking Patterns

### Pattern 1: Basic GPU Timing (Excluding Data Transfer)

The key is to *pre-realize inputs* before starting the timer:

```python
import time
from tinygrad import Tensor, Device

def benchmark_matmul(size):
  # Step 1: Create and transfer data to GPU BEFORE timing
  a = Tensor.randn(size, size).realize()
  b = Tensor.randn(size, size).realize()
  Device[Device.DEFAULT].synchronize()  # Ensure transfer is complete

  # Step 2: Time only the computation
  st = time.perf_counter()
  c = (a @ b).realize()
  Device[Device.DEFAULT].synchronize()
  et = time.perf_counter() - st

  return et

print(f"1024x1024 matmul: {benchmark_matmul(1024)*1000:.2f} ms")
```

### Pattern 2: Multiple Iterations for Reliable Results

A single measurement can be noisy. Run multiple times and take the best (or average):

```python
import time
from tinygrad import Tensor, Device

def benchmark_reliable(fn, warmup=3, iterations=10):
  """Benchmark a function with warmup and multiple iterations."""

  # Warmup runs (may include JIT compilation)
  for _ in range(warmup):
    fn()
    Device[Device.DEFAULT].synchronize()

  # Timed runs
  times = []
  for _ in range(iterations):
    st = time.perf_counter()
    fn()
    Device[Device.DEFAULT].synchronize()
    times.append(time.perf_counter() - st)

  return min(times), sum(times)/len(times)  # best, average

# Usage
a = Tensor.randn(1024, 1024).realize()
b = Tensor.randn(1024, 1024).realize()
Device[Device.DEFAULT].synchronize()

def matmul(): return (a @ b).realize()

best, avg = benchmark_reliable(matmul)
print(f"Best: {best*1000:.2f} ms, Average: {avg*1000:.2f} ms")
```

### Pattern 3: Using DEBUG=2 for Kernel-Level Timing

Get hardware-measured times for each kernel:

```python
from tinygrad import Context, Device, Tensor
from tinygrad.helpers import GlobalCounters

a = Tensor.randn(1024, 1024).realize()
b = Tensor.randn(1024, 1024).realize()
Device[Device.DEFAULT].synchronize()

# Warmup
for _ in range(5):
  (a @ b + a).realize()
  Device[Device.DEFAULT].synchronize()

# Measure
GlobalCounters.reset()
with Context(DEBUG=2):
  result = ((a @ b).realize() + a).realize()  # Force to have two kernels
  Device[Device.DEFAULT].synchronize()

print(f"Total kernel count: {GlobalCounters.kernel_count}")
print(f"Total kernel time:  {GlobalCounters.time_sum_s*1000:.3f} ms")
print(f"Total FLOPS:        {GlobalCounters.global_ops / GlobalCounters.time_sum_s / 1e12:.2f} TFLOPS")
```

With `DEBUG=2`, you also see per-kernel timing printed:
```
*** METAL      1 r_32_8_32_4_2_4_4_128n1                        arg  3 mem   0.01 GB tm   2036.37us/     2.04ms (   1055 GFLOPS    6|134    GB/s) ['__matmul__']
*** METAL      2 E_8192_32_4n1                                  arg  3 mem   0.02 GB tm     96.79us/     2.13ms (     11 GFLOPS  130|130    GB/s) ['__add__']
Total kernel count: 2
Total kernel time:  2.133 ms
Total FLOPS:        1.01 TFLOPS
```

---

## Common Pitfalls

### Pitfall 1: Forgetting to Synchronize

```python
# WRONG - measures submission time only
st = time.perf_counter()
c = (a @ b).realize()
et = time.perf_counter() - st  # GPU still running!

# CORRECT
st = time.perf_counter()
c = (a @ b).realize()
Device[Device.DEFAULT].synchronize()  # Wait for GPU
et = time.perf_counter() - st
```

### Pitfall 2: Including Data Transfer

```python
# WRONG - includes CPU→GPU transfer time
st = time.perf_counter()
a = Tensor.randn(1024, 1024).realize()  # Allocates and transfers!
b = Tensor.randn(1024, 1024).realize()
c = (a @ b).realize()
Device[Device.DEFAULT].synchronize()
et = time.perf_counter() - st

# CORRECT - pre-realize inputs
a = Tensor.randn(1024, 1024).realize()
b = Tensor.randn(1024, 1024).realize()
Device[Device.DEFAULT].synchronize()  # Ensure ready

st = time.perf_counter()
c = (a @ b).realize()
Device[Device.DEFAULT].synchronize()
et = time.perf_counter() - st
```

### Pitfall 3: First-Run Compilation Overhead

The first execution may include kernel compilation:

```python
# First run - includes compilation
st = time.perf_counter()
c = (a @ b).realize()
Device[Device.DEFAULT].synchronize()
first_time = time.perf_counter() - st  # Slow! Includes compile.

# Second run - uses cached kernel
st = time.perf_counter()
d = (a @ b).realize()
Device[Device.DEFAULT].synchronize()
second_time = time.perf_counter() - st  # Fast! Kernel cached.

print(f"First:  {first_time*1000:.2f} ms (includes compile)")
print(f"Second: {second_time*1000:.2f} ms (cached)")
```

*Solution*: Always warmup before measuring:
```python
# Warmup
for _ in range(3):
  (a @ b).realize()
  Device[Device.DEFAULT].synchronize()

# Now measure
```

### Pitfall 4: Not Resetting GlobalCounters

```python
# WRONG - accumulates across multiple measurements
with Context(DEBUG=2):
  (a @ b).realize()
  Device[Device.DEFAULT].synchronize()

with Context(DEBUG=2):
  (a @ b).realize()
  Device[Device.DEFAULT].synchronize()

print(GlobalCounters.time_sum_s)  # Sum of BOTH operations!

# CORRECT - reset before measuring
GlobalCounters.reset()
with Context(DEBUG=2):
  (a @ b).realize()
  Device[Device.DEFAULT].synchronize()
print(GlobalCounters.time_sum_s)  # Just this operation
```

---

## Complete Example

Here's a complete benchmarking script that demonstrates all concepts:

```python
#!/usr/bin/env python3
"""
Complete GPU benchmarking example for tinygrad.

Demonstrates:
- Proper synchronization
- Pre-realizing inputs
- Warmup runs
- Wall-clock vs hardware timing
- Multiple iterations
"""

import time

from tinygrad import Context, Device, Tensor
from tinygrad.helpers import GlobalCounters


def benchmark_operation(name, setup_fn, operation_fn, warmup=5, iterations=20):
  """
  Benchmark a GPU operation with proper methodology.

  Args:
    name: Name for display
    setup_fn: Function that returns input tensors (called once)
    operation_fn: Function that takes inputs and returns output tensor
    warmup: Number of warmup iterations
    iterations: Number of timed iterations
  """
  dev = Device[Device.DEFAULT]

  # Setup: create inputs and ensure they're on GPU
  inputs = setup_fn()
  if isinstance(inputs, Tensor):
    inputs = (inputs,)
  for t in inputs:
    t.realize()
  dev.synchronize()

  # Warmup: run several times to ensure kernels are compiled
  print(f"Warming up {name}...", end=" ", flush=True)
  for _ in range(warmup):
    operation_fn(*inputs).realize()
    dev.synchronize()
  print("done")

  # Measure wall-clock time
  wall_times = []
  for _ in range(iterations):
    st = time.perf_counter()
    operation_fn(*inputs).realize()
    dev.synchronize()
    wall_times.append(time.perf_counter() - st)

  # Compute statistics
  best_wall = min(wall_times)
  avg_wall = sum(wall_times) / len(wall_times)

  # Get kernel time from one run with DEBUG=2
  GlobalCounters.reset()
  with Context(DEBUG=2):
    operation_fn(*inputs).realize()
    dev.synchronize()
  kernel_time = GlobalCounters.time_sum_s
  kernel_count = GlobalCounters.kernel_count
  flops = GlobalCounters.global_ops

  # Report
  print(f"\n{'='*60}")
  print(f"Benchmark: {name}")
  print(f"{'='*60}")
  print(f"Wall-clock time (best):    {best_wall*1000:8.3f} ms")
  print(f"Wall-clock time (average): {avg_wall*1000:8.3f} ms")
  print(f"GPU kernel time:           {kernel_time*1000:8.3f} ms")
  print(f"CPU overhead:              {(best_wall - kernel_time)*1000:8.3f} ms")
  print(f"Kernel count:              {kernel_count:8d}")
  if kernel_time > 0:
    print(f"Throughput:                {flops / kernel_time / 1e12:8.2f} TFLOPS")
  print()

  return best_wall, kernel_time


if __name__ == "__main__":
  print(f"Device: {Device.DEFAULT}\n")

  # Benchmark 1: Matrix multiplication
  benchmark_operation(
    name="Matrix Multiplication (2048x2048)",
    setup_fn=lambda: (Tensor.randn(2048, 2048), Tensor.randn(2048, 2048)),
    operation_fn=lambda a, b: a @ b,
  )

  # Benchmark 2: Element-wise operations
  benchmark_operation(
    name="Element-wise (10M elements): a*b + c",
    setup_fn=lambda: (Tensor.randn(10_000_000), Tensor.randn(10_000_000), Tensor.randn(10_000_000)),
    operation_fn=lambda a, b, c: a * b + c,
  )

  # Benchmark 3: Reduction
  benchmark_operation(
    name="Sum reduction (100M elements)",
    setup_fn=lambda: Tensor.randn(100_000_000),
    operation_fn=lambda a: a.sum(),
  )

  # Benchmark 4: Convolution
  benchmark_operation(
    name="Conv2D (64x64x64 image, 3x3 kernel, 128 channels)",
    setup_fn=lambda: (Tensor.randn(1, 64, 64, 64), Tensor.randn(128, 64, 3, 3)),
    operation_fn=lambda x, w: x.conv2d(w),
  )
```

```
Device: METAL

Warming up Matrix Multiplication (2048x2048)... done
*** METAL      1 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5083.12us/     5.08ms (   3380 GFLOPS   10|426    GB/s) ['__matmul__']

============================================================
Benchmark: Matrix Multiplication (2048x2048)
============================================================
Wall-clock time (best):       5.709 ms
Wall-clock time (average):    5.806 ms
GPU kernel time:              5.083 ms
CPU overhead:                 0.626 ms
Kernel count:                     1
Throughput:                    3.38 TFLOPS

Warming up Element-wise (10M elements): a*b + c... done
*** METAL      1 E_78125_32_4n1                                 arg  4 mem   0.16 GB tm    893.87us/     0.89ms (     22 GFLOPS  179|179    GB/s) ['__mul__', '__add__']

============================================================
Benchmark: Element-wise (10M elements): a*b + c
============================================================
Wall-clock time (best):       1.321 ms
Wall-clock time (average):    1.407 ms
GPU kernel time:              0.894 ms
CPU overhead:                 0.427 ms
Kernel count:                     1
Throughput:                    0.02 TFLOPS

Warming up Sum reduction (100M elements)... done
scheduled    3 kernels in     0.16 ms |  cache hit 8bfe947f | 5492 uops in cache
*** METAL      1 r_250_32_4_3125                                arg  2 mem   0.40 GB tm     10.81ms/    10.81ms (      9 GFLOPS   37|37     GB/s)
*** METAL      2 r_2_32_4_125                                   arg  2 mem   0.40 GB tm     15.75us/    10.83ms (      2 GFLOPS    8|8      GB/s)
*** METAL      3 r_16_16                                        arg  2 mem   0.40 GB tm      4.12us/    10.83ms (      0 GFLOPS    0|1      GB/s)

============================================================
Benchmark: Sum reduction (100M elements)
============================================================
Wall-clock time (best):      11.378 ms
Wall-clock time (average):   11.541 ms
GPU kernel time:             10.832 ms
CPU overhead:                 0.546 ms
Kernel count:                     3
Throughput:                    0.01 TFLOPS

Warming up Conv2D (64x64x64 image, 3x3 kernel, 128 channels)... done
*** METAL      1 r_31_31_32_2_2_4_64_3_3                        arg  3 mem   0.00 GB tm   7521.79us/     7.52ms (     75 GFLOPS    0|189    GB/s) ['conv2d']

============================================================
Benchmark: Conv2D (64x64x64 image, 3x3 kernel, 128 channels)
============================================================
Wall-clock time (best):       8.441 ms
Wall-clock time (average):    8.695 ms
GPU kernel time:              7.522 ms
CPU overhead:                 0.919 ms
Kernel count:                     1
Throughput:                    0.08 TFLOPS
```

---

## Summary

| Concept | Description |
|---------|-------------|
| `realize()` | Submits work to GPU, returns immediately (async) |
| `synchronize()` | Blocks CPU until GPU finishes all work |
| Wall-clock timing | Measures total elapsed time including CPU overhead |
| Hardware timestamps | Measures only GPU kernel execution time |
| `DEBUG=2` | Enables per-kernel timing output and populates `GlobalCounters.time_sum_s` |
| `GlobalCounters.reset()` | Clears accumulated timing statistics |
| Warmup | Always run a few iterations before measuring to exclude compilation time |
| Pre-realize inputs | Move data to GPU before starting the timer to exclude transfer time |
