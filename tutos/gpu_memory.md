# GPU Memory Fundamentals

> [!NOTE]
> This tutorial has been built around tinygrad's commit [526fd4e](https://github.com/tinygrad/tinygrad/tree/526fd4ec7104eda1ef8114e64d99b2788910a8fd) with a Python 3.11 virtual environment. Since tinygrad is evolving rapidly, if you're following along in code, make sure to checkout that commit in order to avoid any discrepancies.
This tutorial explains GPU memory architecture and performance concepts, with practical examples using tinygrad.

## Table of Contents

1. [The Core Problem: Data Movement](#the-core-problem-data-movement)
2. [GPU Architecture Overview](#gpu-architecture-overview)
3. [Memory Interface: The Highway Between Memory and Compute](#memory-interface-the-highway-between-memory-and-compute)
4. [Memory Speed vs. Bandwidth vs. Interface Width](#memory-speed-vs-bandwidth-vs-interface-width)
5. [Memory-Bound vs. Compute-Bound Operations](#memory-bound-vs-compute-bound-operations)
6. [Arithmetic Intensity](#arithmetic-intensity)
7. [Practical Examples with tinygrad](#practical-examples-with-tinygrad)
8. [Profiling GPU Operations](#profiling-gpu-operations)

---

## The Core Problem: Data Movement

When your GPU runs a computation, it performs three fundamental steps:

1. **Read** data from GPU memory (VRAM)
2. **Compute** on that data (using processing units like CUDA cores)
3. **Write** results back to memory

A common misconception is that GPUs are always limited by how fast they can compute. In reality, **moving data between memory and processing units is often the bottleneck**.

Think of it this way: you have the world's fastest chef (the GPU cores), but if ingredients (data) arrive slowly from the pantry (VRAM), the chef spends most of their time waiting.

---

## GPU Architecture Overview

A simplified view of GPU architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                           GPU                                    │
│                                                                  │
│  ┌──────────────┐                        ┌──────────────────┐   │
│  │              │                        │                  │   │
│  │     VRAM     │◄═══ Memory Interface ══►│  Processing     │   │
│  │   (Memory)   │                        │  Units (Cores)   │   │
│  │              │                        │                  │   │
│  └──────────────┘                        └──────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**VRAM (Video RAM)**: Where your tensors live. When you create a tensor in tinygrad, it's stored here.

**Processing Units**: The computational engines (CUDA cores on NVIDIA, Stream Processors on AMD). These perform the actual math operations.

**Memory Interface**: The pathway connecting memory to processing units. This is where the bottleneck often occurs.

---

## Memory Interface: The Highway Between Memory and Compute

The memory interface is like a highway connecting a warehouse (VRAM) to a factory (processing units).

### A Simple Analogy

Imagine a highway with these properties:
- **Number of lanes**: How many cars can travel side-by-side
- **Speed limit**: How fast each car can go
- **Total throughput**: Total cars per hour = lanes × speed

The GPU memory interface works similarly:
- **Interface width**: Number of parallel data lanes (measured in bits)
- **Memory speed**: How fast each lane transfers data (measured in Gbps)
- **Bandwidth**: Total data throughput (measured in GB/s)

### Visual Example: A Toy GPU

Consider a hypothetical GPU with 3 data lanes, each transferring 8 bits per second:

```
        VRAM                    Memory Interface                   Cores
    ┌──────────┐                                              ┌──────────┐
    │          │    Lane 1: ════ 8 bits/sec ════►             │          │
    │  Data    │    Lane 2: ════ 8 bits/sec ════►             │  Compute │
    │  Storage │    Lane 3: ════ 8 bits/sec ════►             │  Units   │
    │          │                                              │          │
    └──────────┘                                              └──────────┘

    Total Bandwidth = 3 lanes × 8 bits/sec = 24 bits/second
```

---

## Memory Speed vs. Bandwidth vs. Interface Width

These three concepts are related by a simple formula:

```
Bandwidth = Memory Speed × Interface Width
```

Let's define each term precisely:

### Memory Speed (Gbps)

The transfer rate of a single data lane. Measured in **gigabits per second (Gbps)**.

Example: A memory speed of 12 Gbps means each lane transfers 12 billion bits per second.

### Interface Width (bits)

The number of parallel data lanes. Measured in **bits**.

Example: A 128-bit interface means 128 lanes transfer data simultaneously.

### Memory Bandwidth (GB/s)

The total data throughput. Measured in **gigabytes per second (GB/s)**.

This is the product of memory speed and interface width. It is what ultimately determines how fast data can flow to/from processing units.

### Real GPU Example: Apple M1 Pro - 16-Core GPU

Let's examine the Apple M1 Pro - 16-Core GPU, which has the following specs (may vary slightly or be wrong, as different sources report different numbers, but it's fine for the purpose of illustration):

| Specification | Value |
|---------------|-------|
| Memory Speed | 6.40 Gbps |
| Memory Bandwidth | 205 GB/s |
| Interface Width | 256 bits |
| FP32 Performance | 5.3 TFLOPS |

Let's verify the math:

```
Given:
- Memory Bandwidth = 205 GB/s = 205 billion bytes/second total
- Memory Interface = 256 bits (i.e., 256 lanes)

Step 1: Convert bandwidth from bytes to bits
  205 GB/s = 205 × 8 Gbps = 1640 Gbps = 1640*10^9 bits/second

Step 2: Calculate memory speed
  Memory Speed = Bandwidth / Interface Width
  Memory Speed = 1640*10^9 / 256 = 6.40*10^9 bits/second = 6.40 Gbps (matches the spec!)
```

> [!IMPORTANT]
> In this tutorial, we make the distinction between *bits* (b) and *bytes* (B). Remember:
> - 1 byte = 8 bits
> - Bandwidth is typically expressed in *bytes per second (B/s)*, while memory speed is in *bits per second (bps)*.
> Also, we use the notation *FLOP* for *floating-point operations*, and *FLOPS* (notice the uppercase 'S') for *floating-point operations per second*. So, in particular, FLOP corresponds to a number of operations, while FLOPS corresponds to a speed. Make sure to keep these distinctions in mind when reading the rest of this tutorial.

---

## Memory-Bound vs. Compute-Bound Operations

Understanding whether an operation is limited by memory or compute is crucial for optimization.

### Memory-Bound Operations

When data transfer takes longer than computation.

**Example: Vector Addition**

Adding two vectors of 1 million floats:

```
Operation: C = A + B
- Read A: 1M floats × 4 bytes = 4 MB
- Read B: 1M floats × 4 bytes = 4 MB
- Write C: 1M floats × 4 bytes = 4 MB
- Total data movement: 12 MB
- Total compute: 1 million additions = 1 MFLOP
```

On our example GPU (205 GB/s bandwidth, 5.3 TFLOPS):

```
Memory time = 12 MB / 205 GB/s = 0.0000585 sec = 58.5 μs
Compute time = 1 MFLOP / 5.3 TFLOPS = 0.000000188 sec = 0.188 μs

Ratio: Memory takes more than 300x longer than compute (under ideal conditions, *i.e.*, supposing that we can fully utilize both the memory bandwidth and the compute speed, which is not possible in practice)!
```

This operation is hence *heavily memory-bound*. The processing units are idle most of the time, waiting for data.

### Compute-Bound Operations

When computation takes longer than data transfer.

**Example: Large Matrix Multiplication**

Multiplying two 4096×4096 matrices:

```
Operation: C = A @ B (matrix multiplication)
- Read A: 4096² floats × 4 bytes = 64 MB
- Read B: 4096² floats × 4 bytes = 64 MB
- Write C: 4096² floats × 4 bytes = 64 MB
- Total data movement: 192 MB
- Total compute: 2 × 4096³ ≈ 137 billion FLOP
```

> [!IMPORTANT]
> Contrary to vector addition, matrix multiplication has to load each element of the input matrices multiple times to compute the output elements. For example, to compute the $i^{\textrm{th}}$ row of the output matrix $C$, each element of the $i^{\textrm{th}}$ row of input matrix $A$ has to be loaded exactly $n = 4096$ times, and not just once. Though we will come back to this later, for now let's assume that we only need to load each element once from memory, and we only need to store each element once to memory as well, *i.e.* assume we have a powerful memory cache on the GPU that allow us to do this.

On our example GPU:

```
Memory time = 192 MB / 205 GB/s = 0.000936 sec = 936 μs = 0.936 ms
Compute time = 137 GFLOP / 5.3 TFLOPS = 0.02585 sec = 25,850 μs = 25.85 ms

Ratio: Compute takes about 27x longer than memory! (again, as for vector addition, under ideal conditions)
```

This operation is hence *compute-bound*. The memory interface can keep up with the processing units' demands.

---

## Arithmetic Intensity

Arithmetic intensity is a metric that predicts whether an operation will be memory-bound or compute-bound, regardless of the specific hardware it runs on. This is very useful for algorithm design and optimization.

```
Arithmetic Intensity = FLOP / Bytes Transferred
```

### Calculating Arithmetic Intensity

**Vector Addition:**
```
Intensity = 1 MFLOP / 12 MB = 0.083 FLOP/byte
```
Very low intensity → Memory-bound

**Matrix Multiplication (n×n):**
```
FLOP = 2n³ (each output element needs n multiply-adds)
Bytes = 3n² × 4 (read 2 matrices, write 1, assuming float32 and ideal memory caching)

Intensity = 2n³ / (12n²) = n/6 FLOP/byte
```

| Matrix Size | Arithmetic Intensity |
|-------------|---------------------|
| 64×64       | 10.7 FLOP/byte     |
| 256×256     | 42.7 FLOP/byte     |
| 1024×1024   | 170.7 FLOP/byte    |
| 4096×4096   | 682.7 FLOP/byte    |

Larger matrices have higher intensity → More likely to be compute-bound.

---

## Practical Examples with tinygrad

Let's see these concepts in action with tinygrad code.

> [!NOTE]
> To know more about benchmarking in tinygrad, check out my [benchmarking guide](https://github.com/cverrier/tinygrad-tutos/blob/main/tutos/benchmark.md).

Before diving into examples, here are the key principles for accurate GPU benchmarking:
1. Pre-realize inputs: Call `.realize()` on input tensors before timing to exclude CPU→GPU data transfer from measurements.
2. Use `TinyJit`: Wrapping operations in `@TinyJit` optimizes the computation graph and enables fair repeated measurements.
3. Warmup runs: The first execution compiles kernels (expensive!). Run 5-10 warmup iterations before measuring.
4. Use hardware timestamps: tinygrad's `GlobalCounters` captures GPU kernel execution time, isolating it from CPU overhead.
5. Multiple iterations: Run several timed iterations and average to reduce noise.

### Example 1: Memory-Bound Operation (Element-wise Addition)

```python
from tinygrad import Context, Tensor, TinyJit
from tinygrad.helpers import GlobalCounters

n = 10_000_000  # 10 million elements
a = Tensor.randn(n)
b = Tensor.randn(n)

# Pre-realize tensors to ensure they're in GPU memory BEFORE timing.
# This excludes CPU→GPU input data transfer from our measurement.
a.realize()
b.realize()

# Wrap in TinyJit for optimized repeated execution
@TinyJit
def add_vectors():
  return (a + b).realize()

# IMPORTANT: Warmup runs!
# - First run compiles the kernel (expensive)
# - TinyJit needs a few runs to trace and optimize the computation graph
for _ in range(5):
  add_vectors()

# Now measure with multiple iterations for statistical reliability
timed_iters = 10
GlobalCounters.reset()
with Context(DEBUG=2):
  for _ in range(timed_iters):
    add_vectors()

kernel_total_us = GlobalCounters.time_sum_s / timed_iters * 1e6  # convert to microseconds
print(f"Vector addition kernel time (hardware timestamps): {kernel_total_us:.3f} us")

bytes_transferred = 3 * n * 4  # 3 vectors × n elements × 4 bytes (float32)
flop = n  # n additions

print(f"Bytes transferred: {bytes_transferred / 1e6:.1f} MB")
print(f"FLOP: {flop / 1e6:.1f} MFLOP")
print(f"Arithmetic intensity: {flop / bytes_transferred:.4f} FLOP/byte")
```

```
*** METAL      1 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    661.83us/     0.66ms (     15 GFLOPS  181|181    GB/s) ['__add__']
*** METAL      2 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    666.13us/     1.33ms (     15 GFLOPS  180|180    GB/s) ['__add__']
*** METAL      3 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    660.08us/     1.99ms (     15 GFLOPS  182|182    GB/s) ['__add__']
*** METAL      4 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    658.75us/     2.65ms (     15 GFLOPS  182|182    GB/s) ['__add__']
*** METAL      5 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    659.46us/     3.31ms (     15 GFLOPS  182|182    GB/s) ['__add__']
*** METAL      6 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    659.88us/     3.97ms (     15 GFLOPS  182|182    GB/s) ['__add__']
*** METAL      7 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    658.21us/     4.62ms (     15 GFLOPS  182|182    GB/s) ['__add__']
*** METAL      8 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    656.67us/     5.28ms (     15 GFLOPS  183|183    GB/s) ['__add__']
*** METAL      9 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    659.75us/     5.94ms (     15 GFLOPS  182|182    GB/s) ['__add__']
*** METAL     10 E_78125_32_4n1                                 arg  3 mem   0.12 GB tm    666.37us/     6.61ms (     15 GFLOPS  180|180    GB/s) ['__add__']
Vector addition kernel time (hardware timestamps): 660.713 us
Bytes transferred: 120.0 MB
FLOP: 10.0 MFLOP
Arithmetic intensity: 0.0833 FLOP/byte
```

You can see from the output that on average, the vector addition kernel takes about 660 microseconds, which is consistent with the per-kernel execution time logs that are shown above (after the `tm`s). The arithmetic intensity is very low (0.0833 FLOP/byte), confirming that this operation is memory-bound: the GPU spends most of its time waiting for data to move rather than performing computations. Let's look closer at this.

I am running on a [Apple M1 Pro 16-Core GPU](https://www.gpu-monkey.com/en/gpu-apple_m1_pro_16_core_gpu), which has a peak memory bandwidth of around 205 GB/s and a peak FP32 performance of about 5.3 TFLOPS. From the logs, we can see that the effective bandwidth during this operation is around 180 GB/s, which is about 90% of the theoretical maximum. However, as shown by the logs, the achieved FLOPS is only about 15 GFLOPS, which is a tiny fraction of the peak compute capability (recall, 5.3 TFLOPS = 5300 GFLOPS), that is, only 0.28%. This discrepancy highlights that the operation is limited by memory bandwidth rather than compute power.

I also want to make sure that the `15 GFLOPS` displayed in the logs are correct and match our calculations:
- 1 FLOP corresponds to one addition in our case.
- To do one addition, we need to load two 32-bit (*i.e.*, 4-byte) floats (one from `a`, one from `b`) from memory, then perform the addition, and finally store one 32-bit float (the result) back to memory: this is a total of 12 bytes transferred per FLOP (therefore, the arithmetic intensity is `1 FLOP / 12 bytes = 0.0833 FLOP/byte`, as we calculated earlier in the script).
- Now, during each kernel execution, tinygrad shows that we have about 180 GB/s of effective memory bandwidth, *i.e.*, we transfer 180 × 10^9 bytes every second. Given an arithmetic intensity of `1 FLOP / 12 bytes = 0.0833 FLOP/byte`, the maximum number of floating-point operations we can perform each second is `180 × 10^9 / 12 = 15 × 10^9` FLOP, which is also 15 GFLOP. So, we do 15 GFLOP per second, which matches the speed of 15 GFLOPS displayed by the logs.

### Example 2: Compute-Bound Operation (Matrix Multiplication)

```python
from tinygrad import Context, Tensor, TinyJit
from tinygrad.helpers import GlobalCounters

n = 2048
a = Tensor.randn(n, n)
b = Tensor.randn(n, n)

# Pre-realize tensors (exclude data transfer from timing)
a.realize()
b.realize()

@TinyJit
def mat_mul():
  return (a @ b).realize()

# Warmup
for _ in range(5):
  mat_mul()

# Timed iterations
timed_iters = 10
GlobalCounters.reset()
with Context(DEBUG=2):
  for _ in range(timed_iters):
    mat_mul()

kernel_total_us = GlobalCounters.time_sum_s / timed_iters * 1e6  # convert to microseconds
print(f"Matrix multiplication kernel time (hardware timestamps): {kernel_total_us:.3f} us")

bytes_transferred = 3 * n**2 * 4  # 3 matrices × n² elements × 4 bytes
flop = 2 * n**3  # 2n³ FLOP for matmul

print(f"Bytes transferred: {bytes_transferred / 1e6:.1f} MB")
print(f"FLOP: {flop / 1e9:.2f} GFLOP")
print(f"Arithmetic intensity: {flop / bytes_transferred:.1f} FLOP/byte")
```

```
*** METAL      1 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5050.29us/     5.05ms (   3402 GFLOPS   10|429    GB/s) ['__matmul__']
*** METAL      2 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5053.46us/    10.10ms (   3400 GFLOPS   10|428    GB/s) ['__matmul__']
*** METAL      3 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5048.08us/    15.15ms (   3403 GFLOPS   10|429    GB/s) ['__matmul__']
*** METAL      4 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5060.75us/    20.21ms (   3395 GFLOPS   10|428    GB/s) ['__matmul__']
*** METAL      5 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5077.00us/    25.29ms (   3384 GFLOPS   10|426    GB/s) ['__matmul__']
*** METAL      6 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5070.79us/    30.36ms (   3388 GFLOPS   10|427    GB/s) ['__matmul__']
*** METAL      7 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5081.54us/    35.44ms (   3381 GFLOPS   10|426    GB/s) ['__matmul__']
*** METAL      8 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5077.38us/    40.52ms (   3384 GFLOPS   10|426    GB/s) ['__matmul__']
*** METAL      9 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5057.33us/    45.58ms (   3397 GFLOPS   10|428    GB/s) ['__matmul__']
*** METAL     10 r_64_16_32_4_2_4_4_256                         arg  3 mem   0.05 GB tm   5101.67us/    50.68ms (   3368 GFLOPS   10|424    GB/s) ['__matmul__']
Matrix multiplication kernel time (hardware timestamps): 5067.829 us
Bytes transferred: 50.3 MB
FLOP: 17.18 GFLOP
Arithmetic intensity: 341.3 FLOP/byte
```

> [!NOTE]
> Here, you can see that the pattern `10|429    GB/s` in the logs differ from the previous example: the two numbers (10 and 429) are different. The first number (10 GB/s) corresponds to the memory bandwidth associated with the total bytes of unique memory accessed (*i.e.*, each buffer is counted only once, regardless of how many times its elements are accessed). The second number (429 GB/s) corresponds to the memory bandwidth associated with the total bytes accessed by all load and store instructions in the kernel (*e.g.*, if the same memory location is accessed 1000 times, it counts 1000 times). This follows what we mentioned earlier about matrix multiplication needing to load each element multiple times. The effective bandwidth here is much higher because of these repeated accesses. I'll write some notes about this in a future tutorial.

Exercise: do the same analysis as for vector addition to verify that the `3400 GFLOPS` (more or less) displayed in the logs are correct.

---

## Summary

| Concept | Definition | Unit | Analogy |
|---------|------------|------|---------|
| Memory Speed | Transfer rate per data lane | Gbps | Speed limit per highway lane |
| Interface Width | Number of parallel data lanes | bits | Number of highway lanes |
| Bandwidth | Total data throughput | GB/s | Total cars per hour |
| Floating-point operations | Floating-point operations | FLOP | Factory production |
| Floating-point operations per second | Floating-point operations per second | FLOPS | Factory production rate |
| Arithmetic Intensity | Floating-point operations per byte transferred | FLOP/byte | Work done per supply delivery |

### Key Takeaways

1. **GPU performance has two limits**: memory bandwidth and compute throughput
2. **Arithmetic intensity** determines which limit applies to your operation
3. **Element-wise operations** (add, multiply, activation functions) are typically memory-bound
4. **Matrix operations** (matmul, convolution) are typically compute-bound for large sizes
5. **Profiling** reveals actual bottlenecks

### Optimization Guidelines

- **For memory-bound operations**: Minimize data movement, fuse operations, use lower precision
- **For compute-bound operations**: Maximize parallelism, use tensor cores, optimize algorithms
- **General**: Batch operations together, avoid unnecessary synchronization, keep data on GPU
