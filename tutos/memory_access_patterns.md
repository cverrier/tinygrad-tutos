# CUDA Memory Access Patterns: A Deep Dive

> [!NOTE]
> This tutorial is still work in progress, and might not be fully accurate yet.

This document explains memory access patterns in CUDA kernels, using a simple matrix multiplication kernel as a case study.

## Table of Contents

1. [The Kernel Under Study](#the-kernel-under-study)
2. [Row-Major Storage](#row-major-storage)
3. [Memory Access Patterns](#memory-access-patterns)
4. [Cache Lines](#cache-lines)
5. [Warps and Coalescing](#warps-and-coalescing)
6. [CUDA Memory Hierarchy](#cuda-memory-hierarchy)
7. [Warp Shuffle Instructions](#warp-shuffle-instructions)
8. [Summary](#summary)

---

## The Kernel Under Study

Consider this simple (unoptimized) matrix multiplication kernel:

```cuda
__global__ void linear(float* out, const float* __restrict__ X,
                       const float* __restrict__ W, const float* __restrict__ B,
                       int n, int k, int m)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= m) return;

    float acc = B[col];

    for (int i = 0; i < k; ++i) {
        acc += X[row * k + i] * W[i * m + col];
    }

    out[row * m + col] = acc;
}
```

Each thread computes one element of the output matrix by computing a dot product between a row of X and a column of W.

---

## Row-Major Storage

In C/C++ and CUDA, 2D arrays are stored in **row-major order**: rows are laid out contiguously in memory.

For a matrix `M` with shape `(rows, cols)`:
- Element `M[i][j]` is at memory address: `base + i * cols + j`
- Consecutive elements in a row are adjacent in memory
- Consecutive elements in a column are `cols` elements apart

**Example:** Matrix W with shape (4, 8):

```
Logical view:                     Memory layout:

     col: 0   1   2   3   4   5   6   7        Address: 0  1  2  3  4  5  6  7  8  9 10 11 ...
        ┌───┬───┬───┬───┬───┬───┬───┬───┐            ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
row 0   │ a │ b │ c │ d │ e │ f │ g │ h │    →       │ a│ b│ c│ d│ e│ f│ g│ h│ i│ j│ k│ l│...
        ├───┼───┼───┼───┼───┼───┼───┼───┤            └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
row 1   │ i │ j │ k │ l │ m │ n │ o │ p │              ↑                       ↑
        ├───┼───┼───┼───┼───┼───┼───┼───┤           row 0                   row 1
row 2   │ q │ r │ s │ t │ u │ v │ w │ x │           (contiguous)            (contiguous)
        ├───┼───┼───┼───┼───┼───┼───┼───┤
row 3   │ y │ z │...│   │   │   │   │   │
        └───┴───┴───┴───┴───┴───┴───┴───┘

Row access (e.g., row 0): a, b, c, d, e, f, g, h → addresses 0, 1, 2, 3, 4, 5, 6, 7 (stride = 1)
Column access (e.g., col 2): c, k, s, ... → addresses 2, 10, 18, 26 (stride = 8 = num_cols)
```

**Key insight:** Accessing a row is fast (consecutive memory), accessing a column is slow (strided memory).

---

## Memory Access Patterns

Let's trace what a single thread accesses in our kernel.

**Setup:**
- `k = 4` (input features)
- `m = 8` (output features)
- Thread computing output at `row_idx = 0`, `col_idx = 2`

### X Access Pattern

```cpp
for (int i = 0; i < k; ++i)
    X[row_idx * k + i]  // X[0*4 + i] = X[i]
```

```
Iteration 0: X[0]
Iteration 1: X[1]
Iteration 2: X[2]
Iteration 3: X[3]
```

```
Memory:  X[0]  X[1]  X[2]  X[3]  X[4]  X[5]  ...
           ★     ★     ★     ★
          i=0   i=1   i=2   i=3

★ = accessed elements
Stride = 1 (CONSECUTIVE - GOOD!)
```

The thread reads along a **row** of X → consecutive memory access.

### W Access Pattern

```cpp
for (int i = 0; i < k; ++i)
    W[i * m + col_idx]  // W[i*8 + 2]
```

```
Iteration 0: W[0*8 + 2] = W[2]
Iteration 1: W[1*8 + 2] = W[10]
Iteration 2: W[2*8 + 2] = W[18]
Iteration 3: W[3*8 + 2] = W[26]
```

```
Memory:  W[0]  W[1]  W[2]  W[3]  ...  W[10] ...  W[18] ...  W[26]
                       ★              ★          ★          ★
                      i=0            i=1        i=2        i=3

★ = accessed elements
Stride = 8 (STRIDED - BAD!)
```

The thread reads down a **column** of W → strided memory access with stride `m`.

---

## Cache Lines

GPU memory is fetched in **cache lines** (typically 128 bytes = 32 floats). When you request one float, the hardware fetches the entire cache line containing it.

### X: Excellent Cache Utilization

```
Thread requests X[0]:
┌─────────────────────────────────────────────────────────────────────────┐
│ Cache line fetched: X[0]  X[1]  X[2]  X[3]  X[4]  X[5]  ... X[31]      │
│                       ★     ★     ★     ★                               │
│                      i=0   i=1   i=2   i=3                              │
│                                                                         │
│ All 4 iterations served from ONE cache line fetch!                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### W: Poor Cache Utilization (with large m)

Consider `m = 512`:

```
Iteration 0: W[2]    → Fetch cache line containing W[0-31]
Iteration 1: W[514]  → Fetch cache line containing W[512-543]
Iteration 2: W[1026] → Fetch cache line containing W[1024-1055]
Iteration 3: W[1538] → Fetch cache line containing W[1536-1567]
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Cache line 0:   W[0]   W[1]   W[2]   ...  W[31]                         │
│                               ★ (only this one used!)                   │
│                                                                         │
│ Cache line 16:  W[512] W[513] W[514] ...  W[543]                        │
│                               ★ (only this one used!)                   │
│                                                                         │
│ Cache line 32:  W[1024] W[1025] W[1026] ... W[1055]                     │
│                                  ★ (only this one used!)                │
│                                                                         │
│ Each iteration fetches 32 floats but uses only 1!                       │
│ Memory efficiency: 1/32 = 3.125%                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Warps and Coalescing

### What is a Warp?

A **warp** is a group of 32 threads that execute in lockstep. When these threads issue memory requests, the GPU **coalesces** them into as few transactions as possible.

### Thread Block to Warp Mapping

Threads in a block are linearized as: `linear_id = threadIdx.x + threadIdx.y * blockDim.x`

**Example:** `blockDim = (8, 32)`, first warp (linear_id 0-31):

```
┌──────────────────────────────────────────────────────────────────────┐
│ linear_id │ threadIdx.x │ threadIdx.y │ row_idx │ col_idx            │
├───────────┼─────────────┼─────────────┼─────────┼────────────────────┤
│     0     │      0      │      0      │    0    │    0               │
│     1     │      1      │      0      │    1    │    0               │
│    ...    │     ...     │     ...     │   ...   │   ...              │
│     7     │      7      │      0      │    7    │    0               │
│     8     │      0      │      1      │    0    │    1               │
│    ...    │     ...     │     ...     │   ...   │   ...              │
│    15     │      7      │      1      │    7    │    1               │
│    16     │      0      │      2      │    0    │    2               │
│    ...    │     ...     │     ...     │   ...   │   ...              │
│    31     │      7      │      3      │    7    │    3               │
└──────────────────────────────────────────────────────────────────────┘

col_idx values in this warp: 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
```

### Coalesced Access to W

At iteration `i = 0` with `m = 512`:

```
Thread 0-7:   W[0*512 + 0] = W[0]
Thread 8-15:  W[0*512 + 1] = W[1]
Thread 16-23: W[0*512 + 2] = W[2]
Thread 24-31: W[0*512 + 3] = W[3]
```

```
Memory:  W[0]  W[1]  W[2]  W[3]  W[4]  W[5]  ...
           ↑     ↑     ↑     ↑
           │     │     │     │
         T0-7  T8-15 T16-23 T24-31

These addresses are CONSECUTIVE → COALESCED (good for this iteration)
```

### The Problem: No Temporal Locality

Even though each iteration is coalesced, look at access patterns across iterations:

```
Memory address
      ↑
 1536 │                                 ████  ← iteration 3
      │
 1024 │                         ████          ← iteration 2
      │
  512 │             ████                      ← iteration 1
      │
    0 │ ████                                  ← iteration 0
      └──────────────────────────────────────→ time

Each iteration jumps m=512 addresses forward.
Data from iteration 0 is NEVER reused in iteration 1.
```

### Comparison: X vs W

| Aspect                        | X Access            | W Access                      |
|-------------------------------|---------------------|-------------------------------|
| Per-thread across iterations  | Consecutive (good)  | Strided by m (bad)            |
| Cache line reuse              | High                | None                          |
| Warp coalescing per iteration | May be strided      | Consecutive (if col_idx varies) |

---

## CUDA Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              GPU DEVICE                                 │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                         GLOBAL MEMORY                             │  │
│  │                    (Accessible by ALL threads)                    │  │
│  │                         ~400 cycles                               │  │
│  │                          8-80 GB                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  ↑↓                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          L2 CACHE                                 │  │
│  │                     ~200 cycles, 4-6 MB                           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  ↑↓                                     │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────┐ │
│  │  SM 0 (Streaming Multiprocessor)│    │           SM 1              │ │
│  │                                 │    │                             │ │
│  │  ┌───────────────────────────┐  │    │  ┌───────────────────────┐  │ │
│  │  │  L1 CACHE  (~30 cycles)   │  │    │  │       L1 CACHE        │  │ │
│  │  └───────────────────────────┘  │    │  └───────────────────────┘  │ │
│  │                                 │    │                             │ │
│  │  ┌───────────────────────────┐  │    │  ┌───────────────────────┐  │ │
│  │  │     SHARED MEMORY         │  │    │  │    SHARED MEMORY      │  │ │
│  │  │  ~20 cycles, 48-164 KB    │  │    │  │                       │  │ │
│  │  │   (per thread block)      │  │    │  │                       │  │ │
│  │  └───────────────────────────┘  │    │  └───────────────────────┘  │ │
│  │                                 │    │                             │ │
│  │  ┌───────┐ ┌───────┐ ┌───────┐  │    │                             │ │
│  │  │ Warp0 │ │ Warp1 │ │ Warp2 │  │    │                             │ │
│  │  │┌─┬─┬─┐│ │┌─┬─┬─┐│ │┌─┬─┬─┐│  │    │                             │ │
│  │  ││R│R│R││ ││R│R│R││ ││R│R│R││  │    │                             │ │
│  │  │└─┴─┴─┘│ │└─┴─┴─┘│ │└─┴─┴─┘│  │    │                             │ │
│  │  └───────┘ └───────┘ └───────┘  │    │                             │ │
│  │      ↑                          │    │                             │ │
│  │   Registers (~1 cycle)          │    │                             │ │
│  │   (per thread, ~256 KB/SM)      │    │                             │ │
│  └─────────────────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Types

| Memory Type      | Scope       | Latency     | Size            |
|------------------|-------------|-------------|-----------------|
| **Registers**    | Per thread  | ~1 cycle    | ~256 KB per SM  |
| **Shared Memory**| Per block   | ~20 cycles  | 48-164 KB per SM|
| **L1 Cache**     | Per SM      | ~30 cycles  | 32-128 KB per SM|
| **L2 Cache**     | All SMs     | ~200 cycles | 4-6 MB          |
| **Global Memory**| All threads | ~400 cycles | 8-80 GB         |

### Key Point: No Warp-Specific Memory

Warps are a **scheduling/execution** concept, not a memory concept:

```
Thread Block (256 threads)
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Warp 0          Warp 1          Warp 2          ...           │
│  ┌────────┐      ┌────────┐      ┌────────┐                     │
│  │ T0-T31 │      │T32-T63 │      │T64-T95 │                     │
│  └────────┘      └────────┘      └────────┘                     │
│       │               │               │                         │
│       └───────────────┼───────────────┘                         │
│                       ↓                                         │
│          ┌─────────────────────────┐                            │
│          │      SHARED MEMORY      │  ← ALL warps in the block  │
│          │                         │    share this memory       │
│          └─────────────────────────┘                            │
│                                                                 │
│   There is NO warp-specific memory between registers and        │
│   shared memory.                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Warp Shuffle Instructions

While warps don't have their own memory, they can exchange **register values** directly using **warp shuffle instructions**. This is faster than going through shared memory.

### Example: `__shfl_sync`

```cuda
float x = 5.0f;  // Each thread has its own x in a register
float y = __shfl_sync(0xffffffff, x, 3);  // All threads get thread 3's value of x
```

### How It Works

```
Before __shfl_sync(mask, x, 3):
┌─────────────────────────────────────────────────────────────────┐
│ Warp                                                            │
│ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐               │
│ │ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ ... │ T31 │               │
│ │ x=1 │ x=2 │ x=3 │ x=4 │ x=5 │ x=6 │     │x=32 │               │
│ └─────┴─────┴─────┴──┬──┴─────┴─────┴─────┴─────┘               │
│                      │                                          │
│                      ↓ broadcast                                │
│         ┌────────────┴────────────┐                             │
│         │  Thread 3's x value (4) │                             │
│         └────────────┬────────────┘                             │
│                      │                                          │
│     ┌────────┬───────┼───────┬────────┬────────┐                │
│     ↓        ↓       ↓       ↓        ↓        ↓                │
│ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐               │
│ │ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ ... │ T31 │               │
│ │ y=4 │ y=4 │ y=4 │ y=4 │ y=4 │ y=4 │     │ y=4 │               │
│ └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘               │
│                                                                 │
│ After: All threads have y = 4 (thread 3's original x value)     │
└─────────────────────────────────────────────────────────────────┘
```

### Why Warp Shuffle is Fast

| Method           | Latency     | Synchronization Needed? |
|------------------|-------------|-------------------------|
| Shared Memory    | ~20 cycles  | Yes (`__syncthreads()`) |
| Warp Shuffle     | ~1 cycle    | No (implicit in warp)   |

### Limitation

Warp shuffle only works **within a single warp** (32 threads). For communication across warps, you must use shared memory or global memory.

---

## Summary

### Memory Access Quality in Our Kernel

| Matrix | Access Pattern              | Stride | Cache Behavior                    |
|--------|-----------------------------|--------|-----------------------------------|
| **X**  | Row access (row-major)      | 1      | Excellent - sequential, reusable  |
| **W**  | Column access (row-major)   | m      | Poor - strided, no reuse          |
| **out**| Sequential write            | 1      | Excellent                         |

### The Core Problem

```
W is stored ROW-MAJOR:        But we need to access COLUMNS:

     col 0  col 1  col 2              Thread needs col 2:
    ┌──────┬──────┬──────┐            W[0,2], W[1,2], W[2,2], W[3,2]
row0│      │      │  ★   │─┐
    ├──────┼──────┼──────┤ │          In memory, these are m elements apart!
row1│      │      │  ★   │─┤
    ├──────┼──────┼──────┤ │          ★───(m)───★───(m)───★───(m)───★
row2│      │      │  ★   │─┤
    ├──────┼──────┼──────┤ │
row3│      │      │  ★   │─┘
    └──────┴──────┴──────┘
```

### Solutions

1. **Transpose W** - Store W in column-major order so column access becomes sequential
2. **Shared memory tiling** - Load tiles of W into fast shared memory, amortizing the strided global memory access

These optimizations are why libraries like cuBLAS are orders of magnitude faster than naive implementations.
