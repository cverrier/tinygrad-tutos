# Understanding tinygrad's Pattern Matcher

> [!NOTE]
> This tutorial has been built around tinygrad's commit [f509019](https://github.com/tinygrad/tinygrad/tree/f5090192c84760be1227f7e3c4f99ad0603117ae) with a Python 3.11 virtual environment. Since tinygrad is evolving rapidly, if you're following along in code, make sure to checkout that commit in order to avoid any discrepancies.

This tutorial teaches tinygrad's [`PatternMatcher`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/uop/ops.py#L1034-L1059) system from first principles. `PatternMatcher` is the core mechanism for graph transformations throughout tinygrad - from symbolic simplification to gradient computation to code generation.


## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Basic Pattern Matching](#2-basic-pattern-matching)
3. [Capturing Values](#3-capturing-values-with-name)
4. [Matching Tree Structure](#4-matching-tree-structure-with-src)
5. [Matching Rules](#5-upat-matching-rules)
6. [Convenience Methods](#6-convenience-methods)
7. [Transforming Entire Graphs](#7-graph_rewrite-transforming-entire-graphs)
8. [Top-Down vs Bottom-Up Traversal](#8-top-down-vs-bottom-up-traversal)
9. [Context Parameter for Shared State](#9-context-parameter-for-shared-state)
10. [List vs Tuple: Handling Commutativity](#10-list-vs-tuple-handling-commutativity)
11. [Combining Pattern Matchers](#11-combining-patternmatchers)
12. [Real Patterns from the Codebase](#12-real-patterns-from-the-codebase)
13. [Case Study: Gradient Computation](#13-case-study-gradient-computation)
14. [Summary](#14-summary)

---

## 1. The Core Idea

Think of `PatternMatcher` like [regex](https://grokipedia.com/page/Regular_expression) for `UOp` graphs. Just as regex matches text patterns and can capture groups, `PatternMatcher` matches `UOp` tree patterns and captures named nodes.

The basic structure:

```python
pm = PatternMatcher([
  (pattern, callback),
  (pattern, callback),
  ...
])
```

When you call `pm.rewrite(uop)`, it:
1. Finds patterns that match the `UOp`
2. Calls the callback with captured values
3. Returns whatever the callback returns (or `None` if no match)

### How tinygrad Works (High-Level)

If you're new to tinygrad, here's a quick overview:

1. **Tensor API** - You write PyTorch-like code using `Tensor` operations (`tensor.py`)
2. **Tensor `UOp`s** - Operations build a graph of `UOp`s (Universal Operations) representing the computation
3. **Schedule** - The tensor UOp graph is converted into kernel `UOp`s (`engine/schedule.py`)
4. **Codegen** - Kernel `UOp`s are transformed and lowered to device-specific code (`codegen/`)
5. **Runtime** - Generated code is compiled and executed on the target device (`runtime/`)

`PatternMatcher` is used at nearly every stage: simplifying symbolic expressions, computing gradients, optimizing kernels, and generating code. Understanding `PatternMatcher` unlocks understanding of tinygrad's internals.

For more details on tinygrad's architecture, see the main documentation. This tutorial focuses specifically on `PatternMatcher`.

---

## 2. Basic Pattern Matching

The simplest pattern matches a specific operation type and argument:

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  # Create a simple pattern:
  #   1. Match any CONST with value 0
  #   2. Replace it with CONST 42
  (UPat(Ops.CONST, arg=0), lambda: UOp.const(dtypes.int, 42)),
])

zero = UOp.const(dtypes.int, 0)
print(pm.rewrite(zero))  # UOp(Ops.CONST, dtypes.int, arg=42, src=())
one = UOp.const(dtypes.int, 1)
print(pm.rewrite(one))   # None (no match)
zero_float = UOp.const(dtypes.float, 0.0)
# Also matches because the pattern did not specify dtype
print(pm.rewrite(zero_float))  # UOp(Ops.CONST, dtypes.int, arg=42, src=())
```

The pattern matched `CONST` with `arg=0`, and the callback returned a new UOp. When no pattern matches, `rewrite()` returns `None`.

---

## 3. Capturing Values

The power of `PatternMatcher` comes from capturing parts of the matched `UOp`. The `name` parameter captures that `UOp` and passes it to your callback:

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  (UPat(Ops.CONST, name="x"), lambda x: print(f"Matched: {x}")),
])

a = UOp.const(dtypes.int, 42)
pm.rewrite(a)  # Matched: UOp(Ops.CONST, dtypes.int, arg=42, src=())
```

The `name="x"` captures the `UOp`, and `lambda x:` receives it. Here's a more useful example - doubling any constant:

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  (UPat(Ops.CONST, dtype=dtypes.int, name="x"), lambda x: UOp.const(dtypes.int, x.arg * 2)),
])

a = UOp.const(dtypes.int, 21)
result = pm.rewrite(a)
print(f"Input: {a}")   # UOp(Ops.CONST, dtypes.int, arg=21, src=())
print(f"Output: {result}")  # UOp(Ops.CONST, dtypes.int, arg=42, src=())
```

> [!WARNING]
> You might wonder: in the callback, can I simply return `x * 2` instead of creating a new `UOp.const`? Yes, but be careful: `x * 2` creates a new `UOp` representing multiplication. This is fine here, but in more complex patterns, beware of unintended graph structures:
> ```python
> from tinygrad import UOp, dtypes
> from tinygrad.uop import Ops
> from tinygrad.uop.ops import PatternMatcher, UPat
> 
> pm = PatternMatcher([
>   (UPat(Ops.CONST, dtype=dtypes.int, name="x"), lambda x: x*2),
> ])
> 
> a = UOp.const(dtypes.int, 21)
> result = pm.rewrite(a)
> print(f"Input: {a}")   # UOp(Ops.CONST, dtypes.int, arg=21, src=())
> print(f"Output:\n{result}")
> # Output:
> # UOp(Ops.MUL, dtypes.int, arg=None, src=(
> #   UOp(Ops.CONST, dtypes.int, arg=21, src=()),
> #   UOp(Ops.CONST, dtypes.int, arg=2, src=()),))
> ```

---

## 4. Matching Tree Structure

Real patterns match tree structure. The `src` parameter matches the children of a `UOp`:

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  # Pattern: ADD(x, x) -> x * 2
  # This matches: any ADD where both sources are the SAME UOp
  (UPat(Ops.ADD, src=(UPat(name="x"), UPat(name="x"))), lambda x: x * UOp.const(x.dtype, 2)),
])

a = UOp.const(dtypes.int, 5)
b = UOp.const(dtypes.int, 7)

add_same = a + a  # ADD where both children are the same object
add_diff = a + b  # ADD where children are different objects

print(f"a + a matches: {pm.rewrite(add_same) is not None}")  # True
print(f"a + b matches: {pm.rewrite(add_diff) is not None}")  # False
```

When you use `name="x"` twice in a pattern, both positions must match *the same object* (Python `is` comparison). This is how tinygrad detects `a + a` vs `a + b`.

`UOp`s are cached. Creating `UOp.const(dtypes.int, 5)` twice returns the same object:

```python
from tinygrad import UOp, dtypes

a = UOp.const(dtypes.int, 5)
b = UOp.const(dtypes.int, 5)
print(f"a is b: {a is b}")  # True
```

---

## 5. Matching Rules

Here's what `UPat` checks when matching:

| `UPat` field | What it matches |
|------------|-----------------|
| `op` | `UOp`'s operation type (single `Ops` or tuple of `Ops`) |
| `dtype` | `UOp`'s data type (single dtype or tuple) |
| `arg` | `UOp`'s argument (exact match) |
| `name` | Captures the `UOp` (same name = same object) |
| `src` | Tuple/list of child `UPat`s to match children |

### Matching Multiple Operations

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  # Match both ADD and MUL
  (UPat((Ops.ADD, Ops.MUL), name="x"), lambda x: print(f"Matched {x.op}")),
])

a = UOp.const(dtypes.int, 3)
b = UOp.const(dtypes.int, 4)

pm.rewrite(a + b)  # Output: Matched Ops.ADD
pm.rewrite(a * b)  # Output: Matched Ops.MUL
pm.rewrite(a % b)  # No match
```

### Matching Data Types

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  (UPat(Ops.CONST, dtype=dtypes.float32, name="x"), lambda x: print(f"Float: {x.arg}")),
])

pm.rewrite(UOp.const(dtypes.float32, 3.14))  # Output: Float: 3.14
pm.rewrite(UOp.const(dtypes.int, 42))  # No match
```

---

## 6. Convenience Methods

`UPat` provides shortcuts for common patterns:

| Method | What it matches |
|--------|-----------------|
| `UPat.var(name)` | Any `UOp`, captured with given name |
| `UPat.cvar(name)` | Constants (`CONST` or `VCONST`) |
| `UPat()` | Any `UOp` (no capture) |

Here's an example to match addition with zero:

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  # Match ADD where second arg is constant 0
  (UPat(Ops.ADD, src=(p:=UPat.var("x"), UPat.cvar("c", p.dtype, arg=0))), lambda x, c: x),
  # (UPat.var("x") + 0, lambda x: x),
])

a = UOp.const(dtypes.int, 42)
zero = UOp.const(dtypes.int, 0)
one = UOp.const(dtypes.int, 1)

print(f"42 + 0 rewrites to: {pm.rewrite(a + zero)}")  # UOp(Ops.CONST, dtypes.int, arg=42, src=())
print(f"42 + 1 rewrites to: {pm.rewrite(a + one)}")  # None
```

Actually, since `UPat` supports operator overloading, you can write the above more succinctly, [as done in `tinygrad/uop/symbolic.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/uop/symbolic.py#L42):

```python
from tinygrad import UOp, dtypes
from tinygrad.uop.ops import PatternMatcher, UPat

pm = PatternMatcher([
  # Match ADD where second arg is constant 0
  (UPat.var("x") + 0, lambda x: x),
])

a = UOp.const(dtypes.int, 42)

print(f"42 + 0 rewrites to: {pm.rewrite(a + 0)}")  # UOp(Ops.CONST, dtypes.int, arg=42, src=())
print(f"42 + 1 rewrites to: {pm.rewrite(a + 1)}")  # None
```

---

## 7. Transforming Entire Graphs

`pm.rewrite()` only checks one `UOp`. To transform an entire graph recursively, use `graph_rewrite()`:

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite

pm = PatternMatcher([
  # Constant folding: ADD(CONST, CONST) -> CONST
  (
    UPat(Ops.ADD, src=(UPat.cvar("a"), UPat.cvar("b"))),
    lambda a, b: UOp.const(a.dtype, a.arg + b.arg)
  ),
])

# Build a graph: (1 + 2) + (3 + 4)
one = UOp.const(dtypes.int, 1)
two = UOp.const(dtypes.int, 2)
three = UOp.const(dtypes.int, 3)
four = UOp.const(dtypes.int, 4)
expr = (one + two) + (three + four)

print(f"Before:\n{expr}")
# UOp(Ops.ADD, dtypes.int, arg=None, src=(
#   UOp(Ops.ADD, dtypes.int, arg=None, src=(
#     UOp(Ops.CONST, dtypes.int, arg=1, src=()),
#     UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),
#   UOp(Ops.ADD, dtypes.int, arg=None, src=(
#     UOp(Ops.CONST, dtypes.int, arg=3, src=()),
#     UOp(Ops.CONST, dtypes.int, arg=4, src=()),)),))

print(f"After:  {graph_rewrite(expr, pm)}")
# UOp(Ops.CONST, dtypes.int, arg=10, src=())
```

`graph_rewrite()` folded `(1+2)+(3+4)` -> `3+7` -> `10` automatically.

**How it works** (simplified):
1. Walk the graph in topological order (leaves first by default)
2. For each `UOp`, try all patterns via `pm.rewrite()`
3. If a pattern matches and returns a new `UOp`, replace it
4. Continue until no more patterns match

---

## 8. Top-Down vs Bottom-Up Traversal

`graph_rewrite()` has two modes controlled by the `bottom_up` parameter:

- **Top-down (default, `bottom_up=False`)**: Rewrites children first, then parent, *i.e.*, “leaves-to-root” path. Patterns see the already-rewritten children, *i.e.*, first recurse into children, then apply patterns to parent.
- **Bottom-up (`bottom_up=True`)**: Rewrites parent first, then children, i.e., “root-to-leaves” path --> patterns see original children.

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite

pm = PatternMatcher([
  (UPat(Ops.ADD, name="x"), lambda x: print(f"Visiting ADD: {[s.arg for s in x.src]}")),
  (UPat(Ops.CONST, name="x"), lambda x: print(f"Visiting CONST: {x.arg}")),
])

# Graph: (1 + 2) + 3
one = UOp.const(dtypes.int, 1)
two = UOp.const(dtypes.int, 2)
three = UOp.const(dtypes.int, 3)
expr = (one + two) + three

print("Top-down order:")
graph_rewrite(expr, pm)
# Visiting CONST: 1
# Visiting CONST: 2
# Visiting ADD: [1, 2]
# Visiting CONST: 3
# Visiting ADD: [None, 3]
print()
print("Bottom-up order:")
graph_rewrite(expr, pm, bottom_up=True)
# Visiting ADD: [None, 3]
# Visiting ADD: [1, 2]
# Visiting CONST: 1
# Visiting CONST: 2
# Visiting CONST: 3
```

---

## 9. Context Parameter for Shared State

Patterns can receive a context dictionary for sharing state across pattern matches. This is essential for complex transformations that need to accumulate information.

### Basic Usage: Counting Nodes

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat, graph_rewrite

def count_consts(ctx, x):
  """Count constants seen so far."""
  ctx["count"] = ctx.get("count", 0) + 1
  print(f"Saw constant {x.arg}, total: {ctx['count']}")

pm = PatternMatcher([
  (UPat(Ops.CONST, name="x"), count_consts),
])

one = UOp.const(dtypes.int, 1)
two = UOp.const(dtypes.int, 2)
three = UOp.const(dtypes.int, 3)
expr = (one + two) + three

ctx = {}
graph_rewrite(expr, pm, ctx=ctx)
print(f"\nFinal count: {ctx['count']}")
# Saw constant 1, total: 1
# Saw constant 2, total: 2
# Saw constant 3, total: 3
#
# Final count: 3
```

When your callback takes `ctx` as its **first parameter**, it receives the context dictionary. This is detected by parameter name.

### Advanced Usage: Context in Autograd

The context parameter is heavily used in autograd. In [`tinygrad/gradient.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/gradient.py#L62), the incoming gradient is passed as context:

```python
lgrads = pm_gradient.rewrite(t0, ctx=grads[t0])
```

[Each gradient rule receives `ctx`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/gradient.py#L16-L49) (the gradient flowing backward) and uses it to compute gradients for inputs. See [Case Study: Gradient Computation](#13-case-study-gradient-computation) for details.

---

## 10. List vs Tuple: Handling Commutativity

This is a subtle but important distinction:

- **`src=(A, B)` (tuple)**: Positional matching - first child must match A, second must match B
- **`src=[A, B]` (list)**: Tries all permutations - matches (A,B) or (B,A)

```python
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat

# tuple src: order matters
pm_tuple = PatternMatcher([
  (
    UPat(Ops.ADD, src=(UPat.cvar("a"), UPat.var("x"))),
    lambda a, x: print(f"tuple: a={a.arg}, x={x}")
  ),
])

# list src: tries all permutations
pm_list = PatternMatcher([
  (
    UPat(Ops.ADD, src=[UPat.cvar("a"), UPat.var("x")]),
    lambda a, x: print(f"list: a={a.arg}, x={x}")
  ),
])

one = UOp.const(dtypes.int, 1)
two = UOp.const(dtypes.int, 2)
add1 = one + two

print("Tuple pattern (first must be cvar):")
pm_tuple.rewrite(add1)
# Tuple: a=1, x=UOp(Ops.CONST, dtypes.int, arg=2, src=())

print("List pattern (either can be cvar):")
pm_list.rewrite(add1)
# List: a=1, x=UOp(Ops.CONST, dtypes.int, arg=2, src=())
# List: a=2, x=UOp(Ops.CONST, dtypes.int, arg=1, src=())
```

The list pattern matched twice because it tries both orderings. Therefore, you should use lists for commutative operations like `ADD` and `MUL`.

---

## 11. Combining Pattern Matchers

`PatternMatcher`s can be combined using the `+` operator:

```python
from tinygrad import UOp, dtypes
from tinygrad.uop.ops import PatternMatcher, UPat

pm1 = PatternMatcher([
  (UPat.var("x") + 0, lambda x: x),
])

pm2 = PatternMatcher([
  (UPat.var("x") * 1, lambda x: x),
])

# Combine them
pm_combined = pm1 + pm2

a = UOp.const(dtypes.int, 42)
zero = UOp.const(dtypes.int, 0)
one = UOp.const(dtypes.int, 1)

print(f"42+0: {pm_combined.rewrite(a + zero)}")  # UOp(Ops.CONST, dtypes.int, arg=42, src=())
print(f"42*1: {pm_combined.rewrite(a * one)}")   # UOp(Ops.CONST, dtypes.int, arg=42, src=())
```

This is how tinygrad builds complex pattern matchers from smaller, composable pieces.

---

## 12. Real Patterns from the Codebase

For more details, look at the actual patterns in [`tinygrad/uop/symbolic.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/uop/symbolic.py#L40-L118).

---

## 13. Case Study: Gradient Computation

The gradient system in [`tinygrad/gradient.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/gradient.py) is an elegant application of `PatternMatcher`. Each rule defines how gradients flow backward through an operation.

### The Pattern Matcher

```python
# ctx is grad_output (the incoming gradient)
pm_gradient = PatternMatcher([
    # RECIPROCAL: d(1/x) = -1/x^2 * ctx
    (UPat(Ops.RECIPROCAL, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),
    # SIN: d(sin(x)) = cos(x) * ctx
    (UPat(Ops.SIN, name="ret"), lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),
    # ADD: chain rule d(a+b) = (1*ctx, 1*ctx)
    (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),
    # MUL: chain rule d(a*b) = (b*ctx, a*ctx)
    (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),
    # RESHAPE: reshape gradient back to input shape
    (UPat(Ops.RESHAPE, name="ret"), lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),
    ...
])
```

A few observations:
1. `ctx` is the incoming gradient (grad_output) - passed via `pm_gradient.rewrite(t0, ctx=grads[t0])` (see below)
2. `ret` is the forward operation being differentiated - captured by `name="ret"`
3. Callbacks return tuples - one gradient per input to the operation
4. `None` means no gradient - used for non-differentiable inputs

### How It's Used

Below is a simplified version of the gradient computation function, that illustrates backpropagation. For the full version, see [`compute_gradient()` in `tinygrad/gradient.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/gradient.py#L58-L78).

```python
def compute_gradient(root:UOp, root_grad:UOp, targets:set[UOp]) -> dict[UOp, UOp]:
    grads = {root: root_grad}
    for t0 in reversed(_deepwalk(root, targets)):
        if t0 not in grads: continue
        # Apply gradient rule with current gradient as context
        lgrads = pm_gradient.rewrite(t0, ctx=grads[t0])
        if lgrads is None:
            raise RuntimeError(f"failed to compute gradient for {t0.op}")
        # Accumulate gradients for each input
        for k, v in zip(t0.src, lgrads):
            if v is None: continue
            if k in grads: grads[k] = grads[k] + v
            else: grads[k] = v
    return grads
```

The elegance here is that **adding a new differentiable operation just requires adding one pattern rule**.

If you don't know how exactly backpropagation works, I highly recommend watching [Andrej Karpathy's video about this topic](https://www.youtube.com/watch?v=VMj-3S1tku0).

---

## 14. Summary

| Concept | What it does |
|---------|--------------|
| `UPat(op, dtype, src, arg, name)` | Describes a pattern to match |
| `name="x"` | Captures matched `UOp`, passed to callback |
| Same name twice | Both positions must be the same object |
| `src=(A, B)` (tuple) | Positional match (order matters) |
| `src=[A, B]` (list) | Tries all permutations (for commutative ops) |
| `UPat.var("x")` | Match any `UOp` |
| `UPat.cvar("c")` | Match constants (`CONST`/`VCONST`) |
| `pm.rewrite(uop)` | Try patterns on one `UOp` |
| `graph_rewrite(uop, pm)` | Apply patterns to entire graph recursively |
| `graph_rewrite(..., ctx={})` | Pass context dict to callbacks |
| `graph_rewrite(..., bottom_up=True)` | Process parents before children |
| `pm1 + pm2` | Combine pattern matchers |
| Callback with `ctx` first param | Receives the context dict |

### Quick Reference: Writing Patterns

```python
# Match specific op
UPat(Ops.ADD)

# Match multiple ops
UPat((Ops.ADD, Ops.MUL))

# Match and capture
UPat(Ops.CONST, name="x")

# Match with children (positional)
UPat(Ops.ADD, src=(UPat.var("a"), UPat.var("b")))

# Match with children (commutative)
UPat(Ops.ADD, src=[UPat.cvar("c"), UPat.var("x")])

# Match specific dtype
UPat(Ops.CONST, dtype=dtypes.float32, name="x")

# Match specific arg value
UPat(Ops.CONST, arg=0)

# Callback receives captured values
lambda x, y: x + y

# Callback with context
lambda ctx, x: ctx["key"] + x
```

### Some Places where Pattern Matcher is Used in tinygrad

- [`tinygrad/uop/symbolic.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/uop/symbolic.py) - Algebraic simplification
- [`tinygrad/gradient.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/gradient.py) - Automatic differentiation
- [`tinygrad/codegen/`](https://github.com/tinygrad/tinygrad/tree/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/codegen)
- [`tinygrad/engine/schedule.py`](https://github.com/tinygrad/tinygrad/blob/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/engine/schedule.py)
- [`tinygrad/renderer/`](https://github.com/tinygrad/tinygrad/tree/f5090192c84760be1227f7e3c4f99ad0603117ae/tinygrad/renderer) - Code generation for different backends

Understanding `PatternMatcher` is key to understanding how tinygrad transforms tensor operations into optimized code.
