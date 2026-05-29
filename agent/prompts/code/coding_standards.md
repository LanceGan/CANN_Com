# C++ Coding Standards for CANN Algorithms

## Namespace
All code must be in the `cann` namespace.

## Headers
- Use `#pragma once` for include guards
- Include `"algorithm/algorithm.h"` for base class
- Include `<cstring>` for memcpy, `<vector>` for std::vector

## Algorithm Class
- Inherit from `Algorithm` base class
- Implement `Execute()`, `Name()`, `NumSteps()`
- Name format: `{Primitive}{Variant}` (e.g., `AllReduceRing`)

## Communication Pattern
- Use `ctx.send(data, bytes, dst_rank)` and `ctx.recv(buf, bytes, src_rank)`
- Use `ReduceBuffer(dst, src, count, dtype, op)` for reduction
- Use `GetDataTypeSize(dtype)` for element size

## Edge Cases
Always handle:
- `nranks <= 1`: direct memcpy
- `count == 0`: no-op

## Memory
- Use `std::vector<uint8_t>` for temporary buffers
- Never use raw `new`/`delete`
