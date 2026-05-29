# HCCL Plugin Code Template

## Header File (.h)
```cpp
#pragma once
#include "algorithm/algorithm.h"

namespace cann {

class {{ClassName}} : public Algorithm {
public:
    Status Execute(void* sendbuf, void* recvbuf, size_t count,
                   HCCLDataType dtype, HCCLReduceOp op,
                   CommContext& ctx) override;

    const char* Name() const override { return "{{ClassName}}"; }

    int NumSteps(uint32_t nranks) const override {
        return {{num_steps_expression}};
    }
};

} // namespace cann
```

## Implementation File (.cpp)
```cpp
#include "{{header_include}}"
#include <cstring>
#include <vector>

namespace cann {

Status {{ClassName}}::Execute(void* sendbuf, void* recvbuf, size_t count,
                               HCCLDataType dtype, HCCLReduceOp op,
                               CommContext& ctx) {
    uint32_t rank = ctx.rank();
    uint32_t nranks = ctx.nranks();
    size_t elem_size = GetDataTypeSize(dtype);

    if (nranks <= 1) {
        if (sendbuf != recvbuf) std::memcpy(recvbuf, sendbuf, count * elem_size);
        return Status::SUCCESS;
    }
    if (count == 0) return Status::SUCCESS;

    {{algorithm_body}}

    return Status::SUCCESS;
}

} // namespace cann
```
