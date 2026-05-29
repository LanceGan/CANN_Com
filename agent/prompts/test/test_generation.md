# Test Generation Prompt Template

## Test Structure
Generate a Google Test file for the algorithm `{{ClassName}}`.

### Required Test Cases
1. Basic correctness: Small nranks (2-4), single element
2. Scaling: 8 ranks, larger data (1024+ elements)
3. Metadata: Name() and NumSteps() verification

### Test Template
```cpp
#include <gtest/gtest.h>
#include "{{header_path}}"
#include "algorithm/algorithm.h"
#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include <vector>
#include <thread>

using namespace cann;

class {{TestClassName}} : public ::testing::Test {
protected:
    void SetUp() override { PureSimChannel::clearMailbox(); }
    void TearDown() override { PureSimChannel::clearMailbox(); }
};
```

### Multithreading
Each rank runs in its own `std::thread`. Use `PureSimChannel::clearMailbox()` in SetUp/TearDown.
