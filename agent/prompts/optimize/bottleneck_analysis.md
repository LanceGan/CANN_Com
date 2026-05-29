# Bottleneck Analysis Prompt Template

## Task
Analyze the performance of `{{algorithm_name}}` and identify bottlenecks.

## Input Data
- Algorithm code: {{algorithm_code}}
- Benchmark results: {{benchmark_results}}
- Topology: {{topology_info}}

## Analysis Framework

### 1. Communication Bottleneck
- Is the algorithm bandwidth-bound or latency-bound?
- What is the achieved bandwidth vs theoretical peak?

### 2. Step Analysis
- How many communication steps?
- Can steps be overlapped (pipelining)?

### 3. Data Movement
- Total data moved per rank?
- Redundant data movement?

### 4. Hardware Utilization
- HCCS vs ROCE utilization?
- Buffer reuse efficiency?

## Output Format
```
## Bottleneck Report: {{algorithm_name}}

### Primary Bottleneck
[description]

### Optimization Opportunities
1. [optimization 1]: [expected improvement]

### Recommended Next Steps
[actionable recommendations]
```
