# Algorithm Design Prompt Template

You are an expert in distributed communication algorithms for Ascend NPU clusters.

## Task
Design a {{primitive}} algorithm for the following topology:
- Topology: {{topology_name}}
- Number of ranks: {{nranks}}
- Data size: {{data_size}} bytes
- Performance goal: {{performance_goal}}

## Available HCCL API
```
{{hccl_api}}
```

## Existing Algorithms (for reference)
{{existing_algorithms}}

## Design Requirements
1. Describe the algorithm step by step
2. Analyze communication complexity (number of steps)
3. Analyze data volume per step
4. Identify the optimal chunk size strategy
5. Consider hardware-specific optimizations (HCCS vs ROCE)

## Output Format
Provide:
1. **Algorithm Name**: [name]
2. **Algorithm Type**: [Ring/Tree/Butterfly/NHR/etc.]
3. **Step-by-step Description**: [numbered steps]
4. **Complexity Analysis**: [steps, data volume]
5. **Pseudocode**: [structured pseudocode]
6. **Expected Performance**: [theoretical bandwidth, latency]
