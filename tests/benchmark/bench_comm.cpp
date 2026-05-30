// f:/Projects/CANN_Com/tests/benchmark/bench_comm.cpp
// Performance benchmark for collective communication algorithms.
// Run: ./bench_comm [--nranks N]

#include "simulator/simulator.h"
#include "simulator/topology/topology_builder.h"
#include "algorithm/allreduce/allreduce_ring.h"
#include "algorithm/allreduce/allreduce_rhd.h"
#include "algorithm/allreduce/allreduce_pipeline.h"
#include "algorithm/allgather/allgather_ring.h"
#include "algorithm/allgather/allgather_butterfly.h"
#include "algorithm/reduce_scatter/reduce_scatter_ring.h"
#include "algorithm/reduce_scatter/reduce_scatter_butterfly.h"
#include "algorithm/alltoall/alltoall_direct.h"
#include "algorithm/broadcast/broadcast_ring.h"
#include "algorithm/algorithm.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace cann;

struct BenchResult {
    std::string algo_name;
    size_t data_bytes;
    uint32_t nranks;
    double time_ms;
    double bandwidth_gbps;
};

BenchResult runBench(const std::string& name, Algorithm& algo,
                     uint32_t nranks, size_t elem_count, HCCLDataType dtype) {
    Topology topo = TopologyBuilder()
        .addNode("node0", nranks, NPUType::ASCEND_910B)
        .build();

    Simulator sim(topo, SimMode::PureSim);
    PureSimChannel::clearMailbox();

    size_t elem_size = GetDataTypeSize(dtype);
    size_t input_bytes = elem_count * elem_size;

    size_t output_count = elem_count;
    if (name.find("AllGather") != std::string::npos) {
        output_count = elem_count * nranks;
    } else if (name.find("ReduceScatter") != std::string::npos) {
        output_count = elem_count / nranks;
    } else if (name.find("Broadcast") != std::string::npos) {
        output_count = elem_count;  // each rank receives a full copy
    }
    size_t output_bytes = output_count * elem_size;

    std::vector<std::vector<uint8_t>> inputs(nranks, std::vector<uint8_t>(input_bytes, 1));
    std::vector<std::vector<uint8_t>> outputs(nranks, std::vector<uint8_t>(output_bytes, 0));

    // Warm up
    {
        std::vector<std::thread> threads;
        for (uint32_t r = 0; r < nranks; r++) {
            threads.emplace_back([&, r]() {
                CommContext ctx(r, nranks, sim.getChannel(r));
                algo.Execute(inputs[r].data(), outputs[r].data(),
                             elem_count, dtype, HCCLReduceOp::SUM, ctx);
            });
        }
        for (auto& t : threads) t.join();
    }

    // Benchmark
    PureSimChannel::clearMailbox();
    sim.resetStats();

    auto start = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::thread> threads;
        for (uint32_t r = 0; r < nranks; r++) {
            threads.emplace_back([&, r]() {
                CommContext ctx(r, nranks, sim.getChannel(r));
                algo.Execute(inputs[r].data(), outputs[r].data(),
                             elem_count, dtype, HCCLReduceOp::SUM, ctx);
            });
        }
        for (auto& t : threads) t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int steps = algo.NumSteps(nranks);
    double total_data_gb = static_cast<double>(steps * input_bytes) / (1024.0 * 1024.0 * 1024.0);
    double bw = (time_ms > 0) ? (total_data_gb / (time_ms / 1000.0)) : 0.0;

    return {name, input_bytes, nranks, time_ms, bw};
}

BenchResult runBenchMultiNode(const std::string& name, Algorithm& algo,
                              uint32_t nodes, uint32_t ranks_per_node,
                              size_t elem_count, HCCLDataType dtype) {
    uint32_t nranks = nodes * ranks_per_node;

    TopologyBuilder builder;
    for (uint32_t n = 0; n < nodes; n++) {
        builder.addNode("node" + std::to_string(n), ranks_per_node,
                        NPUType::ASCEND_910B);
    }
    // Connect nodes in a ring topology
    for (uint32_t n = 0; n < nodes; n++) {
        uint32_t next = (n + 1) % nodes;
        if (n != next) {
            builder.connectNodes("node" + std::to_string(n),
                                "node" + std::to_string(next),
                                LinkType::ROCE, 25.0, 0.005);
        }
    }

    Topology topo = builder.build();
    Simulator sim(topo, SimMode::PureSim);
    PureSimChannel::clearMailbox();

    size_t elem_size = GetDataTypeSize(dtype);
    size_t input_bytes = elem_count * elem_size;

    size_t output_count = elem_count;
    if (name.find("AllGather") != std::string::npos) {
        output_count = elem_count * nranks;
    } else if (name.find("ReduceScatter") != std::string::npos) {
        output_count = elem_count / nranks;
    } else if (name.find("Broadcast") != std::string::npos) {
        output_count = elem_count;
    }
    size_t output_bytes = output_count * elem_size;

    std::vector<std::vector<uint8_t>> inputs(nranks, std::vector<uint8_t>(input_bytes, 1));
    std::vector<std::vector<uint8_t>> outputs(nranks, std::vector<uint8_t>(output_bytes, 0));

    // Warm up
    {
        std::vector<std::thread> threads;
        for (uint32_t r = 0; r < nranks; r++) {
            threads.emplace_back([&, r]() {
                CommContext ctx(r, nranks, sim.getChannel(r));
                algo.Execute(inputs[r].data(), outputs[r].data(),
                             elem_count, dtype, HCCLReduceOp::SUM, ctx);
            });
        }
        for (auto& t : threads) t.join();
    }

    // Benchmark
    PureSimChannel::clearMailbox();
    sim.resetStats();

    auto start = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::thread> threads;
        for (uint32_t r = 0; r < nranks; r++) {
            threads.emplace_back([&, r]() {
                CommContext ctx(r, nranks, sim.getChannel(r));
                algo.Execute(inputs[r].data(), outputs[r].data(),
                             elem_count, dtype, HCCLReduceOp::SUM, ctx);
            });
        }
        for (auto& t : threads) t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int steps = algo.NumSteps(nranks);
    double total_data_gb = static_cast<double>(steps * input_bytes) / (1024.0 * 1024.0 * 1024.0);
    double bw = (time_ms > 0) ? (total_data_gb / (time_ms / 1000.0)) : 0.0;

    return {name, input_bytes, nranks, time_ms, bw};
}

void printResult(const BenchResult& r) {
    std::cout << std::left << std::setw(22) << r.algo_name
              << " nranks=" << r.nranks
              << " data=" << std::setw(10) << r.data_bytes
              << " bytes  time=" << std::fixed << std::setprecision(3)
              << std::setw(10) << r.time_ms << " ms"
              << "  bw=" << std::setw(8) << std::setprecision(2)
              << r.bandwidth_gbps << " GB/s" << std::endl;
}

int main(int argc, char* argv[]) {
    uint32_t nranks = 8;
    bool multinode = false;
    uint32_t nodes = 2;
    uint32_t ranks_per_node = 4;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--nranks" && i + 1 < argc) {
            nranks = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "--multinode") {
            multinode = true;
        } else if (std::string(argv[i]) == "--nodes" && i + 1 < argc) {
            nodes = std::stoul(argv[++i]);
        } else if (std::string(argv[i]) == "--ranks-per-node" && i + 1 < argc) {
            ranks_per_node = std::stoul(argv[++i]);
        }
    }

    std::cout << "=== CANN Communication Benchmark ===" << std::endl;

    AllReduceRing allreduce;
    AllReduceRHD allreduce_rhd;
    AllReducePipeline allreduce_pipeline;
    AllGatherRing allgather;
    AllGatherButterfly allgather_butterfly;
    ReduceScatterRing reduce_scatter;
    ReduceScatterButterfly reduce_scatter_butterfly;
    AlltoAllDirect alltoall;
    BroadcastRing broadcast;

    std::vector<size_t> sizes = {
        1024,                  // 1 KB
        64 * 1024,             // 64 KB
        256 * 1024,            // 256 KB
        1024 * 1024,           // 1 MB
        16 * 1024 * 1024,      // 16 MB
        64 * 1024 * 1024,      // 64 MB
    };

    // Single-node benchmark
    std::cout << "--- Single-Node Benchmark ---" << std::endl;
    std::cout << "Ranks: " << nranks << std::endl << std::endl;

    for (size_t size : sizes) {
        size_t count = size / sizeof(float);
        std::cout << "--- Data size: " << size << " bytes ---" << std::endl;
        printResult(runBench("AllReduceRing", allreduce, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("AllReduceRHD", allreduce_rhd, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("AllReducePipeline", allreduce_pipeline, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("AllGatherRing", allgather, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("AllGatherButterfly", allgather_butterfly, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("ReduceScatterRing", reduce_scatter, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("ReduceScatterButterfly", reduce_scatter_butterfly, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("AlltoAllDirect", alltoall, nranks, count, HCCLDataType::FLOAT32));
        printResult(runBench("BroadcastRing", broadcast, nranks, count, HCCLDataType::FLOAT32));
        std::cout << std::endl;
    }

    // Multi-node benchmark
    if (multinode) {
        uint32_t total_ranks = nodes * ranks_per_node;
        std::cout << "--- Multi-Node Benchmark ---" << std::endl;
        std::cout << "Nodes: " << nodes
                  << "  Ranks/node: " << ranks_per_node
                  << "  Total ranks: " << total_ranks << std::endl << std::endl;

        for (size_t size : sizes) {
            size_t count = size / sizeof(float);
            std::cout << "--- Data size: " << size << " bytes ---" << std::endl;
            printResult(runBenchMultiNode("AllReduceRing", allreduce, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("AllReduceRHD", allreduce_rhd, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("AllReducePipeline", allreduce_pipeline, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("AllGatherRing", allgather, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("AllGatherButterfly", allgather_butterfly, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("ReduceScatterRing", reduce_scatter, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("ReduceScatterButterfly", reduce_scatter_butterfly, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("AlltoAllDirect", alltoall, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            printResult(runBenchMultiNode("BroadcastRing", broadcast, nodes, ranks_per_node, count, HCCLDataType::FLOAT32));
            std::cout << std::endl;
        }
    }

    std::cout << "=== Benchmark complete ===" << std::endl;
    return 0;
}
