// f:/Projects/CANN_Com/src/algorithm/hccl_api/hccl.h
#pragma once

// HCCL Plugin Interface — mirrors Ascend CANN HCCL API
// This is the contract that all algorithm implementations must follow.

#include "common/types.h"
#include "common/error.h"
#include <cstddef>
#include <cstdint>

namespace cann {

// Opaque communication handle
struct HCCLCommImpl;
using HCCLComm = HCCLCommImpl*;

// Configuration for communicator initialization
struct HCCLCommConfig {
    uint32_t ndev;           // Number of devices
    uint32_t rank;           // This rank's ID
    SimMode sim_mode;        // Simulation mode
    // Future: topology hints, algorithm preferences, etc.
};

// === Lifecycle ===

Status hcclCommInit(HCCLComm* comm, const HCCLCommConfig& config);
Status hcclCommDestroy(HCCLComm comm);
Status hcclCommGetRank(HCCLComm comm, uint32_t* rank);
Status hcclCommGetNranks(HCCLComm comm, uint32_t* nranks);

// === Collective Communication Primitives ===

Status hcclAllReduce(void* sendbuf, void* recvbuf, size_t count,
                     HCCLDataType dtype, HCCLReduceOp op, HCCLComm comm);

Status hcclAllGather(void* sendbuf, void* recvbuf, size_t count,
                     HCCLDataType dtype, HCCLComm comm);

Status hcclReduceScatter(void* sendbuf, void* recvbuf, size_t count,
                         HCCLDataType dtype, HCCLReduceOp op, HCCLComm comm);

Status hcclAlltoAll(void* sendbuf, void* recvbuf, size_t count,
                    HCCLDataType dtype, HCCLComm comm);

Status hcclBroadcast(void* buf, size_t count, HCCLDataType dtype,
                     uint32_t root, HCCLComm comm);

// === Algorithm Control ===

Status hcclSetAlgorithm(HCCLComm comm, const char* prim_name,
                        const char* algo_name);

struct HCCLPerfStats {
    double total_time_ms;
    double algo_time_ms;     // Time in algorithm logic
    double comm_time_ms;     // Time in communication
    size_t bytes_transferred;
    double bandwidth_gbps;   // Achieved bandwidth
};

Status hcclGetPerfStats(HCCLComm comm, HCCLPerfStats* stats);

} // namespace cann
