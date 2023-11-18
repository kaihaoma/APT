#ifndef NPC_OPS_COLLECTIVE_H_
#define NPC_OPS_COLLECTIVE_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include <vector>

#include "../utils.h"

namespace npc {

IdType SPSampleAlltoAllWithDst(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand = 1, IdType comm_type = 0);

void SPSampleAlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand = 1, IdType comm_type = 0);

void SPFeatureAlltoAllWithDst(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand = 1, IdType comm_type = 0);

void SPFeatureAlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand = 1, IdType comm_type = 0);

void AlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_offset,
    torch::Tensor recv_offset, IdType expand = 1, IdType comm_type = 0);

void AllBroadcast(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_size,
    torch::Tensor recv_size, IdType expand = 1, IdType comm_type = 0);

void AllBroadcastV2(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_size,
    torch::Tensor recv_size, IdType expand = 1, IdType comm_type = 0);

void AllReduce(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_size,
    IdType expand = 1, IdType comm_type = 0);

torch::Tensor AllGather(torch::Tensor input, IdType comm_type = 0);

void CrossMachineAlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_size, IdType expand = 1, IdType comm_type = 0);

}  // namespace npc

#endif