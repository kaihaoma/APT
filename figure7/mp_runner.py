import torch.multiprocessing as mp
import npc
import npc.utils as utils
import atexit
import importlib
import os
import sys


def get_nl(args):
    if "papers" in args.configs_path:
        nl = [13.25, 14.25, 15.25, 16.25, 17.25]
    elif "friendster" in args.configs_path:
        nl = [15.65, 17.65, 19.65, 21.65, 23.65, 25.65, 27.65, 29.65, 31.65, 33.65]
    elif "igbfull" in args.configs_path:
        nl = [32.11, 33.11, 34.11, 35.11, 36.11]
    else:
        print(f"[Note]args.configs_path:{args.configs_path}")
        nl = [0.1 * i for i in range(1, 10)]
    return nl


def get_pre_defined_args(args):
    cache_memory_in_gbs = [4]
    model = "SAGE"

    num_localnode_feats_in_workers = get_nl(args)[:1]
    # generate args
    for cache_mem in cache_memory_in_gbs:
        for nl in num_localnode_feats_in_workers:
            # model specific
            num_heads, num_hidden = (-1, 16)
            yield {
                "model": model,
                "cache_memory": cache_mem * 1024 * 1024 * 1024,
                "num_localnode_feats_in_workers": nl,
                "num_heads": num_heads,
                "num_hidden": num_hidden,
            }


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getenv("APT_HOME"), "examples"))

    args, shared_tensor_list, global_nfeat = utils.pre_spawn()

    assert args.nproc_per_node != -1
    world_size = args.world_size
    nproc = args.nproc_per_node
    ranks = args.ranks
    local_ranks = args.local_ranks
    fanout_info = str(args.fan_out).replace(" ", "")
    config_key = args.configs_path.split("/")[-2]
    print(
        f"[Note]procs:{nproc}\t world_size:{world_size}\t ranks:{ranks}\t local_ranks:{local_ranks}"
    )

    # determine the best parallelism strategy for each config
    for inputs in get_pre_defined_args(args):
        for key, value in inputs.items():
            setattr(args, key, value)
        args.tag = f"{args.model}_nl{args.num_localnode_feats_in_workers}of8_cm{round(args.cache_memory / (1024*1024*1024))}GB"
        utils.show_args(args)

        npc.determine_best_strategy(
            nproc, ranks, local_ranks, world_size, args, shared_tensor_list, 1, 5
        )

    train_module = importlib.import_module("train")
    for inputs in get_pre_defined_args(args):
        for key, value in inputs.items():
            setattr(args, key, value)

        for system in ["DP", "NP", "SP", "MP"]:
            args.system = system
            args.tag = f"{system}_{args.model}_nl{args.num_localnode_feats_in_workers}of8_cm{round(args.cache_memory / (1024*1024*1024))}GB"
            key = "npc" if system == "NP" else "ori"
            args.dryrun_file_path = (
                f"{args.caching_candidate_path_prefix}/{key}_{config_key}_{fanout_info}"
            )
            utils.show_args(args)

            shared_tensors_with_nfeat = utils.determine_feature_reside_cpu(
                args, global_nfeat, shared_tensor_list
            )

            # start training process
            run = importlib.reload(train_module).train_with_strategy
            processes = []
            for i in range(nproc):
                p = mp.Process(
                    target=run,
                    args=(
                        ranks[i],
                        local_ranks[i],
                        world_size,
                        args,
                        shared_tensors_with_nfeat,
                    ),
                )
                atexit.register(utils.kill_proc, p)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            del shared_tensors_with_nfeat
