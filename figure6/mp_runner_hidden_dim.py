import torch.multiprocessing as mp
import npc.utils as utils
import atexit
import importlib
import os
import sys


def get_pre_defined_args(args):
    fanout_info = str(args.fan_out).replace(" ", "")
    config_key = args.configs_path.split("/")[-2]

    # [NOTE] run the exp of varying cache memory
    cache_memory_in_gbs = [4]
    # [NOTE] run the varying hidden_dim exp
    hidden_dims = [8, 32, 128, 512]
    model, nl = "SAGE", -1

    # generate args
    for cache_mem in cache_memory_in_gbs:
        for num_hidden in hidden_dims:
            for sys in ["DP", "NP", "SP", "MP"]:
                # cross-machine feat loading case
                tag = f"{sys}_{model}_nl{nl}of8_cm{cache_mem}GB"
                # model specific
                num_heads, num_hidden = (-1, num_hidden)
                key = "npc" if sys == "NP" else "ori"
                # dryrun path
                dryrun_file_path = f"{args.caching_candidate_path_prefix}/{key}_{config_key}_{fanout_info}"
                yield {
                    "system": sys,
                    "model": "SAGE",
                    "cache_memory": cache_mem * 1024 * 1024 * 1024,
                    "num_localnode_feats_in_workers": nl,
                    "tag": tag,
                    "num_heads": num_heads,
                    "num_hidden": num_hidden,
                    "dryrun_file_path": dryrun_file_path,
                }


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.getenv("APT_HOME"), "examples"))

    args, shared_tensor_list, global_nfeat = utils.pre_spawn()

    world_size = args.world_size
    nproc = world_size if args.nproc_per_node == -1 else args.nproc_per_node
    ranks = args.ranks
    local_ranks = args.local_ranks
    print(
        f"[Note]procs:{nproc}\t world_size:{world_size}\t ranks:{ranks}\t local_ranks:{local_ranks}"
    )
    if args.nproc_per_node == -1:
        shared_tensors_with_nfeat = utils.determine_feature_reside_cpu(
            args, global_nfeat, shared_tensor_list
        )

    train_module = importlib.import_module("train")
    for inputs in get_pre_defined_args(args):
        for key, value in inputs.items():
            setattr(args, key, value)
        utils.show_args(args)

        if args.nproc_per_node != -1:
            shared_tensors_with_nfeat = utils.determine_feature_reside_cpu(
                args, global_nfeat, shared_tensor_list
            )

        # reimport train.py
        run = importlib.reload(train_module).run
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

        if args.nproc_per_node != -1:
            del shared_tensors_with_nfeat

    sys.path.pop(0)
