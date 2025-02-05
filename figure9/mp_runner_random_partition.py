import torch.multiprocessing as mp
import npc.utils as utils
import atexit
import importlib
import sys
import os


def get_pre_defined_args(args):
    # [NOTE] run the exp of varying cache memory
    cache_memory_in_gbs = [4]
    # [NOTE] run the varying hidden_dim exp
    hidden_dims = [8, 32, 128, 512]
    model, nl = "SAGE", -1

    # generate args
    for cache_mem in cache_memory_in_gbs:
        for num_hidden in hidden_dims:
            # model specific
            num_heads, num_hidden = (-1, num_hidden)
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

    assert args.nproc_per_node == -1
    world_size = args.world_size
    nproc = world_size if args.nproc_per_node == -1 else args.nproc_per_node
    ranks = args.ranks
    local_ranks = args.local_ranks
    fanout_info = str(args.fan_out).replace(" ", "")
    config_key = args.configs_path.split("/")[-2]
    args.cross_machine_feat_load = False
    print(
        f"[Note]procs:{nproc}\t world_size:{world_size}\t ranks:{ranks}\t local_ranks:{local_ranks}"
    )

    # load CPU resident features
    shared_tensors_with_nfeat = utils.determine_feature_reside_cpu(
        args, global_nfeat, shared_tensor_list
    )

    # start training with al parallelism strategies under each config
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

    sys.path.pop(0)
