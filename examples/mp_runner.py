import torch
import torch.multiprocessing as mp
import utils
import atexit
import importlib


def get_pre_defined_args(tag_prefix):
    num_try_times = 1
    # cache_memory_in_gbs = [1]
    cache_memory_in_gbs = list(range(6))
    # system = ["DP", "NP", "SP", "MP"]
    system = ["DP", "NP"]
    num_localnode_feats_in_workers = [4, 5, 6, 7]
    # num_localnode_feats = [0]
    # generate args
    for try_times in range(num_try_times):
        for sys in system:
            for cache_mem in cache_memory_in_gbs:
                for nl in num_localnode_feats_in_workers:
                    cm = cache_mem * 1024 * 1024 * 1024
                    # cross-machine feat loading case
                    tag = f"t{try_times}_{sys}_nl{nl}of8_cm{cache_mem}GB"
                    # single-machine case
                    # tag = f"{tag_prefix}_{sys}_cm{cache_mem}GB"
                    yield {"system": sys, "cache_memory": cm, "num_localnode_feats_in_workers": nl, "tag": tag}


if __name__ == "__main__":
    args, shared_tensor_list, global_nfeat = utils.pre_spawn()
    world_size = args.world_size
    nproc = world_size if args.nproc_per_node == -1 else args.nproc_per_node
    ranks = args.ranks
    local_ranks = args.local_ranks
    print(f"[Note]procs:{nproc}\t world_size:{world_size}\t ranks:{ranks}\t local_ranks:{local_ranks}")

    train_module = importlib.import_module("train")
    for inputs in get_pre_defined_args(args.tag):
        for key, value in inputs.items():
            setattr(args, key, value)
        utils.show_args(args)
        shared_tensors_with_nfeat = utils.determine_feature_reside_cpu(args, global_nfeat, shared_tensor_list)
        # reimport train.py
        run = importlib.reload(train_module).run
        processes = []
        for i in range(nproc):
            p = mp.Process(target=run, args=(ranks[i], local_ranks[i], world_size, args, shared_tensors_with_nfeat))
            atexit.register(utils.kill_proc, p)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
