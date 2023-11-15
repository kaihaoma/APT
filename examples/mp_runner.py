import torch
import torch.multiprocessing as mp
import utils
import atexit
import importlib


def get_pre_defined_args(tag_prefix):
    num_try_times = 1
    # cache_memory_in_gbs = [1]
    cache_memory_in_gbs = list(range(8))
    system = ["DP", "NP", "SP", "MP"]
    models = ["SAGE", "GCN", "GAT"]
    # num_localnode_feats_in_workers = list(range(4, 8))
    num_localnode_feats_in_workers = [-1]
    # generate args

    for try_times in range(num_try_times):
        for nl in num_localnode_feats_in_workers:
            for cache_mem in cache_memory_in_gbs:
                for model in models:
                    for sys in system:
                        cm = cache_mem * 1024 * 1024 * 1024
                        # cross-machine feat loading case
                        tag = f"t{try_times}_{sys}_{model}_nl{nl}of8_cm{cache_mem}GB"
                        # single-machine case
                        # tag = f"{tag_prefix}_{sys}_cm{cache_mem}GB"
                        yield {"system": sys, "model": model, "cache_memory": cm, "num_localnode_feats_in_workers": nl, "tag": tag}


def get_user_input(tag_prefix):
    try_times = 0
    sys = "DP"
    model = "SAGE"
    while True:
        user_args = input("Please input the args:").split(";")
        try:
            nl = int(user_args[0])
            cache_mem = int(user_args[1])
            assert nl >= 0 and nl <= 8
            assert cache_mem >= 0 and cache_mem < 6

            cm = cache_mem * 1024 * 1024 * 1024
            tag = f"test_t{try_times}_{sys}_{model}_nl{nl}of8_cm{cache_mem}GB"
            try_times += 1
            yield {"system": sys, "model": model, "cache_memory": cm, "num_localnode_feats_in_workers": nl, "tag": tag}
        except:
            sys = "DP"
            model = "SAGE"
            nl = 4
            cache_mem = 1
            cm = cache_mem * 1024 * 1024 * 1024

            tag = f"test_t{try_times}_{sys}_nl{nl}of8_cm{cache_mem}GB"
            yield {"system": sys, "model": model, "cache_memory": cm, "num_localnode_feats_in_workers": nl, "tag": tag}


if __name__ == "__main__":
    args, shared_tensor_list, global_nfeat = utils.pre_spawn()
    world_size = args.world_size
    nproc = world_size if args.nproc_per_node == -1 else args.nproc_per_node
    ranks = args.ranks
    local_ranks = args.local_ranks
    print(f"[Note]procs:{nproc}\t world_size:{world_size}\t ranks:{ranks}\t local_ranks:{local_ranks}")

    train_module = importlib.import_module("train")
    for inputs in get_pre_defined_args(args.tag):
        # for inputs in get_user_input(args.tag):
        for key, value in inputs.items():
            setattr(args, key, value)
        if args.model == "GAT":
            assert args.num_heads == 8
            args.num_hidden = 8
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
