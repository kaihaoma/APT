import torch
import torch.multiprocessing as mp
import utils
import atexit
import importlib


def get_pre_defined_args():
    cache_memory_in_gbs = [1]
    # cache_memory_in_gbs = [0, 1, 2, 3, 4, 5]
    system = ["DP"]
    # generate args
    for try_times in range(1):
        for cache_mem in cache_memory_in_gbs:
            for sys in system:
                cm = cache_mem * 1024 * 1024 * 1024
                args = f"system={sys};cache_memory={cm}"
                # ;cache_mode=dryrun
                yield args

    while False:
        inputs = input("args: ")
        if inputs == "":
            continue
        if inputs == "exit":
            exit(0)
        yield inputs


if __name__ == "__main__":
    args, shared_tensor_list = utils.pre_spawn()
    world_size = args.world_size

    nproc = world_size if args.nproc_per_node == -1 else args.nproc_per_node
    processes = []
    ranks = args.ranks
    local_ranks = args.local_ranks
    print(
        f"[Note]procs:{nproc}\t world_size:{world_size}\t ranks:{ranks}\t local_ranks:{local_ranks}"
    )

    train_module = importlib.import_module("train")
    for inputs in get_pre_defined_args():
        comm_key_value = inputs.split(";")
        for comm in comm_key_value:
            key, value = comm.split("=")
            print(f"[Note]comm:{comm}\t key:{key}\t value:{value}")
            if key == "cache_memory":
                value = int(value)
            setattr(args, key, value)
        # reimport train.py
        run = importlib.reload(train_module).run
        for i in range(nproc):
            p = mp.Process(
                target=run,
                args=(ranks[i], local_ranks[i], world_size, args, shared_tensor_list),
            )
            atexit.register(utils.kill_proc, p)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
