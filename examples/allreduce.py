import npc
import torch
import torch.multiprocessing as mp
import atexit
import torch.distributed as dist
import utils


def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass


def run(rank, world_size, args):
    local_rank = rank
    device = torch.device(f"cuda:{local_rank}")
    utils.setup(rank=rank, local_rank=local_rank, world_size=world_size, args=args)

    npc.init(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        init_mp=True,
    )

    send_offset = torch.arange(1, world_size + 1).cumsum(0)
    recv_offset = torch.full((world_size,), rank + 1).cumsum(0)
    feat_dim = 2
    num_input = torch.sum(send_offset).item()
    input = torch.arange(feat_dim * num_input, dtype=torch.float32).reshape(
        num_input, feat_dim
    ) * (1 + rank)
    input = input.to(device)
    print(
        f"[Note]input:{input}\t send:{send_offset}\t recv:{recv_offset}\t feat_dim:{feat_dim}"
    )

    ret = npc.mp_feat_shuffle(
        input=input,
        send_offset=send_offset,
        recv_offset=recv_offset,
        feat_dim=feat_dim,
    )
    print(f"[Note]mp feat shuffle: {ret}")


if __name__ == "__main__":
    nproc = 4
    args = utils.init_args()
    processes = []
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        run,
        args=(
            nproc,
            args,
        ),
        nprocs=nproc,
        join=True,
    )
