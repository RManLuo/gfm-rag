import datetime
import os
from typing import Any

import torch
from torch import distributed as dist


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def cleanup() -> None:
    if get_world_size() > 1:
        dist.destroy_process_group()


def synchronize() -> None:
    if get_world_size() > 1:
        dist.barrier()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(get_local_rank())
    else:
        device = torch.device("cpu")
    return device


def setup_for_distributed(is_master: bool) -> None:
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(timeout: None = None) -> None:
    world_size = get_world_size()
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(get_local_rank())
        if timeout is not None:
            timeout = datetime.timedelta(minutes=timeout)
        dist.init_process_group(
            "nccl", init_method="env://", timeout=timeout, device_id=get_device()
        )
        synchronize()
        setup_for_distributed(get_rank() == 0)
