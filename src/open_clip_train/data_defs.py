# ===== 文件: src/open_clip_train/data_defs.py (新创建) =====

from dataclasses import dataclass
from torch.utils.data import DataLoader, DistributedSampler
from multiprocessing import Value

@dataclass
class DataInfo:
    """
    一个用于封装 Dataloader 及其相关信息的简单数据类。
    """
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: 'SharedEpoch' = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class SharedEpoch:
    """
    一个使用多进程安全值的类，用于在 Dataloader 的工作进程之间同步 epoch 计数。
    """
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value