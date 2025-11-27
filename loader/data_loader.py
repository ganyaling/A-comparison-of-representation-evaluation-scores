

from torchvision import datasets
from loader.view_generator import DataAugmentationMOCO, DataAugmentationDINO
from torch.utils.data import DataLoader, DistributedSampler
import os


class ContrastiveDataLoader:
    def __init__(self, args, dataset_name, method="moco"):
        """
        args: 参数对象，包含 data_path, batch_size_per_gpu, num_workers, distributed, etc.
        dataset_name: "cifar10", "mnist", "tinyimagenet"
        method: "dino" 或 "moco"，控制数据增强策略
        """

        self.args = args
        self.dataset_name = dataset_name
        self.method = method.lower()

        # build transform
        self.transform = self._build_transform()

        # build dataset
        self.dataset = self._build_dataset()

        # build sampler
        self.sampler = DistributedSampler(self.dataset, shuffle=True) if args.distributed else None

        # build DataLoader
        if self.method == 'moco':
            self.data_loader = DataLoader(
                self.dataset,
                batch_size=args.batch_size,
                shuffle=(self.sampler is None),
                sampler=self.sampler,
                num_workers=args.num_workers,
                pin_memory=getattr(args, 'pin_memory', False),
                drop_last=True,
            )
        else:
            self.data_loader = DataLoader(
                self.dataset,
                batch_size=args.batch_size_per_gpu,
                shuffle=(self.sampler is None),
                sampler=self.sampler,
                num_workers=args.num_workers,
                pin_memory=getattr(args, 'pin_memory', False),
                drop_last=True,
            )

    def _build_transform(self):
        if self.method == "dino":
            return DataAugmentationDINO(
                global_crops_scale=self.args.global_crops_scale,
                local_crops_scale=self.args.local_crops_scale,
                local_crops_number=self.args.local_crops_number,
                use_grayscale3=getattr(self.args, 'use_grayscale3', False),
            )
        elif self.method == "moco":
            return DataAugmentationMOCO(
                self.args.global_crops_scale,
                use_grayscale3=getattr(self.args, 'use_grayscale3', False),
            )
        else:
            raise ValueError(f"Unknown method {self.method}")

    def _build_dataset(self):
        if self.dataset_name == "cifar10":
            return datasets.CIFAR10(self.args.data_path, train=True, transform=self.transform, download=True)
        elif self.dataset_name == "mnist":
            return datasets.MNIST(self.args.data_path, train=True, transform=self.transform, download=True)
        elif self.dataset_name == "tinyimagenet":
            return datasets.ImageFolder(
                os.path.join(self.args.data_path, "tiny-imagenet-200", "train"),
                transform=self.transform,
            )
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
