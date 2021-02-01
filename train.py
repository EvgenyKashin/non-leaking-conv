from argparse import ArgumentParser

import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from model import LitResnet


def make_cifar10_dm(batch_size, num_workers):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])

    cifar10_dm = CIFAR10DataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    return cifar10_dm


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    # parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--experiment_name', type=str, required=True,
                               help='folder for experiment logs')
    parent_parser.add_argument('--batch_size', type=int, default=32)
    parent_parser.add_argument('--num_workers', type=int, default=4)
    parent_parser.add_argument('--max_epochs', type=int, default=40)
    parent_parser.add_argument('--gpus', type=int, default=0)

    parser = LitResnet.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler="simple",
    )
    args = parser.parse_args()
    main(args)


def main(args):
    pl.seed_everything(24)

    model = LitResnet(**vars(args))
    model.datamodule = make_cifar10_dm(args.batch_size, args.num_workers)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=pl.loggers.TensorBoardLogger('lightning_logs/',
                                            name=args.experiment_name),
        callbacks=[LearningRateMonitor(logging_interval='step')])

    trainer.fit(model, model.datamodule)
    trainer.test(model, datamodule=model.datamodule)


if __name__ == '__main__':
    run_cli()
