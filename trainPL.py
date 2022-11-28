import os
import argparse
from config import get_config
from dataset import Potsdam
from torch.utils.data import DataLoader
from trainer import IndexNetTrainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything


parser = argparse.ArgumentParser(description="IndexContrastNet")
# Data
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--work_dir", type=str, required=True, help="output path of checkpoint and log file")
parser.add_argument("--txt_file_dir", type=str, required=True, help="path of image name txt file")
parser.add_argument("--batch_size", type=int, required=True)

# Control
parser.add_argument("--no_pyd", type=bool, default=False, help="if true, only use the last stage of resnet")
parser.add_argument("--no_index", type=bool, default=False, help="if true, not use index mask to map feature")
parser.add_argument("--with_global", type=bool, default=True, help="if true, add global contrast branch")
parser.add_argument("--wandb", type=bool, default=False, help="if true, use wanbd")

# hardware
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')

# checkpoint
parser.add_argument("--load_checkpoint_path", type=str, default=None, help="path to load checkpoint_path."
                                                                           "For pretrain: IndexNetModel path; For transfer-train: IndexNetModel path for backbone;"
                                                                           "For test: deeplabv3+ checkpoint path")


def main():
    args = parser.parse_args()
    config = get_config()
    config.data.batch_size = args.batch_size
    config.network.no_pyd = args.no_pyd
    config.network.no_index = args.no_index
    config.network.with_global = args.with_global
    seed_everything(322)

    dataset = Potsdam(args.data_root, args.txt_file_dir, config)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)

    model = IndexNetTrainer(config)
    callbacks = []
    if args.wandb:
        wandb_logger = WandbLogger(
           name="IndexNetBatchNorm",
            project="IndexNetModel",
            entity="pumpkinn",
            offline=False
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(config)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    checkpoint_callback = ModelCheckpoint(monitor="loss",
                                          dirpath=os.path.join(args.work_dir, "checkpoint"),
                                          filename="IndexNetBatchNorm{epoch:02d}-{loss:.4f}",
                                          save_last=True,
                                          save_on_train_epoch_end=True)
    callbacks.append(checkpoint_callback)
    trainer = Trainer(
                      accelerator="cpu",
                      amp_backend="native",
                      # precision=16,
                      accumulate_grad_batches=config.schedule.accumulate_iter,
                      callbacks=callbacks,
                      logger=wandb_logger if args.wandb else None,
                      max_epochs=config.schedule.epochs,
                      enable_progress_bar=True)
    trainer.fit(model, train_loader)
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    main()