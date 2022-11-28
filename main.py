import os
import cv2
import time
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from config import get_config
from dataset.dataset import Potsdam
import torch.backends.cudnn as cudnn
from utils.metrics import StreamSegMetric
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import validate, Denormalize
from utils.utils import get_logger, save_check_point
from deeplabv3plus.deeplabc3_plus import DeeplabV3Plus


parser = argparse.ArgumentParser(description="IndexContrastNet")
# Data
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--work_dir", type=str, required=True, help="output path of checkpoint and log file")
parser.add_argument("--txt_file_dir", type=str, required=True, help="path of image name txt file")
parser.add_argument("--batch_size", type=int, required=True)

# Control
parser.add_argument("--mode", type=int, default=1,
                    help="1: fine-tuning train; 2: fine-tuning test 3:Inference")
parser.add_argument("--no_pyd", type=bool, default=False, help="if true, only use the last stage of resnet")
parser.add_argument("--no_index", type=bool, default=False, help="if true, not use index mask to map feature")
parser.add_argument("--use_wandb", type=bool, default=False, help="if true, use wandb")

# hardware
parser.add_argument("--gpu_counts", type=int, default=0)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--mixed_precision", default=False, help="Whether to use mixed precision")

# checkpoint
parser.add_argument("--resumed_checkpoint_path", type=str, default=None, help="path to resumed train")
parser.add_argument("--load_checkpoint_path", type=str, default=None, help="path to load checkpoint_path."
                                                                           "For pretrain: IndexNetModel path; For transfer-train: IndexNetModel path for backbone;"
                                                                           "For test: deeplabv3+ checkpoint path")
parser.add_argument("--pretrain_backbone_mode", type=int, default=0, help="0:None; 1: IndexContrast 2:ImageNet")
parser.add_argument("--save_inference_result_raw", default=True, help="Whether to save raw inference result")
parser.add_argument("--save_inference_result_blind", default=True,
                    help="Whether to save inference result blinded with img")


def transfer(args, config, logger, device):
    print("=> Create transfering model")
    if args.pretrain_backbone_mode == 2:
        model = DeeplabV3Plus(config, mode="train", pretrain_base=True)
    else:
        model = DeeplabV3Plus(config, mode="train", pretrain_base=False)

    if args.load_checkpoint_path is not None and args.resumed_checkpoint_path is None:
        checkpoint = torch.load(args.load_checkpoint_path)
        model.load_backbone_checkpoint(checkpoint["backbone"])
    # if args.load_checkpoint_path is not None:
    #     checkpoint = torch.load(args.load_checkpoint_path)
    #     model.load_backbone_checkpoint(checkpoint, strict=False)

    print("=> creating dataset and data loader")
    train_dataset = Potsdam(args.data_root, args.txt_file_dir, config, split="train")
    logger.info(f"Load:{len(train_dataset)} Train Image")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                                               num_workers=args.workers, shuffle=True, drop_last=True)

    val_dataset = Potsdam(args.data_root, args.txt_file_dir, config, split="test")
    logger.info(f"Load:{len(val_dataset)} Val Image")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True,
                                             num_workers=args.workers, shuffle=True, drop_last=True)

    print("=> creating optimizer")
    base_params = list(map(id, model.backbone.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": config.transfer_schedule.lr},
        {"params": model.backbone.parameters(), "lr": config.transfer_schedule.backbone_lr},
    ]
    optimizer = torch.optim.SGD(params, momentum=config.transfer_schedule.momentum,
                                weight_decay=config.transfer_schedule.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.transfer_schedule.epochs,
                                                           eta_min=1e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=config.transfer_schedule.weight)

    best_score = 0.0
    start_epoch = 0
    best_loss = None
    if args.resumed_checkpoint_path is not None:
        checkpoint = torch.load(args.resumed_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["start_epoch"]
        best_score = checkpoint["best_score"]

    metrics = StreamSegMetric(len(train_dataset.CLASSES))
    checkpoint_path = os.path.join(args.work_dir, "transfer_checkpoint")
    os.makedirs(checkpoint_path, 0o777, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.work_dir, "tranfer_tensorboard_log"))

    print("=> Start training!")
    interval_loss = 0
    iters = 0

    model.to(device)
    criterion.to(device)

    for epoch in range(start_epoch, config.transfer_schedule.epochs):
        model.train()
        loss_hist = []
        for i, imgdict in enumerate(train_loader):
            img = imgdict["img"].to(device=device, dtype=torch.float32)
            label = imgdict["label"].to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            loss_hist.append(np_loss)
            interval_loss += np_loss

            iters += 1
            if (iters) % 10 == 0:
                interval_loss = interval_loss / 10
                logger.info(
                    "Epoch %d/%d, Itrs %d, Loss=%.6f" % (epoch, config.transfer_schedule.epochs, iters, interval_loss))
                writer.add_scalar("loss", loss.item(), iters)
                interval_loss = 0.0

        logger.info("Epoch %d \t  Loss = %.6f" % (epoch, np.mean(loss_hist)))
        if epoch % config.transfer_schedule.validate_interval == 0:
            print("=> Validation...")
            vis_sample_id = np.random.randint(0, len(val_loader), config.others.vis_num_samples,
                                              np.int32) if config.others.enable_vis else None  # sample idxs for visualization
            model.eval()
            val_score, ret_samples = validate(config, args, model, loader=val_loader, device=device, metrics=metrics,
                                              ret_samples_ids=vis_sample_id)
            logger.info(metrics.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                save_check_point(checkpoint_path, model, optimizer, epoch, np.mean(loss_hist), scheduler=scheduler,
                                 best_score=best_score, save_backbone=False)

            # tensorBoard
            writer.add_scalar("val mIoU", val_score["Mean IoU"], epoch)
            writer.add_scalar("val OA", val_score["Overall Acc"], epoch)
            writer.add_scalar("val Kappa", val_score["kappa"], epoch)
            for index, class_name in enumerate(val_dataset.CLASSES):
                writer.add_scalar(class_name + " acc", val_score["Class Acc"][index], epoch)
                writer.add_scalar(class_name + " iu", val_score["Class IoU"][index], epoch)
                logger.info("%20s \t ACC: %.6f \t IOU: %.6f." % (
                class_name, val_score['Class Acc'][index], val_score['Class IoU'][index]))
        elif best_loss is None or np.mean(loss_hist) < best_loss or epoch % 15 == 0:
            save_check_point(checkpoint_path, model, optimizer, epoch, np.mean(loss_hist), scheduler=scheduler,
                             best_score=best_score, save_backbone=False)
        scheduler.step()


def test(args, config, logger, device):
    print("=> Create test model")
    model = DeeplabV3Plus(config, mode="test", pretrain_base=False)
    assert args.load_checkpoint_path is not None

    checkpoint = torch.load(args.load_checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    print("=> creating dataset and data loader")
    test_dataset = Potsdam(args.data_root, args.txt_file_dir, config, split="test")
    logger.info(f"Load:{len(test_dataset)} test Image")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                                              num_workers=args.workers, shuffle=True, drop_last=True)

    metrics = StreamSegMetric(len(test_dataset.CLASSES))

    print("=> Validation...")
    vis_sample_id = np.random.randint(0, len(test_loader), config.others.vis_num_samples,
                                      np.int32) if config.others.enable_vis else None  # sample idxs for visualization
    model.eval()
    val_score, ret_samples = validate(config, args, model, loader=test_loader, device=device, metrics=metrics,
                                      ret_samples_ids=vis_sample_id)
    logger.info(metrics.to_str(val_score))
    for index, class_name in enumerate(test_dataset.CLASSES):
        logger.info("%20s \t ACC: %.6f \t IOU: %.6f." % (
        class_name, val_score['Class Acc'][index], val_score['Class IoU'][index]))


def inference(args, config, logger, device, opacity=0.5):
    print("=====>start inference")
    model = DeeplabV3Plus(config, mode="test", pretrain_base=False)
    assert args.load_checkpoint_path is not None

    checkpoint = torch.load(args.load_checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    print("=> creating dataset and data loader")
    test_dataset = Potsdam(args.data_root, args.txt_file_dir, config, split="test")
    logger.info(f"Load:{len(test_dataset)} inference Image")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True,
                                              num_workers=args.workers, shuffle=False, drop_last=True)

    if args.save_inference_result_raw:
        raw_out_path = os.path.join(args.work_dir, "inference_result_raw")
        os.makedirs(raw_out_path, 0o777, exist_ok=True)

    if args.save_inference_result_blind:
        blind_out_path = os.path.join(args.work_dir, "inference_result_blind")
        os.makedirs(blind_out_path, 0o777, exist_ok=True)

    for index, imgdict in enumerate(tqdm(test_loader)):
        model.eval()
        img = imgdict["img"].to(device)
        labels = imgdict["label"].to(device).squeeze()
        outputs = model(img)
        preds = outputs.detach().max(dim=1)[1].cpu().numpy().astype(np.uint8)[0]

        denorm = Denormalize(mean=[0.33797, 0.3605, 0.3348],
                             std=[0.1359, 0.1352, 0.1407])
        img = img.detach().cpu().numpy().squeeze()
        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)

        palette = np.array(test_dataset.PALETTE)
        color_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        color_seg_true = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[preds == label, :] = color
            color_seg_true[labels == label, :] = color
        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)

        if args.save_inference_result_raw:
            color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(raw_out_path, str(index) + ".png"), color_seg)

        if args.save_inference_result_blind:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(blind_out_path, str(index) + ".png"), img)


def main():
    args = parser.parse_args()
    if args.use_wandb:
        wandb.login()

    config = get_config()
    config.data.batch_size = args.batch_size
    config.network.no_pyd = args.no_pyd
    config.network.no_index = args.no_index

    os.makedirs(args.work_dir, 0o777, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.work_dir, "log")
    os.makedirs(log_dir, 0o777, exist_ok=True)
    log_file = os.path.join(log_dir, f'{timestamp}.log')
    logger = get_logger(name="IndexContrast", output=log_file, distributed_rank=0, color=True)
    logger.info('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    logger.info('\n'.join(f'{k}={v}' for k, v in vars(config).items()))

    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    if args.gpu_counts == 0:
        device = torch.device("cpu")
        logger.info("Not Use GPU For task")
    else:
        if (args.gpu_counts > 1):
            raise "Multiprocessing Note implement!"
        else:
            logger.info("Use GPU: 0 for training")
            device = torch.device("cuda:0")

    if (args.mode == 1):
        transfer(args, config, logger, device)
    elif args.mode == 2:
        test(args, config, logger, device)
    elif args.mode == 3:
        inference(args, config, logger, device)
    else:
        raise "Please check your mode selection!"


if __name__ == '__main__':
    main()
