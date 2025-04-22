import argparse
import datetime
import os
import warnings
from zoneinfo import ZoneInfo
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, f1_score
from torchvision.transforms import Compose
import torch.nn.functional as F

from dataset import MyDataset
from mean_std import get_mean, get_std
from metrics import calc_average_precision
from models.detection_model import DetectionModel
from transformers import ColorJitter, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation
from utils.config import load_config, save_config
from utils.logger import get_logger, set_logger
from loss_fn.focal import FocalLoss

warnings.filterwarnings("ignore", category=UserWarning)

timezone = ZoneInfo("Asia/Tokyo")
logger = get_logger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    # parser.add_argument(
    #     "--device", type=str, default="cpu", help="Device to use (cpu, cuda)"
    # )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to the checkpoint to resume"
    )
    return parser.parse_args()


def make_transform(config) -> Dict[str, Any]:
    transform = []
    if config.transform.color_jitter:
        transform.append(
            ColorJitter(
                brightness=list(config.transform.color_jitter_brightness),
                contrast=list(config.transform.color_jitter_contrast),
                saturation=list(config.transform.color_jitter_saturation),
                hue=list(config.transform.color_jitter_hue),
            )
        )
    if config.transform.random_horizontal_flip:
        transform.append(
            RandomHorizontalFlip(
                p=config.transform.random_horizontal_flip_p,
            )
        )
    if config.transform.random_rotation:
        transform.append(
            RandomRotation(
                degrees=config.transform.random_rotation_degrees,
            )
        )
    transform.append(ToTensor())
    transform.append(Normalize(mean=get_mean(), std=get_std()))
    return Compose(transform)


def main():
    args = get_arguments()
    assert os.path.exists(args.config), f"{args.config} doesn't exist."

    config = load_config(args.config)

    log_dir = os.path.join(
        config.training.log_dir,
        datetime.datetime.now(tz=timezone).strftime("%Y-%m-%d-%H-%M"),
    )
    os.makedirs(log_dir, exist_ok=True)
    set_logger(logger, os.path.join(log_dir, "train.log"))
    save_config(config, os.path.join(log_dir, "config.yaml"))

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    logger.info("config: \n" + OmegaConf.to_yaml(config))

    logger.info("Loading dataset...")
    train_transform = make_transform(config)
    val_transform = Compose(
        [
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std()),
        ]
    )

    train_dataset = MyDataset(
        train=True,
        annot_file_path=config.dataset.train_annot_file_path,
        frame_root_dir=config.dataset.frame_root_dir,
        clip_length=config.dataset.clip_length,
        transform=train_transform,
    )
    val_dataset = MyDataset(
        train=False,
        annot_file_path=config.dataset.val_annot_file_path,
        frame_root_dir=config.dataset.frame_root_dir,
        clip_length=config.dataset.clip_length,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size ,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    logger.info("Loading model...")
    model = DetectionModel(
        cnn=config.model.cnn,
        sequence_model=config.model.sequence_model,
        train_entire_cnn=config.model.train_entire_cnn,
        in_channel=config.model.in_channel,
        n_features=config.model.n_features,
        n_classes=config.model.n_classes,
        n_stages=config.model.n_stages,
        n_layers=config.model.n_layers,
    )

    if args.resume is not None:
        # model.load_state_dict(torch.load(args.resume))
        states = torch.load(args.resume)
        model.load_state_dict(states["model_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])
        scheduler.load_state_dict(states["scheduler_state_dict"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma,
        )

    # device = torch.device(args.device)

    # model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=config.training.gpu_ids).cuda()
    model = model.cuda()  # シングルGPUで動かす


    ce_criterion = torch.nn.CrossEntropyLoss()
    # mse_loss = torch.nn.MSELoss()
    focal_criterion = FocalLoss()

    logger.info("Start training...")
    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0
        for i, (x, y, _) in enumerate(train_loader):
            # x, y = x.to(device), y.to(device)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            if y.ndim == 3 and y.shape[1] > 1:
                y = y.argmax(dim=1)

            y_preds = model(x) # (batch_size, n_classes, n_frames)
            loss = 0
            if isinstance(y_preds, list):  # MSTCN
                for y_pred in y_preds:
                    loss += ce_criterion(y_pred, y)
            else:  # Other models
                loss = ce_criterion(y_preds, y)
            
    
            # y_pred_flat = y_pred.view(-1, 2)  
            # y_flat      = y.view(-1)
            #loss += ce_criterion(y_pred, y)
                # loss += focal_criterion(y_pred, y)
                # print(y_pred.size(),y.size(), loss.size(), len(y_preds))
                # loss += 0.15*mse_loss(y_pred, y)
                # print(mse_loss(y_pred, y))

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % config.training.log_interval == 0 or i == len(train_loader) - 1:
                logger.info(f"Epoch {epoch}, Iter {i}, Loss {total_loss / (i+1)}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            total_ap = 0
            total_f1 = 0
            for i, (x, y, _) in enumerate(val_loader):
                # x, y = x.to(device), y.to(device)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                y_preds = model(x)
                # loss = ce_criterion(y_pred[-1], y) + focal_criterion(y_pred[-1], y)
                # loss = ce_criterion(y_pred[-1], y)
                #loss = ce_criterion(y_pred, y)
                #mstcnの時はloss = ce_criterion(y_pred[-1], y)
                if y.ndim == 3 and y.shape[1] > 1:
                        y = y.argmax(dim=1)
                        
                if isinstance(y_preds, list):  # MSTCN
                    loss = ce_criterion(y_preds[-1], y)  # 最終ステージのみloss計算（val時）
                    pred_for_eval = y_preds[-1]
                else:
                    loss = ce_criterion(y_preds, y)
                    pred_for_eval = y_preds
                val_loss += loss.item()
                pred_for_eval = pred_for_eval.permute(0, 2, 1).reshape(-1, config.model.n_classes)
                if y.ndim == 3:
                    y = y.permute(0, 2, 1).reshape(-1, config.model.n_classes).argmax(dim=1)
                else:
                    y = y.reshape(-1)


                # batch_size, n_classes, n_frames -> batch_size * n_frames, n_classes
                #y_pred = y_pred.permute(0, 2, 1).reshape(-1, config.model.n_classes)
                #MSTCNはy_pred = y_pred[-1].permute(0, 2, 1).reshape(-1, config.model.n_classes)
                # batch_size, n_classes, n_frames -> batch_size * n_frames, n_classes -> batch_size * n_frames
                #y = y.permute(0, 2, 1).reshape(-1, config.model.n_classes).argmax(dim=1)
                # logger.info(f"[DEBUG] y unique labels: {y.unique().cpu().numpy()}")
                # logger.info(f"[DEBUG] pred_for_eval unique preds: {pred_for_eval.argmax(dim=1).unique().cpu().numpy()}")

                ap = calc_average_precision(pred_for_eval, y)[0]

                f1 = f1_score(y.cpu(), pred_for_eval.argmax(dim=1).cpu(), average="macro")

                total_ap += ap
                total_f1 += f1

            logger.info(
                f"Epoch {epoch}, Val Loss {val_loss / len(val_loader)}, AP {total_ap / len(val_loader)}, F1 {total_f1 / len(val_loader)}"
            )

        if epoch % config.training.save_interval == 0:
            save_path = os.path.join(log_dir, "weights", f"model_{epoch}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            save_states = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(save_states, os.path.join(log_dir, f"checkpoint.pth"))
            logger.info(f"Model saved at {save_path}")


if __name__ == "__main__":
    main()
