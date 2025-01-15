import argparse
import os
from zoneinfo import ZoneInfo

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import Compose
import cv2

from dataset import MyDataset
from src.mean_std import get_mean, get_std
from metrics import calc_average_precision
from models.detection_model import DetectionModel
from transformers import Normalize, ToTensor
from utils.config import load_config
from utils.logger import get_logger
from sklearn.metrics import f1_score



logger = get_logger(__name__)
timezone = ZoneInfo("Asia/Tokyo")


def get_arguments():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu, cuda)"
    )
    parser.add_argument(
        "--test_annotations",
        type=str,
        default="/home/ubuntu/slocal/PitchDetection_torimi/data/test_annotations.json",
        help="Path to the test annotations JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_frames",
        help="Directory to save output frames with predictions",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    assert os.path.exists(args.config), f"{args.config} doesn't exist."
    assert os.path.exists(args.checkpoint), f"{args.checkpoint} doesn't exist."
    assert os.path.exists(
        args.test_annotations
    ), f"{args.test_annotations} doesn't exist."

    config = load_config(args.config)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    logger.info("config: \n" + OmegaConf.to_yaml(config))

    logger.info("Loading dataset...")
    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=get_mean(), std=get_std()),
        ]
    )

    test_dataset = MyDataset(
        train=False,
        annot_file_path=args.test_annotations,
        frame_root_dir=config.dataset.frame_root_dir,
        clip_length=config.dataset.clip_length,
        transform=transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.testing.batch_size,
        shuffle=False,
        num_workers=config.testing.num_workers,
    )

    logger.info(f"Test dataset size: {len(test_dataset)}")

    logger.info("Loading model...")
    model = DetectionModel(
        cnn=config.model.cnn,
        train_entire_cnn=config.model.train_entire_cnn,
        in_channel=config.model.in_channel,
        n_features=config.model.n_features,
        n_classes=config.model.n_classes,
        n_stages=config.model.n_stages,
        n_layers=config.model.n_layers,
    )

    checkpoint = torch.load(args.checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    device = torch.device(args.device)
    model = torch.nn.DataParallel(model).to(device)

    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Start testing...")
    with torch.no_grad():
        total_ap = 0
        total_f1 = 0
        total_samples = 0

        for i, (x, y, frame_paths) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            # Use the final output for evaluation
            y_pred_final = (
                y_pred[-1].permute(0, 2, 1).reshape(-1, config.model.n_classes)
            )
            y_final = (
                y.permute(0, 2, 1).reshape(-1, config.model.n_classes).argmax(dim=1)
            )

            ap = calc_average_precision(y_pred_final, y_final)[0]
            f1 = f1_score(
                y_final.cpu(), y_pred_final.argmax(dim=1).cpu(), average="weighted"
            )

            total_ap += ap * x.size(0)
            total_f1 += f1 * x.size(0)
            total_samples += x.size(0)

            # Save predictions and corresponding frames
            for batch_idx in range(x.size(0)):
                frame_path = frame_paths[batch_idx]
                frame_predictions = y_pred[-1][batch_idx].permute(1, 0).cpu().numpy()
                frame_labels = y[batch_idx].permute(1, 0).cpu().numpy()

                # Load the original frame image
                frame_img = cv2.imread(frame_path)
                if frame_img is None:
                    logger.warning(f"Could not read frame: {frame_path}")
                    continue

                # Add predictions and labels to the frame as text
                for t, (pred, label) in enumerate(zip(frame_predictions, frame_labels)):
                    pred_class = np.argmax(pred)
                    label_class = np.argmax(label)
                    text = f"Pred: {pred_class}, Label: {label_class}"
                    cv2.putText(
                        frame_img,
                        text,
                        (10, 30 + t * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if pred_class == label_class else (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Save the frame with annotations
                output_file = os.path.join(
                    args.output_dir, os.path.basename(frame_path)
                )
                cv2.imwrite(output_file, frame_img)

        mean_ap = total_ap / total_samples
        mean_f1 = total_f1 / total_samples

        logger.info(f"Mean Average Precision: {mean_ap}")
        logger.info(f"Mean F1 Score: {mean_f1}")


if __name__ == "__main__":
    main()
