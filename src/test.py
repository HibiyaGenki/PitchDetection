import argparse
import os
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import Compose
import cv2
import json

from dataset import MyDataset
from mean_std import get_mean, get_std
from metrics import calc_average_precision
from models.detection_model import DetectionModel
from transformers import Normalize, ToTensor
from utils.config import load_config
from utils.logger import get_logger
from sklearn.metrics import accuracy_score, f1_score, accuracy_score, recall_score, classification_report

warnings.filterwarnings("ignore", category=UserWarning)


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
    # parser.add_argument(
    #     "--device", type=str, default="cpu", help="Device to use (cpu, cuda)"
    # )
    # parser.add_argument(
    #     "--frame_root_dir",
    #     type=str,
    #     required=True,
    #     help="Path to the root directory containing video-specific frame directories",
    # )
    # parser.add_argument(
    #     "--test_annotations",
    #     type=str,
    #     required=True,
    #     help="Path to the test annotations JSON file",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="output_frames",
    #     help="Directory to save output frames with predictions",
    # )
    return parser.parse_args()




def main():
    args = get_arguments()
    assert os.path.exists(args.config), f"{args.config} doesn't exist."
    assert os.path.exists(args.checkpoint), f"{args.checkpoint} doesn't exist."
    #assert os.path.exists(args.frame_root_dir), f"{args.frame_root_dir} doesn't exist."
    #assert os.path.exists(args.test_annotations), f"{args.test_annotations} doesn't exist."

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
        annot_file_path=config.dataset.test_annot_file_path,
        frame_root_dir=config.dataset.frame_root_dir,
        clip_length=config.dataset.clip_length,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.testing.batch_size * len(config.training.gpu_ids),
        shuffle=False,
        num_workers=config.testing.num_workers,
    )




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
    
    def load_module_state_dict(model, state_dict):
        for k, v in state_dict.items():
            if k[:7] == "module.":
                if k[7:] in model.state_dict():
                    model.state_dict()[k[7:]].copy_(v)
                else:
                    print(f"Key {k} not found in model state dict")
            else:
                if k in model.state_dict():
                    model.state_dict()[k].copy_(v)
                else:
                    print(f"Key {k} not found in model state dict")

    if "model_state_dict" in checkpoint:
        # model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        load_module_state_dict(model, checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        # model.load_state_dict(checkpoint["state_dict"], strict=True)
        load_module_state_dict(model, checkpoint["state_dict"])
    else:
        try:
            load_module_state_dict(model, checkpoint)
        except:
            raise ValueError("No valid state dict found in the checkpoint file")
    
    #device = torch.device(args.device)
    #model = torch.nn.DataParallel(model).cuda()
    def extract_max_values(array):
        values = [0 if pair[0] > pair[1] else 1 for pair in array]
        return values
    criterion = torch.nn.CrossEntropyLoss()
    model = model.cuda()
    model.eval()
    with torch.no_grad():
            test_loss = 0
            total_ap = 0
            total_accuracy = 0
            total_recall_macro = 0
            total_recall_micro = 0
            total_recall_weighted = 0
            # output_list = []
            total_f1 = 0
            # output_dict = {
            #         "0":[],
            #         "1":[],
            #         "2":[],
            #         "3":[],
            #         "4":[]}
            
            y_preds_original = {}
            preds_dict = {}
            for i, (x, y, meta) in enumerate(test_loader):
                video_name = meta["video_name"][0]
                # x, y = x.to(device), y.to(device)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                y_pred = model(x)
                #print(y, y_pred)
                loss = criterion(y_pred, y)
                test_loss += loss.item()

                #mstcnの時はloss = criterion(y_pred[-1], y)
                # print("ypred",y_pred[0].size(),len(y_pred))
                # batch_size, n_classes, n_frames -> batch_size * n_frames, n_classes
                
                # y_pred = y_pred[-1].permute(0, 2, 1).reshape(-1, config.model.n_classes)
                # result = extract_max_values(y_pred)
                
                y = y.permute(0, 2, 1).reshape(-1, config.model.n_classes).argmax(dim=1)
                y_pred = y_pred.permute(0, 2, 1).reshape(-1, config.model.n_classes)
                #MSTCNはy_pred = y_pred[-1].permute(0, 2, 1).reshape(-1, config.model.n_classes)
                ap = calc_average_precision(y_pred, y)[0]

                if video_name in y_preds_original.keys():
                    y_preds_original[video_name].append(y_pred.tolist())
                else:
                    y_preds_original[video_name] = y_pred.tolist()

                
                
                y_pred = y_pred.argmax(dim=1).cpu()
                
                #output_list.extend(result)
                # batch_size, n_classes, n_frames -> batch_size * n_frames, n_classes -> batch_size * n_frames
                
                y = y.cpu()
                
                #f1 = f1_score(y.cpu(), result)
                # accuracy = accuracy_score(y.cpu(), result)
                # recall_macro = recall_score(y.cpu(), result, average='macro')  # 各クラスの平均
                # recall_micro = recall_score(y.cpu(), result, average='micro')  # 全体のリコール
                # recall_weighted = recall_score(y.cpu(), result, average='weighted')  # ク
                
                f1 = f1_score(y, y_pred, average='macro')
                accuracy = accuracy_score(y, y_pred)
                recall_macro = recall_score(y, y_pred, average='macro', zero_division=0)
                recall_micro = recall_score(y, y_pred, average='micro', zero_division=0)
                recall_weighted = recall_score(y, y_pred, average='weighted', zero_division=0)


                total_ap += ap
                total_f1 += f1
                total_accuracy += accuracy
                total_recall_macro += recall_macro
                total_recall_micro += recall_micro
                total_recall_weighted += recall_weighted
                
                if video_name in preds_dict.keys():
                    preds_dict[video_name].append(y_pred.tolist())
                else:
                    preds_dict[video_name] = y_pred.tolist()

                # if i <= 6231 // 256:
                #     output_dict[0].extend(result)
                # elif i <= 6231 // 256 + 2891 // 256:
                #     output_dict[1].extend(result)
                # elif i <= 6231 // 256 + 2891 // 256 + 4955 // 256:
                #     output_dict[2].extend(result)
                # elif i <= 6231 // 256 + 2891 // 256 + 4955 // 256 + 5248 // 256:
                #     output_dict[3].extend(result)
                # else:
                #     output_dict[4].extend(result)

            # with open('output.txt', 'w') as file:
            #     for value in output_list:
            #         file.write(f"{value}")

            with open('test_lstm_original_predictions.json', 'w') as file:
                json.dump(y_preds_original, file, indent=4)
            
            with open('test_lstm_predictions.json', 'w') as file:
                json.dump(preds_dict, file, indent=4)
                
            
            logger.info(
                f"test Loss {test_loss / len(test_loader)}, AP {total_ap / len(test_loader)},f1 {total_f1 / len(test_loader)}, accuracy {total_accuracy / len(test_loader)}, recall macro {total_recall_macro / len(test_loader)}, recall micro {total_recall_micro / len(test_loader)}, recall wighted {total_recall_weighted / len(test_loader)}"
            )

    #labeled_output_dir = os.path.join(args.output_dir, "labels")
    #os.makedirs(labeled_output_dir, exist_ok=True)

    # # Iterate over video-specific frame directories
    # video_dirs = [
    #     os.path.join(args.frame_root_dir, d)
    #     for d in sorted(os.listdir(args.frame_root_dir))
    #     if os.path.isdir(os.path.join(args.frame_root_dir, d))
    # ]

    # for video_dir in video_dirs:
    #     video_name = os.path.basename(video_dir)
    #     video_output_file = os.path.join(labeled_output_dir, f"{video_name}.txt")

    #     frame_paths = [
    #         os.path.join(video_dir, f)
    #         for f in sorted(os.listdir(video_dir))
    #         if f.endswith(".jpg")
    #     ]

    #     correct = 0
    #     total = 0

    #     with open(video_output_file, "w") as label_file:
    #         with torch.no_grad():
    #             for frame_path in frame_paths:
    #                 frame_name = os.path.basename(frame_path)
    #                 ground_truth = annotations.get(frame_name, "Unknown")

    #                 frame = cv2.imread(frame_path)
    #                 if frame is None:
    #                     logger.warning(f"Could not read frame: {frame_path}")
    #                     continue

    #                 input_tensor = transform(frame).unsqueeze(0).to(device)
    #                 prediction = model(input_tensor)

    #                 # Assume final output layer gives class probabilities
    #                 pred_class = prediction[-1].argmax(dim=1).item()
    #                 predicted_label = "Pitch" if pred_class == 1 else "None"

    #                 # Compare with ground truth
    #                 if predicted_label == ground_truth:
    #                     correct += 1
    #                 total += 1

    #                 # Write frame label to file
    #                 label_file.write(
    #                     f"{frame_name}: Predicted={predicted_label}, GroundTruth={ground_truth}\n"
    #                 )

       # accuracy = correct / total if total > 0 else 0
       # logger.info(f"Accuracy for video '{video_name}': {accuracy:.2%}")
       

if __name__ == "__main__":
    main()
