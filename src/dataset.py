import json
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

EVENT_IDX_TO_NAME = {0: "none", 1: "pitch", 2: "pick_off"}

EVENT_NAME_TO_IDX = {v: k for k, v in EVENT_IDX_TO_NAME.items()}


class MyDataset(Dataset):
    def __init__(
        self,
        train: bool,
        annot_file_path: str,
        frame_root_dir: str,
        clip_length: int,
        transform: object = None,
    ) -> None:
        self.train = train
        self.frame_root_dir = frame_root_dir
        self.clip_length = clip_length
        self.transform = transform

        assert os.path.exists(annot_file_path), f"{annot_file_path:} doesn't exist."
        assert os.path.exists(frame_root_dir), f"{frame_root_dir} doesn't exist."


        with open(annot_file_path, "r") as annot_file:
            annot = json.load(annot_file)

        self.annot = {}
        self.idx_to_event = {}
        idx = 0

        self.video_name_list = [item["name"] for item in annot if "name" in item]
        # if self.train:
        #     self.video_name_list.remove("vsWASEDA3T.mp4")

        # Create annotation dictionary
        # annot: List[Dict[str, Any]]
        for item in annot:
            idx_in_video = 0
            video_name = item["name"]
            # if video_name == "vsWASEDA3T.mp4":
            #     continue
            frame_count = item["frameCount"]
            frame_dir = os.path.join(self.frame_root_dir, video_name.split(".")[0])
            events = {}
            for event_annotation in item["attributes"]:
                if len(event_annotation["value"]) == 0:
                    continue

                keyframes = []
                for keyframe in event_annotation["value"]:
                    keyframes.append([keyframe[0] - 1, keyframe[1] - 1])
                    self.idx_to_event[idx] = {
                        "event_idx": EVENT_NAME_TO_IDX[event_annotation["key"]],
                        "event_name": event_annotation["key"],
                        "video_name": video_name,
                        "idx_in_video": idx_in_video,
                        "start_frame": keyframe[0] - 1,
                        "end_frame": keyframe[1] - 1,
                        "frame_count": frame_count,
                    }
                    idx_in_video += 1
                    idx += 1

                events[event_annotation["key"]] = keyframes

            self.annot[video_name] = {
                "frame_count": frame_count,
                "frame_dir": frame_dir,
                "events": events,
            }

            if self.train:
                self.roi_annot_path = "/home/ubuntu/slocal/PitchDetection/json/ROI_annot_annotations.json"
            else:
                self.roi_annot_path = "/home/ubuntu/slocal/PitchDetection/json/val_ROI_annotations.json"

            self.roi_dict = self._get_roi(self.roi_annot_path)


    def _get_roi(self, annot_file_path):
        with open(annot_file_path, "r") as annot_file:
            annot = json.load(annot_file)
        values_dict = {}
        for video in annot:
            video_name = video.get("name")
            annotations = video.get("annotations", [])
            if annotations:  # Check if annotations exist
                points = annotations[0].get("points", {})
                first_value = points.get("1", {}).get("value", [])
                values_dict[video_name] = first_value
        return values_dict

    def __len__(self) -> int:
        # Return the number of events if training, otherwise return the number of divided frames
        if self.train:
            events_count = 0
            for video_name in self.video_name_list:
                for event_name, keyframes in self.annot[video_name]["events"].items():
                    events_count += len(keyframes)

            return events_count
        else:
            devided_frame_count = 0
            for video_name in self.video_name_list:
                devided_frame_count += (
                    self.annot[video_name]["frame_count"] // self.clip_length
                )
            return devided_frame_count

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frames and event annotation.
        While training, this method returns frames and event annotation near the event.
        While validation, this method returns frames and event annotation in the video.

        Args:
            idx (int): index
        """
        if self.train:
            event = self.idx_to_event[idx]
            event_name = event["event_name"]
            video_name = event["video_name"]
            idx_in_video = event["idx_in_video"]
            event_start_frame = event["start_frame"]
            event_end_frame = event["end_frame"]
            frame_count = event["frame_count"]
            #print("video_name",video_name)
            # Randomly select a clip
            # if the random number is less than 0.5, the clip is selected before the event
            # besides, if the random number is greater than or equal to 0.5, the clip is selected after the event
            if random.random() < 0.5:
                clip_start_frame = event_start_frame - random.randint(
                    0, self.clip_length
                )
                if clip_start_frame < 0:
                    clip_start_frame = 0
                clip_end_frame = clip_start_frame + self.clip_length - 1
                if clip_end_frame >= frame_count:
                    clip_end_frame = frame_count - 1
                    clip_start_frame = clip_end_frame - self.clip_length + 1
            else:
                clip_end_frame = event_end_frame + random.randint(0, self.clip_length)
                if clip_end_frame >= frame_count:
                    clip_end_frame = frame_count - 1
                clip_start_frame = clip_end_frame - self.clip_length + 1
                if clip_start_frame < 0:
                    clip_start_frame = 0
                    clip_end_frame = clip_start_frame + self.clip_length - 1

            frames = self._get_frames(video_name, clip_start_frame, clip_end_frame)
            event_annotation = self._get_event_anntoation(
                video_name, clip_start_frame, clip_end_frame
            )
        else:
            for video_name in self.video_name_list:
                if idx < self.annot[video_name]["frame_count"] // self.clip_length:
                    clip_start_frame = idx * self.clip_length
                    clip_end_frame = clip_start_frame + self.clip_length - 1

                    frames = self._get_frames(
                        video_name, clip_start_frame, clip_end_frame
                    )
                    event_annotation = self._get_event_anntoation(
                        video_name, clip_start_frame, clip_end_frame
                    )
                    break
                else:
                    idx -= self.annot[video_name]["frame_count"] // self.clip_length

        frames_np = np.array(frames)
        event_annotation = np.array(event_annotation)
        
        if self.transform:
            frames = self.transform(frames_np)
            
        meta = {
            "video_name": video_name,
            # "frames": frames,
        }

        return frames, event_annotation, meta

    def _get_frames(
        self, video_name: str, start_frame: int, end_frame: int
    ) -> List[np.ndarray]:
        """
        Get frames from video.

        Args:
            video_name (str): video name
            start_frame (int): start frame index
            end_frame (int): end frame index
        """
        frames = []
        roi_random = random.random()
        for i in range(start_frame, end_frame + 1):
            frame_path = os.path.join(
                self.annot[video_name]["frame_dir"], f"frame_{i:05d}.jpg"
            )
            assert os.path.exists(frame_path), f"{frame_path} doesn't exist."
            frame = cv2.imread(frame_path)
            # if self.train:
            #     roi_value = self.roi_dict[video_name]
            #     #print("frame_path", frame_path)
            #     if roi_random < 0.5:
            #         x1, y1, x2, y2 = map(int, roi_value)
            #         x1, x2, y1, y2 = determine_suited_roi(frame.shape[1], frame.shape[0], {"x1": x1, "x2": x2, "y1": y1, "y2": y2})
            #         frame = frame[y1:y2, x1:x2]
            #ここから
            apply_roi = self.train and roi_random < 0.5 or not self.train  # trainは50%、valは常に適用

            if video_name in self.roi_dict and apply_roi:
                roi_value = self.roi_dict[video_name]
                x1, y1, x2, y2 = map(int, roi_value)
                x1, x2, y1, y2 = determine_suited_roi(
                    frame.shape[1], frame.shape[0],
                    {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
                )
                frame = frame[y1:y2, x1:x2]

            #ここ

            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames

    def _get_event_anntoation(
        self, video_name: str, start_frame: int, end_frame: int
    ) -> np.ndarray:
        """
        Get event annotation.

        Args:
            video_name (str): video name
            start_frame (int): start frame index
            end_frame (int): end frame index
        """
        inner_frame_index = np.arange(start_frame, end_frame + 1)
        event_annotation = np.zeros((2, self.clip_length))
        event_annotation[0] = 1

        # label for each event
        # if the event is included in the clip, the label is 1
        # event_idx: 0: none, 1: pitch, 2: pick_off
        for i, (event_name, keyframes) in enumerate(
            self.annot[video_name]["events"].items()
        ):
            if event_name == "pitch":
                for start_keyframe, end_keyframe in keyframes:
                    inner_keyframe_index = np.arange(start_keyframe, end_keyframe + 1)
                    for keyframe in inner_keyframe_index:
                        if keyframe in inner_frame_index:
                            event_annotation[1, keyframe - start_frame] = 1
                            event_annotation[0, keyframe - start_frame] = 0
            elif event_name == "pick_off":
                for start_keyframe, end_keyframe in keyframes:
                    inner_keyframe_index = np.arange(start_keyframe, end_keyframe + 1)
                    for keyframe in inner_keyframe_index:
                        if keyframe in inner_frame_index:
                            event_annotation[1, keyframe - start_frame] = 1
                            event_annotation[0, keyframe - start_frame] = 0
            else:
                raise ValueError(f"Unknown event name: {event_name}")

        return event_annotation

def determine_suited_roi(original_width: int, original_height: int, roi: List[float], resize: List[int] = [224, 224]) -> List[float]:
    # x1, x2, y1, y2 = int(roi["x1"] * width), int(roi["x2"] * width), int(roi["y1"] * height), int(roi["y2"] * height)
    x1, x2, y1, y2 = roi["x1"], roi["x2"], roi["y1"], roi["y2"]

    width, height = x2 - x1, y2 - y1
    center = [(x1 + x2) / 2, (y1 + y2) / 2]
    aspect_ratio = width / height
    target_aspect_ratio = resize[0] / resize[1]
    if aspect_ratio == target_aspect_ratio:
        # width is same as height
        pass
    elif aspect_ratio > target_aspect_ratio:
        # width is longer than height
        if width * target_aspect_ratio >= original_height:
            # height is longer than original height
            width = original_height / target_aspect_ratio
            x1, x2 = center[0] - width / 2, center[0] + width / 2
            center[1] = original_height / 2
            y1, y2 = 0, original_height
            if x1 < 0:
                x1, x2 = 0, width
            elif x2 > original_width:
                x1, x2 = original_width - width, original_width
            center[0] = (x1 + x2) / 2
        else:
            height = width / target_aspect_ratio
            y1, y2 = center[1] - height / 2, center[1] + height / 2
            if y1 < 0:
                y1, y2 = 0, height
            elif y2 > original_height:
                y1, y2 = original_height - height, original_height
            center[1] = (y1 + y2) / 2
    else:
        # height is longer than width
        if height * target_aspect_ratio >= original_width:
            # width is longer than original width
            height = original_width / target_aspect_ratio
            y1, y2 = center[1] - height / 2, center[1] + height / 2
            center[0] = original_width / 2
            x1, x2 = 0, original_width
            if y1 < 0:
                y1, y2 = 0, height
            elif y2 > original_height:
                y1, y2 = original_height - height, original_height
            center[1] = (y1 + y2) / 2
        else:
            width = height * target_aspect_ratio
            x1, x2 = center[0] - width / 2, center[0] + width / 2
            if x1 < 0:
                x1, x2 = 0, width
            elif x2 > original_width:
                x1, x2 = original_width - width, original_width
            center[0] = (x1 + x2) / 2
                
    assert x1 >= 0 and x2 <= original_width, "x1: {}, x2: {}, original_width: {}".format(x1, x2, original_width)
    assert y1 >= 0 and y2 <= original_height, "y1: {}, y2: {}, original_height: {}".format(y1, y2, original_height)
        
    suited_roi = [int(x1), int(x2), int(y1), int(y2)]
    return suited_roi
