from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from models.pytorch_i3d import load_i3d
from models.tcn import MultiStageTCN


class DetectionModel(nn.Module):
    def __init__(
        self,
        cnn: str,
        train_entire_cnn: bool,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cnn_name = cnn
        if cnn == "resnet50":
            self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.cnn.fc = nn.Linear(2048, in_channel)
            if not train_entire_cnn:
                for param in self.cnn.parameters():
                    param.requires_grad = False
                self.cnn.fc.requires_grad = True
        elif cnn == "i3d":
            self.cnn = load_i3d()
            if not train_entire_cnn:
                for param in self.cnn.parameters():
                    param.requires_grad = False
            assert in_channel == 1024, "I3D's in_channel is fixed to 1024."
        else:
            raise ValueError(f"Invalid CNN: {cnn}")

        # self.tcn = MultiStageTCN(in_channel, n_features, n_classes, n_stages, n_layers)
        self.lstm = nn.LSTM(in_channel, hidden_size = 128, num_layers  = 2, batch_first = True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, c, h, w = x.size()
        if self.cnn_name == "i3d":
            # (batch_size, seq_len, c, h, w) -> (batch_size, c, seq_len, h, w)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = self.cnn.extract_features(x)
            x = F.upsample(
                x, size=(seq_len, 1, 1), mode="trilinear", align_corners=False
            )
            x = x.squeeze(4).squeeze(3)
            x = x.permute(0, 2, 1).contiguous()
        elif self.cnn_name == "resnet50":
            # (batch_size, seq_len, c, h, w) -> (batch_size * seq_len, c, h, w)
            x = x.view(-1, c, h, w)
            x = self.cnn(x)

        # (batch_size * seq_len, in_channel) -> (batch_size, seq_len, in_channel)
        x = x.view(batch_size, seq_len, -1).permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        # (batch_size, seq_len, in_channel) -> (batch_size, n_classes, seq_len)
        out, (h_n, c_n) = self.lstm(x)
        # print("out_size",out.size())
        out = self.fc(out)
        out = out.permute(0,2,1).contiguous()
        return out
        # 最後の時刻ステップのみを全結合層に通す例
        x = self.tcn(x)
        return x
   
    


# 入力データの形状: (batch_size, seq_len, c, h, w)（動画のバッチ）。
# 形状を変換: ResNet に渡すために、各フレームを個別のサンプルとして扱う。
# ResNet で特徴量抽出: フレームごとの高次元特徴を取得。
# 時系列データに再構成: フレームの特徴量を時系列データとして整理。
# TCN に入力: 時系列モデルでフレーム間の関係を学習し、最終出力を得る。


if __name__ == "__main__":
    model = DetectionModel(512, 256, 3, 4, 10)
    x = torch.randn(1, 1, 3, 224, 224)
    y = model(x)[-1]
    y = F.softmax(y, dim=1)
    print(y)
