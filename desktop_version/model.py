"""
MNIST 손글씨 숫자 인식용 CNN 모델 정의 모듈.

학습 스크립트(train.py)와 손글씨 입력 앱(app.py)이
동일한 신경망 구조를 공유하기 위해 별도 파일로 분리했다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """28x28 흑백 손글씨 이미지를 0~9 중 하나로 분류하는 합성곱 신경망."""

    def __init__(self):
        super().__init__()
        # 합성곱 1층: 입력 1채널(흑백) -> 출력 32채널, 3x3 커널
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # 합성곱 2층: 32채널 -> 64채널, 3x3 커널
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # 과적합을 막기 위한 드롭아웃
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 완전연결 1층: 12*12*64 = 9216개의 특징을 128차원으로 압축
        self.fc1 = nn.Linear(9216, 128)
        # 완전연결 2층: 128차원 -> 10개 숫자(0~9) 점수
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """순전파. 입력 x의 형태는 (배치, 1, 28, 28)."""
        x = F.relu(self.conv1(x))   # (N, 32, 26, 26)
        x = F.relu(self.conv2(x))   # (N, 64, 24, 24)
        x = F.max_pool2d(x, 2)      # (N, 64, 12, 12) 가로세로 크기를 절반으로 축소
        x = self.dropout1(x)
        x = torch.flatten(x, 1)     # (N, 9216) 1차원으로 펼치기
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)             # (N, 10) 각 숫자에 대한 로짓 값
        # 로그 소프트맥스를 적용해 로그 확률로 변환
        return F.log_softmax(x, dim=1)
