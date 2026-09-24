"""
웹 버전이 쓰는 경량 CNN 정의.

데스크톱 모델은 완전연결 층 하나가 파라미터의 98%를 차지해 4.8MB가 된다.
여기서는 풀링을 한 번 더 넣어 완전연결 입력을 9216에서 1568로 줄였다.
파라미터 105,866개, float32로 약 414KB다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class 웹CNN(nn.Module):
    """28x28 흑백 손글씨를 0~9로 분류하는 경량 합성곱 신경망."""

    def __init__(self):
        super().__init__()
        # padding=1 을 주어 합성곱 뒤에도 가로세로 크기가 유지되게 한다
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """순전파. 입력 x의 형태는 (배치, 1, 28, 28)."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # (N, 16, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # (N, 32, 7, 7)
        x = torch.flatten(x, 1)                       # (N, 1568)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)
