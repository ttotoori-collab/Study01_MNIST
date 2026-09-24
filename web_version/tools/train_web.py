"""
웹 버전용 경량 CNN을 학습시키고 web_mnist_cnn.pt 로 저장한다.

사람이 마우스로 쓴 글씨는 MNIST보다 기울기와 크기 변화가 크다. 그래서 학습할 때
어파인 증강(회전/이동/확대축소/전단)을 걸어 그 변형을 미리 겪게 한다.

MNIST만으로는 부족하다. MNIST의 숫자 모양은 특정 필기 스타일에 치우쳐 있어서,
가혹 벤치마크에 쓰이는 일부 글꼴(예: 세리프 깃발과 밑변이 있는 '1')은 MNIST에
아예 없는 모양이다. 그래서 다양한 글꼴로 그린 숫자를 실제 전처리까지 거쳐
학습 표본에 섞는다. 이때 쓰는 학습_글꼴은 bench.py의 벤치_글꼴과 겹치지 않아야
벤치마크가 진짜 일반화를 재는 독립적인 잣대로 남는다.

실행 예)
    python train_web.py --epochs 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from preprocess_ref import 전처리
from web_model import 웹CNN
from 손글씨_생성 import 학습_글꼴, 한장_그리기

가중치_파일 = Path(__file__).with_name("web_mnist_cnn.pt")
데이터_경로 = Path(__file__).with_name("data")
글꼴표본_파일 = 데이터_경로 / "글꼴표본.npz"


class 획굵기_변화:
    """
    획을 무작위로 굵게(팽창) 또는 가늘게(침식) 만든다.
    사람마다 붓 굵기가 다른 것을 흉내 낸다. MNIST는 흰 획/검은 배경이므로
    최대 풀링이 팽창, 최소 풀링의 부호를 뒤집은 것이 침식이 된다.
    정규화 전, 0~1 텐서 상태에서 적용해야 한다.
    """

    def __init__(self, 확률=0.4):
        self.확률 = 확률

    def __call__(self, 텐서):
        if torch.rand(1).item() > self.확률:
            return 텐서
        크기 = 3
        덩어리 = 텐서.unsqueeze(0)
        if torch.rand(1).item() < 0.5:
            결과 = torch.nn.functional.max_pool2d(덩어리, 크기, stride=1, padding=크기 // 2)
        else:
            결과 = -torch.nn.functional.max_pool2d(-덩어리, 크기, stride=1, padding=크기 // 2)
        return 결과.squeeze(0)


def _글꼴표본_생성(캐시_파일, 개수당):
    """학습용 글꼴로 숫자를 그려 실제 전처리를 거친 표본을 만들어 캐시한다."""
    print(f"글꼴 표본 생성 중... (숫자당 {개수당}장, 학습_글꼴 {len(학습_글꼴)}종)")
    시작 = time.time()
    이미지들, 레이블들 = [], []
    for 숫자 in range(10):
        모은수 = 0
        while 모은수 < 개수당:
            캔버스 = 한장_그리기(숫자, 학습_글꼴)
            이미지 = 전처리(캔버스)
            if 이미지 is None:
                continue
            이미지들.append(이미지)
            레이블들.append(숫자)
            모은수 += 1
        print(f"  숫자 {숫자}: {모은수}장 완료")
    캐시_파일.parent.mkdir(parents=True, exist_ok=True)
    np.savez(캐시_파일, 이미지=np.stack(이미지들), 레이블=np.array(레이블들, dtype=np.int64))
    print(f"글꼴 표본 {len(레이블들)}장 저장 완료 ({time.time() - 시작:.0f}초): {캐시_파일}")


class 글꼴숫자_데이터셋(torch.utils.data.Dataset):
    """폰트로 그려 전처리까지 거친 숫자 표본. 캐시를 읽고 에폭마다 다른 변형을 추가로 건다."""

    def __init__(self, 캐시_파일, 개수당=2000):
        if not 캐시_파일.exists():
            _글꼴표본_생성(캐시_파일, 개수당)
        데이터 = np.load(캐시_파일)
        self.이미지 = 데이터["이미지"]
        self.레이블 = 데이터["레이블"]
        self.변형 = transforms.Compose([
            transforms.RandomAffine(
                degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6, fill=0,
            ),
            획굵기_변화(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def __len__(self):
        return len(self.레이블)

    def __getitem__(self, 색인):
        텐서 = torch.from_numpy(self.이미지[색인]).unsqueeze(0)
        return self.변형(텐서), int(self.레이블[색인])


def 데이터로더_만들기(배치크기):
    """학습용에는 증강을, 테스트용에는 정규화만 적용한다."""
    학습_전처리 = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,                  # 삐뚤게 쓴 글씨
            translate=(0.12, 0.12),      # 가운데에서 벗어난 글씨
            scale=(0.8, 1.2),            # 크게 또는 작게 쓴 글씨
            shear=12,                    # 기울여 쓴 글씨
            fill=0,
        ),
        transforms.ToTensor(),
        획굵기_변화(),                    # 붓 굵기 차이를 흉내 낸다
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    테스트_전처리 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    학습셋_MNIST = datasets.MNIST(데이터_경로, train=True, download=True, transform=학습_전처리)
    글꼴셋 = 글꼴숫자_데이터셋(글꼴표본_파일, 개수당=2000)
    학습셋 = torch.utils.data.ConcatDataset([학습셋_MNIST, 글꼴셋])
    테스트셋 = datasets.MNIST(데이터_경로, train=False, download=True, transform=테스트_전처리)
    return (
        torch.utils.data.DataLoader(학습셋, batch_size=배치크기, shuffle=True),
        torch.utils.data.DataLoader(테스트셋, batch_size=1000, shuffle=False),
    )


def 평가(모델, 장치, 데이터로더):
    """테스트 정확도를 백분율로 돌려준다."""
    모델.eval()
    맞음 = 0
    with torch.no_grad():
        for 이미지, 정답 in 데이터로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            맞음 += 모델(이미지).argmax(dim=1).eq(정답).sum().item()
    return 100.0 * 맞음 / len(데이터로더.dataset)


def main():
    파서 = argparse.ArgumentParser(description="웹 버전 경량 CNN 학습")
    파서.add_argument("--epochs", type=int, default=20)
    파서.add_argument("--batch-size", type=int, default=128)
    파서.add_argument("--lr", type=float, default=1.0)
    파서.add_argument("--seed", type=int, default=1)
    인자 = 파서.parse_args()

    torch.manual_seed(인자.seed)
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {장치}")

    학습로더, 테스트로더 = 데이터로더_만들기(인자.batch_size)
    모델 = 웹CNN().to(장치)
    옵티마이저 = optim.Adadelta(모델.parameters(), lr=인자.lr)
    스케줄러 = StepLR(옵티마이저, step_size=3, gamma=0.7)

    최고정확도 = 0.0
    시작 = time.time()
    for 에폭 in range(1, 인자.epochs + 1):
        모델.train()
        for 이미지, 정답 in 학습로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            옵티마이저.zero_grad()
            손실 = F.nll_loss(모델(이미지), 정답)
            손실.backward()
            옵티마이저.step()

        정확도 = 평가(모델, 장치, 테스트로더)
        표시 = ""
        if 정확도 > 최고정확도:
            # 가장 좋았던 시점의 가중치만 남긴다
            최고정확도 = 정확도
            torch.save(모델.state_dict(), 가중치_파일)
            표시 = "  <- 저장"
        print(f"에폭 {에폭:2d}/{인자.epochs}  테스트 정확도 {정확도:.2f}%{표시}")
        스케줄러.step()

    print(f"최고 정확도 {최고정확도:.2f}% / 소요 {time.time() - 시작:.0f}초")
    print(f"가중치 저장 완료: {가중치_파일}")


if __name__ == "__main__":
    main()
