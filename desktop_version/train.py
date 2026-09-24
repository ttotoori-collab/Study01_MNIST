"""
MNIST 데이터셋으로 손글씨 숫자 인식 CNN을 학습시키고
가중치를 mnist_cnn.pt 파일로 저장하는 스크립트.

실행 예)
    python train.py              # 기본 5에폭 학습
    python train.py --epochs 3   # 에폭 수 지정
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from model import MnistCNN

# 학습된 가중치를 저장할 파일 경로 (이 스크립트와 같은 폴더)
가중치_파일 = Path(__file__).with_name("mnist_cnn.pt")


def 학습_한_에폭(모델, 장치, 데이터로더, 옵티마이저, 에폭):
    """한 에폭 동안 학습을 진행하고 진행 상황을 출력한다."""
    모델.train()  # 드롭아웃 등을 학습 모드로 전환
    for 배치번호, (이미지, 정답) in enumerate(데이터로더):
        이미지, 정답 = 이미지.to(장치), 정답.to(장치)
        옵티마이저.zero_grad()            # 이전 배치의 기울기 초기화
        출력 = 모델(이미지)                # 순전파
        손실 = F.nll_loss(출력, 정답)      # 로그 소프트맥스 출력에 맞는 손실 함수
        손실.backward()                   # 역전파로 기울기 계산
        옵티마이저.step()                  # 가중치 갱신

        if 배치번호 % 100 == 0:
            진행률 = 100.0 * 배치번호 / len(데이터로더)
            print(
                f"  에폭 {에폭} [{배치번호 * len(이미지):5d}/{len(데이터로더.dataset)}"
                f" ({진행률:3.0f}%)]  손실: {손실.item():.4f}"
            )


def 평가(모델, 장치, 데이터로더):
    """테스트 데이터로 평균 손실과 정확도를 계산해 출력하고 정확도를 반환한다."""
    모델.eval()  # 평가 모드 (드롭아웃 비활성화)
    총손실 = 0.0
    맞힌개수 = 0
    with torch.no_grad():  # 평가 시에는 기울기를 계산하지 않는다
        for 이미지, 정답 in 데이터로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            출력 = 모델(이미지)
            총손실 += F.nll_loss(출력, 정답, reduction="sum").item()
            예측 = 출력.argmax(dim=1)  # 가장 점수가 높은 숫자를 예측값으로
            맞힌개수 += 예측.eq(정답).sum().item()

    전체개수 = len(데이터로더.dataset)
    평균손실 = 총손실 / 전체개수
    정확도 = 100.0 * 맞힌개수 / 전체개수
    print(f"  [테스트] 평균 손실: {평균손실:.4f} | 정확도: {맞힌개수}/{전체개수} ({정확도:.2f}%)")
    return 정확도


def main():
    파서 = argparse.ArgumentParser(description="MNIST 손글씨 숫자 인식 CNN 학습")
    파서.add_argument("--epochs", type=int, default=5, help="학습 에폭 수 (기본값 5)")
    파서.add_argument("--batch-size", type=int, default=128, help="학습 배치 크기 (기본값 128)")
    파서.add_argument("--lr", type=float, default=1.0, help="Adadelta 학습률 (기본값 1.0)")
    파서.add_argument("--seed", type=int, default=1, help="난수 시드 (기본값 1)")
    인자 = 파서.parse_args()

    torch.manual_seed(인자.seed)  # 결과 재현을 위해 시드 고정

    # GPU가 있으면 GPU, 없으면 CPU 사용
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {장치}")

    # 이미지를 텐서로 바꾸고 MNIST 전체 평균/표준편차로 정규화한다
    전처리 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    데이터경로 = Path(__file__).with_name("data")
    학습셋 = datasets.MNIST(데이터경로, train=True, download=True, transform=전처리)
    테스트셋 = datasets.MNIST(데이터경로, train=False, download=True, transform=전처리)

    학습로더 = torch.utils.data.DataLoader(학습셋, batch_size=인자.batch_size, shuffle=True)
    테스트로더 = torch.utils.data.DataLoader(테스트셋, batch_size=1000, shuffle=False)

    모델 = MnistCNN().to(장치)
    옵티마이저 = optim.Adadelta(모델.parameters(), lr=인자.lr)
    # 에폭마다 학습률을 0.7배로 줄여 후반부에 세밀하게 수렴시킨다
    스케줄러 = StepLR(옵티마이저, step_size=1, gamma=0.7)

    시작시각 = time.time()
    for 에폭 in range(1, 인자.epochs + 1):
        print(f"\n=== 에폭 {에폭}/{인자.epochs} ===")
        학습_한_에폭(모델, 장치, 학습로더, 옵티마이저, 에폭)
        평가(모델, 장치, 테스트로더)
        스케줄러.step()

    걸린시간 = time.time() - 시작시각
    print(f"\n학습 완료 (총 {걸린시간:.1f}초)")

    # 학습된 가중치만 저장한다 (모델 구조는 model.py에 있음)
    torch.save(모델.state_dict(), 가중치_파일)
    print(f"가중치 저장 완료: {가중치_파일}")


if __name__ == "__main__":
    main()
