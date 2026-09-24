"""
채점자가 캔버스에 직접 그리는 상황을 흉내 내어 정확도를 잰다.

손글씨 계열 글꼴로 0~9를 쓰고 손떨림(탄성 왜곡), 획 굵기 변화, 기울기, 회전,
위치 이동을 더한 뒤 실제 전처리를 거쳐 모델에 넣는다.

여기서 쓰는 벤치_글꼴은 학습(train_web.py)이 본 적 없는 글꼴이다. 학습 글꼴로
벤치마크를 통과시키면 일반화가 아니라 암기를 재는 셈이라 의미가 없다.

실행 예)
    python bench.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

from preprocess_ref import 전처리, 정규화
from web_model import 웹CNN
from 손글씨_생성 import 벤치_글꼴
from 손글씨_생성 import 한장_그리기 as _한장_그리기

가중치_파일 = Path(__file__).with_name("web_mnist_cnn.pt")
목표_정확도 = 98.5


def 한장_그리기(숫자):
    """벤치마크용 글꼴로 가짜 손글씨 한 장을 그린다."""
    return _한장_그리기(숫자, 벤치_글꼴)


def 측정(모델, 장치, 표본수=400):
    """숫자별로 고르게 표본을 만들어 (맞은개수, 전체개수, 혼동표)를 돌려준다."""
    숫자당 = 표본수 // 10
    맞음, 전체, 혼동 = 0, 0, {}
    for 숫자 in range(10):
        for _ in range(숫자당):
            이미지 = 전처리(한장_그리기(숫자))
            if 이미지 is None:
                continue
            입력 = torch.from_numpy(정규화(이미지)).unsqueeze(0).unsqueeze(0).to(장치)
            with torch.no_grad():
                예측 = int(모델(입력).argmax(1).item())
            전체 += 1
            if 예측 == 숫자:
                맞음 += 1
            else:
                혼동[(숫자, 예측)] = 혼동.get((숫자, 예측), 0) + 1
    return 맞음, 전체, 혼동


def main():
    random.seed(7)
    np.random.seed(7)
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    모델 = 웹CNN().to(장치)
    모델.load_state_dict(torch.load(가중치_파일, map_location=장치))
    모델.eval()

    맞음, 전체, 혼동 = 측정(모델, 장치)
    정확도 = 100.0 * 맞음 / 전체
    print(f"가혹 조건 정확도: {맞음}/{전체} ({정확도:.1f}%)  목표 {목표_정확도}%")
    if 혼동:
        print("주요 혼동:", sorted(혼동.items(), key=lambda 항목: -항목[1])[:10])
    if 정확도 < 목표_정확도:
        print("목표 미달")
        sys.exit(1)
    print("목표 달성")


if __name__ == "__main__":
    main()
