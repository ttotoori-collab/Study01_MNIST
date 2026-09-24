"""
자바스크립트 구현을 검증할 기준 데이터를 만든다.

- 전처리 fixture : 가짜 손글씨 캔버스와 그것을 전처리한 28x28 결과
- 추론 fixture   : 28x28 입력과 파이토치가 낸 로짓

캔버스는 행 우선 런렝스로 줄여 저장한다. 배경이 대부분이라 크게 줄어든다.

실행 예)
    python make_fixtures.py
"""

import json
import random
from pathlib import Path

import numpy as np
from torchvision import datasets

from bench import 한장_그리기
from export_weights import 넘파이_순전파, 레이어_순서
from preprocess_ref import 전처리

출력_파일 = Path(__file__).parent.parent / "tests" / "fixtures.json"
모델_폴더 = Path(__file__).parent.parent / "model"


def 런렝스로_줄이기(배열):
    """2차원 uint8 배열을 [값, 반복횟수, ...] 형태로 압축한다."""
    평탄 = 배열.ravel()
    결과 = []
    값, 개수 = int(평탄[0]), 1
    for 현재 in 평탄[1:]:
        현재 = int(현재)
        if 현재 == 값:
            개수 += 1
        else:
            결과 += [값, 개수]
            값, 개수 = 현재, 1
    결과 += [값, 개수]
    return 결과


def main():
    random.seed(11)
    np.random.seed(11)

    메타 = json.loads((모델_폴더 / "weights.json").read_text(encoding="utf-8"))
    층사전 = {층["이름"]: 층 for 층 in 메타["레이어"]}
    바이트열 = (모델_폴더 / "weights.bin").read_bytes()
    assert [층["이름"] for 층 in 메타["레이어"]] == 레이어_순서

    # 1) 전처리 fixture: 가짜 손글씨 5장 + 가장자리에 붙여 그린 1장
    전처리_목록 = []
    캔버스들 = [한장_그리기(숫자) for 숫자 in (0, 1, 4, 7, 9)]
    가장자리 = np.full((280, 280), 255, dtype=np.uint8)
    가장자리[0:40, 30:250] = 0                     # 맨 위에 붙은 가로 획
    캔버스들.append(가장자리)
    for 캔버스 in 캔버스들:
        결과 = 전처리(캔버스)
        전처리_목록.append({
            "폭": int(캔버스.shape[1]),
            "높이": int(캔버스.shape[0]),
            "rle": 런렝스로_줄이기(캔버스),
            "기대": [round(float(값), 6) for 값 in 결과.ravel()],
        })

    # 2) 추론 fixture: MNIST 테스트 이미지 20장
    테스트셋 = datasets.MNIST(Path(__file__).with_name("data"), train=False, download=True)
    추론_목록 = []
    for 번호 in range(20):
        이미지, 정답 = 테스트셋[번호]
        입력 = np.array(이미지, dtype=np.float32) / 255.0
        로짓 = 넘파이_순전파(층사전, 입력, 바이트열)
        추론_목록.append({
            "입력": [round(float(값), 6) for 값 in 입력.ravel()],
            "기대로짓": [round(float(값), 6) for 값 in 로짓],
            "정답": int(정답),
        })

    출력_파일.parent.mkdir(parents=True, exist_ok=True)
    출력_파일.write_text(
        json.dumps({"전처리": 전처리_목록, "추론": 추론_목록}, ensure_ascii=False),
        encoding="utf-8",
    )
    크기 = 출력_파일.stat().st_size
    print(f"fixture 저장: {출력_파일} ({크기 / 1024:.0f}KB)")
    print(f"  전처리 {len(전처리_목록)}건 / 추론 {len(추론_목록)}건")


if __name__ == "__main__":
    main()
