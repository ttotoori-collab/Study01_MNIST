"""
학습된 web_mnist_cnn.pt 를 브라우저가 읽을 수 있는 형태로 내보낸다.

- weights.bin  : 리틀엔디언 float32를 레이어 순서대로 이어 붙인 한 덩어리
- weights.json : 각 레이어의 형태와 bin 안에서의 위치

자바스크립트는 json을 읽어 하나의 ArrayBuffer 위에 Float32Array 부분 뷰를 만든다.
복사가 일어나지 않아 메모리와 시작 시간을 아낀다.

실행 예)
    python export_weights.py
"""

import json
from pathlib import Path

import numpy as np
import torch

# 이 순서가 곧 weights.bin 의 배치 순서다. 바꾸면 JS 쪽도 함께 바꿔야 한다.
레이어_순서 = [
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
]

평균 = 0.1307
표준편차 = 0.3081


def 내보내기(가중치파일, 출력폴더):
    """.pt 파일을 읽어 weights.bin 과 weights.json 을 쓰고 메타데이터를 돌려준다."""
    출력폴더 = Path(출력폴더)
    출력폴더.mkdir(parents=True, exist_ok=True)
    사전 = torch.load(가중치파일, map_location="cpu")

    조각들, 레이어들, 오프셋 = [], [], 0
    for 이름 in 레이어_순서:
        값 = 사전[이름].detach().cpu().numpy().astype(np.float32).ravel()
        레이어들.append({
            "이름": 이름,
            "형태": list(사전[이름].shape),
            "오프셋": 오프셋,
            "개수": int(값.size),
        })
        조각들.append(값)
        오프셋 += 값.size

    전체 = np.concatenate(조각들).astype("<f4")   # 리틀엔디언 고정
    (출력폴더 / "weights.bin").write_bytes(전체.tobytes())

    메타 = {
        "구조": "web-cnn-v1",
        "정규화": {"평균": 평균, "표준편차": 표준편차},
        "레이어": 레이어들,
    }
    (출력폴더 / "weights.json").write_text(
        json.dumps(메타, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 메타


def _꺼내기(층사전, 이름, 바이트열):
    """bin 바이트열에서 한 레이어를 원래 형태로 꺼낸다."""
    층 = 층사전[이름]
    시작 = 층["오프셋"] * 4
    개수 = 층["개수"]
    값 = np.frombuffer(바이트열, dtype="<f4", count=개수, offset=시작)
    return 값.reshape(층["형태"])


def 넘파이_순전파(층사전, 입력배열, 바이트열):
    """
    내보낸 가중치만으로 순전파를 재현한다. 자바스크립트 구현의 기준값이자
    내보내기가 제대로 됐는지 확인하는 수단이다.
    입력배열은 0~1 범위의 28x28이고 반환값은 로그 확률 10개다.
    """
    꺼내기 = lambda 이름: _꺼내기(층사전, 이름, 바이트열)

    x = ((입력배열 - 평균) / 표준편차).astype(np.float32)[None, :, :]   # (1,28,28)

    def 합성곱(입력, 가중치, 편향, 패딩):
        입력채널, 높이, 너비 = 입력.shape
        출력채널, _, 커널, _ = 가중치.shape
        덧댐 = np.zeros((입력채널, 높이 + 2 * 패딩, 너비 + 2 * 패딩), dtype=np.float32)
        덧댐[:, 패딩:패딩 + 높이, 패딩:패딩 + 너비] = 입력
        출높이 = 높이 + 2 * 패딩 - 커널 + 1
        출너비 = 너비 + 2 * 패딩 - 커널 + 1
        결과 = np.empty((출력채널, 출높이, 출너비), dtype=np.float32)
        for oc in range(출력채널):
            누적 = np.full((출높이, 출너비), 편향[oc], dtype=np.float32)
            for ic in range(입력채널):
                for ky in range(커널):
                    for kx in range(커널):
                        누적 += 가중치[oc, ic, ky, kx] * 덧댐[ic, ky:ky + 출높이, kx:kx + 출너비]
            결과[oc] = 누적
        return 결과

    def 최대풀링(입력, 크기):
        채널, 높이, 너비 = 입력.shape
        잘린 = 입력[:, :높이 // 크기 * 크기, :너비 // 크기 * 크기]
        모양 = (채널, 높이 // 크기, 크기, 너비 // 크기, 크기)
        return 잘린.reshape(모양).max(axis=(2, 4))

    x = 최대풀링(np.maximum(합성곱(x, 꺼내기("conv1.weight"), 꺼내기("conv1.bias"), 1), 0), 2)
    x = 최대풀링(np.maximum(합성곱(x, 꺼내기("conv2.weight"), 꺼내기("conv2.bias"), 1), 0), 2)
    x = x.ravel()
    x = np.maximum(꺼내기("fc1.weight") @ x + 꺼내기("fc1.bias"), 0)
    로짓 = 꺼내기("fc2.weight") @ x + 꺼내기("fc2.bias")
    # log_softmax
    최대 = 로짓.max()
    return (로짓 - 최대 - np.log(np.exp(로짓 - 최대).sum())).astype(np.float32)


def main():
    여기 = Path(__file__).parent
    메타 = 내보내기(여기 / "web_mnist_cnn.pt", 여기.parent / "model")
    전체개수 = sum(층["개수"] for 층 in 메타["레이어"])
    크기 = (여기.parent / "model" / "weights.bin").stat().st_size
    print(f"레이어 {len(메타['레이어'])}개 / 파라미터 {전체개수:,}개")
    print(f"weights.bin {크기:,} 바이트 ({크기 / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
