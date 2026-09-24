"""웹 전용 모델과 전처리 기준 구현에 대한 테스트."""

import unittest
from pathlib import Path

import numpy as np
import torch

from preprocess_ref import 전처리, 정규화
from web_model import 웹CNN


class 웹모델_테스트(unittest.TestCase):
    def test_파라미터_수가_설계대로다(self):
        개수 = sum(p.numel() for p in 웹CNN().parameters())
        self.assertEqual(개수, 105866)

    def test_출력_형태가_배치당_10개다(self):
        출력 = 웹CNN()(torch.zeros(2, 1, 28, 28))
        self.assertEqual(tuple(출력.shape), (2, 10))


class 전처리_테스트(unittest.TestCase):
    def 빈_캔버스(self):
        return np.full((280, 280), 255, dtype=np.uint8)

    def test_빈_캔버스는_None을_돌려준다(self):
        self.assertIsNone(전처리(self.빈_캔버스()))

    def test_점_하나만_찍어도_28x28을_돌려준다(self):
        캔버스 = self.빈_캔버스()
        캔버스[140, 140] = 0
        결과 = 전처리(캔버스)
        self.assertIsNotNone(결과)
        self.assertEqual(결과.shape, (28, 28))

    def test_결과는_0과_1_사이다(self):
        캔버스 = self.빈_캔버스()
        캔버스[100:180, 130:150] = 0
        결과 = 전처리(캔버스)
        self.assertGreaterEqual(결과.min(), 0.0)
        self.assertLessEqual(결과.max(), 1.0)

    def test_가장자리에_그려도_획이_반대편으로_넘어가지_않는다(self):
        # 캔버스 맨 위에 가로로 긴 획을 그린다. 전처리는 이 획을 긴 변 20픽셀로 줄여
        # 28x28 한가운데에 놓으므로 결과는 가운데 몇 줄(13~15행)에만 있어야 하고
        # 위아래 가장자리는 비어 있어야 한다. 무게중심 정렬이 순환 이동이면
        # 밀려난 픽셀이 반대편 가장자리에 나타난다.
        캔버스 = self.빈_캔버스()
        캔버스[0:30, 40:240] = 0
        결과 = 전처리(캔버스)
        self.assertAlmostEqual(결과[:12].sum(), 0.0, places=5)
        self.assertAlmostEqual(결과[17:].sum(), 0.0, places=5)
        y = np.indices(결과.shape)[0]
        self.assertAlmostEqual((y * 결과).sum() / 결과.sum(), 13.5, delta=0.6)

    def test_평행이동은_밀려난_자리를_0으로_채운다(self):
        # 순환 이동(np.roll)이면 맨 윗줄이 맨 아랫줄로 돌아와 합이 그대로 유지된다.
        # 0으로 채우는 이동이면 밖으로 나간 값은 사라져 합이 0이 된다.
        from preprocess_ref import _평행이동
        배열 = np.zeros((4, 4), dtype=np.float64)
        배열[0] = 1.0
        결과 = _평행이동(배열, -1, 0)       # 위로 1칸 밀어 윗줄을 밖으로 내보낸다
        self.assertAlmostEqual(결과.sum(), 0.0, places=6)

    def test_무게중심이_가운데로_옮겨진다(self):
        캔버스 = self.빈_캔버스()
        캔버스[20:80, 20:60] = 0          # 왼쪽 위에 치우친 획
        결과 = 전처리(캔버스)
        y, x = np.indices(결과.shape)
        총합 = 결과.sum()
        self.assertAlmostEqual((y * 결과).sum() / 총합, 13.5, delta=0.6)
        self.assertAlmostEqual((x * 결과).sum() / 총합, 13.5, delta=0.6)

    def test_긴_변이_20픽셀로_맞춰진다(self):
        캔버스 = self.빈_캔버스()
        캔버스[40:240, 120:160] = 0        # 세로로 긴 획
        결과 = 전처리(캔버스)
        채워진행 = np.where(결과.sum(axis=1) > 1e-6)[0]
        self.assertLessEqual(채워진행.max() - 채워진행.min() + 1, 21)

    def test_반올림은_0_5에서_위로_올린다(self):
        from preprocess_ref import _반올림
        self.assertEqual(_반올림(0.5), 1)    # 파이썬 기본 round는 0을 돌려준다
        self.assertEqual(_반올림(1.5), 2)
        self.assertEqual(_반올림(2.5), 3)    # 파이썬 기본 round는 2를 돌려준다
        self.assertEqual(_반올림(-0.5), 0)

    def test_정규화는_학습과_같은_값을_쓴다(self):
        값 = 정규화(np.array([[0.0, 1.0]], dtype=np.float32))
        self.assertAlmostEqual(float(값[0, 0]), (0.0 - 0.1307) / 0.3081, places=5)
        self.assertAlmostEqual(float(값[0, 1]), (1.0 - 0.1307) / 0.3081, places=5)


class 글꼴_분리_테스트(unittest.TestCase):
    """
    학습 글꼴과 벤치마크 글꼴이 겹치면 정확도 게이트가 스스로를 채점하게 된다.
    벤치마크는 학습에서 본 적 없는 글자꼴에 대한 일반화를 재는 잣대여야 한다.
    """

    def test_학습_글꼴과_벤치_글꼴은_겹치지_않는다(self):
        from 손글씨_생성 import 벤치_글꼴, 학습_글꼴
        겹침 = {경로.name for 경로 in 학습_글꼴} & {경로.name for 경로 in 벤치_글꼴}
        self.assertEqual(겹침, set(), f"겹치는 글꼴: {겹침}")

    def test_글꼴_개수가_설계대로다(self):
        from 손글씨_생성 import 벤치_글꼴, 학습_글꼴
        self.assertEqual(len(학습_글꼴), 19)
        self.assertEqual(len(벤치_글꼴), 5)


class 전처리_기준값_테스트(unittest.TestCase):
    """
    커밋된 fixtures.json 은 자바스크립트 이식의 정답지다. 파이썬 전처리를 고치면
    이 테스트가 깨져 'fixture 를 다시 구워라'라는 신호를 준다. 이것이 없으면
    파이썬만 고쳤을 때 두 구현이 조용히 갈라진다.
    """

    def test_fixture_의_기대값을_그대로_재현한다(self):
        import json

        fixture경로 = Path(__file__).parent.parent / "tests" / "fixtures.json"
        자료 = json.loads(fixture경로.read_text(encoding="utf-8"))
        for 번호, 사례 in enumerate(자료["전처리"]):
            회색 = np.empty(사례["폭"] * 사례["높이"], dtype=np.uint8)
            위치 = 0
            for i in range(0, len(사례["rle"]), 2):
                값, 개수 = 사례["rle"][i], 사례["rle"][i + 1]
                회색[위치:위치 + 개수] = 값
                위치 += 개수
            결과 = 전처리(회색.reshape(사례["높이"], 사례["폭"]))
            오차 = np.abs(결과.ravel() - np.array(사례["기대"], dtype=np.float32)).max()
            self.assertLess(float(오차), 1e-3, f"{번호}번 사례 최대 오차 {오차}")


import tempfile

from export_weights import 넘파이_순전파, 내보내기, 레이어_순서, 평균, 표준편차


class 내보내기_테스트(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.모델 = 웹CNN()
        self.모델.eval()
        self.임시폴더 = tempfile.TemporaryDirectory()
        경로 = Path(self.임시폴더.name)
        torch.save(self.모델.state_dict(), 경로 / "임시.pt")
        self.메타 = 내보내기(경로 / "임시.pt", 경로)
        self.출력폴더 = 경로

    def tearDown(self):
        self.임시폴더.cleanup()

    def test_레이어가_여덟개고_순서가_고정이다(self):
        self.assertEqual(레이어_순서, [
            "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
        ])
        self.assertEqual([층["이름"] for 층 in self.메타["레이어"]], 레이어_순서)

    def test_bin_크기가_파라미터_수와_일치한다(self):
        크기 = (self.출력폴더 / "weights.bin").stat().st_size
        self.assertEqual(크기, 105866 * 4)
        self.assertLessEqual(크기, 500 * 1024)

    def test_오프셋이_빈틈없이_이어진다(self):
        다음 = 0
        for 층 in self.메타["레이어"]:
            self.assertEqual(층["오프셋"], 다음)
            다음 += 층["개수"]
        self.assertEqual(다음, 105866)

    def test_넘파이_순전파가_파이토치와_같은_로짓을_낸다(self):
        # 넘파이_순전파는 0~1 입력을 받아 **내부에서** 정규화한다. 따라서 파이토치
        # 기준값을 구할 때도 같은 정규화를 거친 입력을 넣어야 같은 계산을 비교하게 된다.
        # 원본 0~1 을 그대로 파이토치에 넣으면 서로 다른 것을 비교해 오차가 0.07 까지 벌어진다.
        입력 = np.random.RandomState(0).rand(28, 28).astype(np.float32)
        정규화입력 = ((입력 - 평균) / 표준편차).astype(np.float32)
        기대 = self.모델(torch.from_numpy(정규화입력).unsqueeze(0).unsqueeze(0))
        기대로짓 = 기대.detach().numpy()[0]
        실제 = 넘파이_순전파(
            {층["이름"]: 층 for 층 in self.메타["레이어"]},
            입력,
            (self.출력폴더 / "weights.bin").read_bytes(),
        )
        self.assertLess(float(np.abs(실제 - 기대로짓).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
