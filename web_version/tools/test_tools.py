"""웹 전용 모델과 전처리 기준 구현에 대한 테스트."""

import unittest

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
        # 캔버스 맨 위에 가로로 긴 획을 그린다. 무게중심 정렬이 순환 이동이면
        # 획의 일부가 아래쪽 모서리에 나타난다.
        캔버스 = self.빈_캔버스()
        캔버스[0:30, 40:240] = 0
        결과 = 전처리(캔버스)
        위쪽합 = 결과[:14].sum()
        아래쪽합 = 결과[14:].sum()
        self.assertGreater(위쪽합, 0.0)
        self.assertAlmostEqual(아래쪽합, 0.0, places=5)

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


if __name__ == "__main__":
    unittest.main()
