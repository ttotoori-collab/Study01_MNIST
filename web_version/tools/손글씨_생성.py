"""
가짜 손글씨 캔버스를 만드는 공용 렌더링/왜곡 로직.

train_web.py(학습용 합성 표본)와 bench.py(가혹 벤치마크)가 이 모듈을 함께 쓴다.
두 글꼴 목록은 절대 겹치지 않는다 — 학습에서 본 적 없는 글꼴로 벤치마크를 재야
모델이 글꼴 자체를 외운 게 아니라 진짜 일반화를 했는지 확인할 수 있다.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

글꼴_폴더 = Path(r"C:\Windows\Fonts")

# 학습용: 다양한 굵기/스타일의 글꼴로 모델이 폭넓게 일반화하도록 한다.
학습_글꼴 = [
    글꼴_폴더 / "comic.ttf",
    글꼴_폴더 / "comicbd.ttf",
    글꼴_폴더 / "segoepr.ttf",
    글꼴_폴더 / "segoeprb.ttf",
    글꼴_폴더 / "Gabriola.ttf",
    글꼴_폴더 / "arial.ttf",
    글꼴_폴더 / "arialbd.ttf",
    글꼴_폴더 / "times.ttf",
    글꼴_폴더 / "timesbd.ttf",
    글꼴_폴더 / "tahoma.ttf",
    글꼴_폴더 / "tahomabd.ttf",
    글꼴_폴더 / "consola.ttf",
    글꼴_폴더 / "cour.ttf",
    글꼴_폴더 / "courbd.ttf",
    글꼴_폴더 / "verdana.ttf",
    글꼴_폴더 / "trebuc.ttf",
    글꼴_폴더 / "segoeui.ttf",
    글꼴_폴더 / "segoeuib.ttf",
    글꼴_폴더 / "l_10646.ttf",
]

# 벤치마크 전용: 학습_글꼴과 겹치지 않는다. 처음 보는 글꼴에 대한 일반화를 잰다.
벤치_글꼴 = [
    글꼴_폴더 / "Inkfree.ttf",
    글꼴_폴더 / "segoesc.ttf",
    글꼴_폴더 / "segoescb.ttf",
    글꼴_폴더 / "calibri.ttf",
    글꼴_폴더 / "georgia.ttf",
]


def _존재_확인(글꼴목록, 이름):
    """
    글꼴 파일이 실제로 있는지 모듈을 불러올 때 미리 확인한다.
    없으면 표본을 수천 번 만드는 도중 알 수 없는 OSError로 죽는 대신,
    어떤 파일이 없는지 바로 알려 준다.
    """
    없음 = [str(경로) for 경로 in 글꼴목록 if not 경로.exists()]
    if 없음:
        raise FileNotFoundError(f"{이름}에 없는 글꼴 파일: {', '.join(없음)}")


_존재_확인(학습_글꼴, "학습_글꼴")
_존재_확인(벤치_글꼴, "벤치_글꼴")


def _박스흐림(판, 반경):
    """PIL이 실수형 이미지를 흐리게 하지 못해 누적합으로 직접 구현한다."""
    반경 = max(1, int(반경))
    누적 = np.cumsum(np.pad(판, ((반경, 반경), (0, 0)), mode="edge"), axis=0)
    판 = (누적[2 * 반경:] - 누적[:-2 * 반경]) / (2 * 반경)
    누적 = np.cumsum(np.pad(판, ((0, 0), (반경, 반경)), mode="edge"), axis=1)
    return (누적[:, 2 * 반경:] - 누적[:, :-2 * 반경]) / (2 * 반경)


def _탄성왜곡(배열, 세기, 부드러움):
    """손떨림처럼 픽셀을 물결치듯 밀어 낸다."""
    높이, 너비 = 배열.shape
    dx = np.random.uniform(-1, 1, (높이, 너비)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (높이, 너비)).astype(np.float32)
    for _ in range(3):                       # 박스 흐림 3회는 가우시안에 가깝다
        dx, dy = _박스흐림(dx, 부드러움), _박스흐림(dy, 부드러움)
    dx *= 세기 * 8                            # 흐림으로 줄어든 진폭을 되살린다
    dy *= 세기 * 8
    y, x = np.indices((높이, 너비))
    return 배열[np.clip((y + dy).round().astype(int), 0, 높이 - 1),
                np.clip((x + dx).round().astype(int), 0, 너비 - 1)]


def 한장_그리기(숫자, 글꼴들):
    """한 장의 가짜 손글씨 캔버스(280x280 uint8 배열)를 만든다."""
    글꼴 = ImageFont.truetype(str(random.choice(글꼴들)), int(280 * random.uniform(0.40, 0.82)))
    판 = Image.new("L", (280, 280), 255)
    그리기 = ImageDraw.Draw(판)
    왼, 위, 오른, 아래 = 그리기.textbbox((0, 0), str(숫자), font=글꼴)
    그리기.text(((280 - (오른 - 왼)) / 2 - 왼, (280 - (아래 - 위)) / 2 - 위),
               str(숫자), font=글꼴, fill=0)

    기울기 = random.uniform(-0.25, 0.25)
    판 = 판.transform((280, 280), Image.AFFINE, (1, 기울기, -기울기 * 140, 0, 1, 0),
                      resample=Image.BICUBIC, fillcolor=255)
    판 = 판.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, fillcolor=255)

    굵기 = random.choice([-2, -1, 0, 0, 1, 2, 3])
    if 굵기 < 0:
        판 = 판.filter(ImageFilter.MaxFilter(1 - 2 * 굵기))   # 흰 배경 확장 = 획이 가늘어짐
    elif 굵기 > 0:
        판 = 판.filter(ImageFilter.MinFilter(1 + 2 * 굵기))   # 검은 획 확장 = 굵어짐

    배열 = _탄성왜곡(np.array(판, dtype=np.float32),
                     세기=random.uniform(2, 6), 부드러움=random.uniform(4, 9))
    이동판 = Image.new("L", (280, 280), 255)
    이동판.paste(Image.fromarray(배열.astype(np.uint8)),
                 (random.randint(-30, 30), random.randint(-30, 30)))
    return np.array(이동판, dtype=np.uint8)
