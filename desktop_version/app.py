"""
마우스로 숫자를 직접 써 넣으면 학습된 CNN이 인식해 주는 손글씨 앱.

왼쪽 검은 칸에 마우스를 누른 채 숫자를 그리면 실시간으로 결과가 갱신된다.
실행 전에 train.py를 먼저 실행해 mnist_cnn.pt 파일을 만들어야 한다.

실행 방법)
    - 바탕 화면의 '손글씨 숫자 인식기' 바로가기를 더블클릭 (검은 콘솔 창 없이 실행)
    - 탐색기에서 app.py 파일을 더블클릭
    - 또는 터미널에서  python app.py
"""

import ctypes
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import MnistCNN

캔버스_크기 = 280          # 그림판 한 변의 픽셀 수 (28의 10배)
붓_굵기 = 18               # 붓 두께 (28x28로 줄였을 때 약 2픽셀이 되도록)
가중치_파일 = Path(__file__).with_name("mnist_cnn.pt")
아이콘_파일 = Path(__file__).with_name("app.ico")
# 작업 표시줄이 이 앱을 파이썬이 아닌 독립 프로그램으로 인식하게 하는 고유 이름
앱_아이디 = "MNIST.HandwritingRecognizer.1"

# pythonw.exe(콘솔 없는 실행기)로 실행하면 표준 출력이 없어 print가 실패한다.
# 출력을 버리는 통로로 바꿔 두어 어느 방식으로 실행해도 동작하게 만든다.
콘솔_있음 = sys.stdout is not None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def 작업표시줄_준비():
    """작업 표시줄에서 고유 아이콘으로 표시되고 고정도 되도록 앱 ID를 등록한다."""
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(앱_아이디)
    except (AttributeError, OSError):
        pass  # 윈도우가 아니거나 지원하지 않으면 무시


def 오류_알리기(제목, 내용):
    """콘솔이 없을 때도 오류를 볼 수 있도록 메시지 상자로 알린다."""
    print(f"{제목}: {내용}")
    try:
        임시창 = tk.Tk()
        임시창.withdraw()  # 빈 창은 숨기고 메시지 상자만 띄운다
        messagebox.showerror(제목, 내용)
        임시창.destroy()
    except tk.TclError:
        pass
    if 콘솔_있음:
        창_닫히기_전에_멈춤()


def 창_닫히기_전에_멈춤():
    """더블클릭으로 실행했을 때 오류 메시지를 읽을 수 있도록 입력을 기다린다."""
    try:
        print()
        input("엔터 키를 누르면 창이 닫힙니다...")
    except (EOFError, OSError, AttributeError):
        pass  # 콘솔이 없는 환경에서는 그냥 넘어간다


def 모델_불러오기():
    """저장된 가중치를 읽어 추론 전용 모델을 만든다."""
    if not 가중치_파일.exists():
        오류_알리기(
            "가중치 파일이 없습니다",
            f"{가중치_파일}\n\n먼저 'python train.py' 를 실행해 모델을 학습시켜 주세요.",
        )
        sys.exit(1)

    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    모델 = MnistCNN().to(장치)
    모델.load_state_dict(torch.load(가중치_파일, map_location=장치))
    모델.eval()  # 추론 모드 (드롭아웃 비활성화)
    return 모델, 장치


def 이미지_전처리(그림):
    """
    사용자가 그린 그림을 MNIST 형식(28x28, 흰 글씨/검은 배경)으로 변환한다.

    MNIST는 숫자를 20x20 크기로 맞춘 뒤 무게중심을 28x28의 한가운데에
    놓는 방식으로 만들어졌기 때문에, 같은 방식으로 맞춰 주어야 정확도가 높다.
    """
    배열 = np.array(그림, dtype=np.float32)  # 흰 배경 255, 획 0
    배열 = 255.0 - 배열                      # 색 반전: 배경 0, 획 255 (MNIST와 동일)

    # 획이 있는 영역(값이 있는 부분)만 잘라낸다
    좌표 = np.argwhere(배열 > 20)
    if 좌표.size == 0:
        return None  # 아무것도 그리지 않은 상태

    위, 왼 = 좌표.min(axis=0)
    아래, 오른 = 좌표.max(axis=0)
    잘린배열 = 배열[위:아래 + 1, 왼:오른 + 1]

    # 가로세로 비율을 유지한 채 긴 변이 20픽셀이 되도록 축소
    높이, 너비 = 잘린배열.shape
    비율 = 20.0 / max(높이, 너비)
    새높이 = max(1, int(round(높이 * 비율)))
    새너비 = max(1, int(round(너비 * 비율)))
    축소 = Image.fromarray(잘린배열).resize((새너비, 새높이), Image.LANCZOS)

    # 28x28 검은 도화지 한가운데에 붙인다
    도화지 = Image.new("F", (28, 28), 0.0)
    도화지.paste(축소, ((28 - 새너비) // 2, (28 - 새높이) // 2))
    결과 = np.array(도화지, dtype=np.float32)

    # 픽셀 무게중심이 이미지 중앙(13.5, 13.5)에 오도록 평행 이동
    총합 = 결과.sum()
    if 총합 > 0:
        y좌표, x좌표 = np.indices(결과.shape)
        중심y = (y좌표 * 결과).sum() / 총합
        중심x = (x좌표 * 결과).sum() / 총합
        이동y = int(round(13.5 - 중심y))
        이동x = int(round(13.5 - 중심x))
        결과 = np.roll(결과, (이동y, 이동x), axis=(0, 1))

    결과 = np.clip(결과 / 255.0, 0.0, 1.0)          # 0~1 범위로 정규화
    결과 = (결과 - 0.1307) / 0.3081                 # 학습 때와 동일한 표준화
    return torch.from_numpy(결과).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)


class 손글씨앱:
    """tkinter로 만든 손글씨 입력 및 인식 화면."""

    def __init__(self, 루트, 모델, 장치):
        self.모델 = 모델
        self.장치 = 장치
        self.루트 = 루트
        루트.title("손글씨 숫자 인식기")
        루트.resizable(False, False)
        # 창 왼쪽 위와 작업 표시줄에 앱 아이콘을 표시한다
        if 아이콘_파일.exists():
            try:
                루트.iconbitmap(default=str(아이콘_파일))
            except tk.TclError:
                pass

        바깥틀 = tk.Frame(루트, padx=12, pady=12)
        바깥틀.pack()

        # --- 왼쪽: 그림판 ---
        왼쪽 = tk.Frame(바깥틀)
        왼쪽.grid(row=0, column=0, padx=(0, 12))
        tk.Label(왼쪽, text="여기에 숫자를 쓰세요", font=("맑은 고딕", 11)).pack(pady=(0, 6))
        self.캔버스 = tk.Canvas(
            왼쪽, width=캔버스_크기, height=캔버스_크기,
            bg="white", cursor="crosshair", highlightthickness=1,
            highlightbackground="#888888",
        )
        self.캔버스.pack()

        단추틀 = tk.Frame(왼쪽)
        단추틀.pack(pady=8, fill="x")
        tk.Button(단추틀, text="지우기", width=12, command=self.지우기).pack(side="left")
        tk.Button(단추틀, text="인식하기", width=12, command=self.인식).pack(side="right")

        # --- 오른쪽: 인식 결과 ---
        오른쪽 = tk.Frame(바깥틀)
        오른쪽.grid(row=0, column=1, sticky="n")
        tk.Label(오른쪽, text="인식 결과", font=("맑은 고딕", 11)).pack()
        self.결과표시 = tk.Label(오른쪽, text="?", font=("맑은 고딕", 90, "bold"), fg="#1a56db")
        self.결과표시.pack(pady=(0, 4))
        self.확률표시 = tk.Label(오른쪽, text="확신도 --", font=("맑은 고딕", 11))
        self.확률표시.pack(pady=(0, 8))

        tk.Label(오른쪽, text="숫자별 확률", font=("맑은 고딕", 10)).pack(anchor="w")
        self.막대들 = []
        for 숫자 in range(10):
            줄 = tk.Frame(오른쪽)
            줄.pack(anchor="w", pady=1)
            tk.Label(줄, text=str(숫자), width=2, font=("Consolas", 10)).pack(side="left")
            막대 = tk.Canvas(줄, width=140, height=12, bg="#eeeeee", highlightthickness=0)
            막대.pack(side="left")
            수치 = tk.Label(줄, text="  0%", width=5, font=("Consolas", 9))
            수치.pack(side="left")
            self.막대들.append((막대, 수치))

        # 화면에 보이는 캔버스와 똑같은 그림을 PIL 이미지로도 그려 둔다
        # (화면 캡처 없이 픽셀 값을 바로 읽기 위함)
        self.그림 = Image.new("L", (캔버스_크기, 캔버스_크기), 255)
        self.그리기 = ImageDraw.Draw(self.그림)
        self.직전좌표 = None

        # 마우스 이벤트 연결
        self.캔버스.bind("<Button-1>", self.그리기_시작)
        self.캔버스.bind("<B1-Motion>", self.그리는_중)
        self.캔버스.bind("<ButtonRelease-1>", self.그리기_끝)

    def 그리기_시작(self, 이벤트):
        self.직전좌표 = (이벤트.x, 이벤트.y)
        self.점_찍기(이벤트.x, 이벤트.y)

    def 그리는_중(self, 이벤트):
        if self.직전좌표 is None:
            self.직전좌표 = (이벤트.x, 이벤트.y)
        x0, y0 = self.직전좌표
        x1, y1 = 이벤트.x, 이벤트.y
        # 화면용 선
        self.캔버스.create_line(
            x0, y0, x1, y1, width=붓_굵기, fill="black",
            capstyle=tk.ROUND, smooth=True,
        )
        # 내부 이미지용 선 (같은 좌표, 같은 두께)
        self.그리기.line([x0, y0, x1, y1], fill=0, width=붓_굵기)
        self.점_찍기(x1, y1)  # 선을 이을 때 생기는 모서리를 둥글게 메운다
        self.직전좌표 = (x1, y1)

    def 그리기_끝(self, _이벤트):
        self.직전좌표 = None
        self.인식()  # 획을 마칠 때마다 자동으로 인식

    def 점_찍기(self, x, y):
        """붓 굵기만큼의 둥근 점을 화면과 내부 이미지에 동시에 찍는다."""
        반지름 = 붓_굵기 / 2
        self.캔버스.create_oval(x - 반지름, y - 반지름, x + 반지름, y + 반지름,
                                fill="black", outline="black")
        self.그리기.ellipse([x - 반지름, y - 반지름, x + 반지름, y + 반지름], fill=0)

    def 지우기(self):
        """그림판과 결과를 초기 상태로 되돌린다."""
        self.캔버스.delete("all")
        self.그리기.rectangle([0, 0, 캔버스_크기, 캔버스_크기], fill=255)
        self.결과표시.config(text="?")
        self.확률표시.config(text="확신도 --")
        for 막대, 수치 in self.막대들:
            막대.delete("all")
            수치.config(text="  0%")

    def 인식(self):
        """현재 그림을 모델에 넣어 숫자를 예측하고 화면을 갱신한다."""
        입력 = 이미지_전처리(self.그림)
        if 입력 is None:
            return  # 빈 화면이면 아무것도 하지 않음

        with torch.no_grad():  # 추론이므로 기울기 계산 불필요
            출력 = self.모델(입력.to(self.장치))
            확률 = F.softmax(출력, dim=1)[0].cpu().numpy()

        예측숫자 = int(확률.argmax())
        self.결과표시.config(text=str(예측숫자))
        self.확률표시.config(text=f"확신도 {확률[예측숫자] * 100:.1f}%")

        for 숫자, (막대, 수치) in enumerate(self.막대들):
            막대.delete("all")
            길이 = int(확률[숫자] * 140)
            색 = "#1a56db" if 숫자 == 예측숫자 else "#9db5e8"
            if 길이 > 0:
                막대.create_rectangle(0, 0, 길이, 12, fill=색, outline=색)
            수치.config(text=f"{확률[숫자] * 100:3.0f}%")


def main():
    작업표시줄_준비()
    모델, 장치 = 모델_불러오기()
    print(f"모델을 불러왔습니다. (사용 장치: {장치})")
    루트 = tk.Tk()
    손글씨앱(루트, 모델, 장치)
    루트.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 콘솔이 없어도 오류 내용을 확인할 수 있도록 메시지 상자로 알린다
        import traceback

        오류_알리기("실행 중 오류가 발생했습니다", traceback.format_exc())
        sys.exit(1)
