"""
바로가기에 쓸 앱 아이콘(app.ico)을 만드는 스크립트.

손글씨 느낌의 글꼴(Segoe Script)로 숫자를 쓴 모양을 파란 사각형 위에 그려서
윈도우 아이콘 파일(.ico)로 저장한다. 한 번만 실행하면 된다.

실행 예)
    python make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

아이콘_파일 = Path(__file__).with_name("app.ico")
기본_크기 = 256                                   # 가장 큰 아이콘 한 변의 픽셀 수
글꼴_경로 = r"C:\Windows\Fonts\segoescb.ttf"      # Segoe Script Bold (손글씨체)
위_색 = (59, 130, 246)                            # 배경 위쪽 파랑 #3b82f6
아래_색 = (29, 78, 216)                           # 배경 아래쪽 파랑 #1d4ed8


def 배경_만들기(크기):
    """위에서 아래로 색이 변하는 둥근 사각형 배경을 만든다."""
    배경 = Image.new("RGB", (크기, 크기))
    그리기 = ImageDraw.Draw(배경)
    for y in range(크기):
        비율 = y / (크기 - 1)
        색 = tuple(int(위 + (아래 - 위) * 비율) for 위, 아래 in zip(위_색, 아래_색))
        그리기.line([(0, y), (크기, y)], fill=색)

    # 모서리를 둥글게 깎기 위한 마스크
    마스크 = Image.new("L", (크기, 크기), 0)
    ImageDraw.Draw(마스크).rounded_rectangle(
        [0, 0, 크기 - 1, 크기 - 1], radius=int(크기 * 0.22), fill=255
    )

    결과 = Image.new("RGBA", (크기, 크기), (0, 0, 0, 0))
    결과.paste(배경, (0, 0), 마스크)
    return 결과


def 숫자_그리기(그림, 글자="3"):
    """손글씨체 숫자를 그림 한가운데에 하얗게 쓴다."""
    크기 = 그림.width
    그리기 = ImageDraw.Draw(그림)
    글꼴 = ImageFont.truetype(글꼴_경로, int(크기 * 0.72))

    왼, 위, 오른, 아래 = 그리기.textbbox((0, 0), 글자, font=글꼴)
    x = (크기 - (오른 - 왼)) / 2 - 왼
    y = (크기 - (아래 - 위)) / 2 - 위
    # 살짝 어두운 그림자를 먼저 깔아 글씨가 또렷해 보이게 한다
    그리기.text((x + 크기 * 0.012, y + 크기 * 0.012), 글자, font=글꼴, fill=(23, 55, 140, 160))
    그리기.text((x, y), 글자, font=글꼴, fill=(255, 255, 255, 255))
    return 그림


def main():
    그림 = 숫자_그리기(배경_만들기(기본_크기))
    # 윈도우가 상황에 맞게 골라 쓰도록 여러 크기를 한 파일에 함께 담는다
    크기들 = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]
    그림.save(아이콘_파일, format="ICO", sizes=크기들)
    print(f"아이콘 저장 완료: {아이콘_파일}")


if __name__ == "__main__":
    main()
