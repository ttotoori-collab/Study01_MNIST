# desktop_version — PyTorch + tkinter 손글씨 앱

## 이 폴더의 역할

마우스로 숫자를 그리면 CNN이 인식하는 윈도우 데스크톱 앱이다.
`web_version/` 과는 독립적이며 코드를 공유하지 않는다.

## 규칙

- 모든 코드와 주석, 변수명은 한글로 쓴다.
- 파이썬 실행 경로는 `C:\Users\somes\AppData\Local\Programs\Python\Python313\python.exe` 이다.
  `python` 만 치면 Microsoft Store 스텁이 잡히므로 전체 경로를 쓴다.
- 모델 구조를 바꾸면 `train.py` 를 다시 돌려 `mnist_cnn.pt` 를 갱신해야 한다.

## 명령

| 목적 | 명령 |
| --- | --- |
| 학습 | `python train.py --epochs 5` |
| 실행 | `python app.py` 또는 바탕 화면 바로가기 |
| 아이콘 재생성 | `python make_icon.py` |

## 주의

- `app.py` 는 `pythonw.exe` 로도 실행된다. 이때 `sys.stdout` 이 `None` 이므로
  `print` 를 직접 쓰기 전에 파일 상단의 표준 출력 가드를 확인한다.
- 바탕 화면 바로가기는 이 폴더의 절대경로를 담고 있다. 폴더를 옮기면 바로가기도 고쳐야 한다.
- `data/` 는 64MB라 git에서 제외되어 있다. 없으면 `train.py` 가 자동으로 내려받는다.
