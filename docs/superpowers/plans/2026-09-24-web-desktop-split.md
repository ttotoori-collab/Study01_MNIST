# 손글씨 숫자 인식기 웹/데스크톱 분리 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 MNIST 인식 코드를 `desktop_version/`으로 옮기고, 외부 라이브러리 없이 순수 자바스크립트로 추론하는 `web_version/`을 새로 만들어 GitHub Pages에 배포한다.

**Architecture:** PyTorch로 학습한 경량 CNN(105,866 파라미터)의 가중치를 float32 바이너리로 내보내고, 브라우저는 이를 `fetch`로 읽어 `Float32Array` 위에서 직접 순전파를 계산한다. 전처리는 데스크톱과 같은 알고리즘을 JS로 이식하며, 두 구현의 동등성을 node 테스트로 못 박는다. 빌드 단계는 없고 ES 모듈을 그대로 서빙한다.

**Tech Stack:** Python 3.13 + PyTorch 2.14 (오프라인 학습/내보내기), 순수 ES 모듈 자바스크립트 (런타임), node v24 (테스트), GitHub Actions + Pages (배포)

**Spec:** `docs/superpowers/specs/2026-09-24-web-desktop-split-design.md`

## Global Constraints

- 모든 코드와 주석, 변수명은 한글로 작성한다.
- 웹 런타임의 외부 의존성은 0개다. CDN, npm 패키지, 빌드 단계 모두 금지한다.
- 웹의 모든 경로는 상대경로로 쓴다. 사이트가 하위 경로에 배포될 수 있다.
- 파이썬 실행 경로는 `C:\Users\somes\AppData\Local\Programs\Python\Python313\python.exe` 이며, 이 계획에서는 `$PY` 로 표기한다.
- node는 v24, ES 모듈(`.mjs`)로 테스트를 작성한다.
- `web_version/model/weights.bin` 은 500KB 이하여야 한다.
- 획을 뗀 뒤 화면 갱신까지 50ms 이내여야 한다.
- 웹 모델의 MNIST 테스트 정확도는 99.0% 이상, 가혹 손글씨 시뮬레이션 정확도는 98.5% 이상이어야 한다.
- JS 추론과 PyTorch 로짓의 최대 오차는 1e-4 이하, 전처리 결과 오차는 1e-3 이하여야 한다.
- 모든 git 커밋 메시지는 다음 줄로 끝낸다.
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Review Focus

스펙이 전제하지만 어느 작업의 테스트도 직접 건드리지 않는 입력들이다. 각 항목은 해당 코드를 소유한 작업에 테스트로 박아 넣었다.

1. **빈 캔버스에서 인식 요청** — 아무 획도 없으면 경계 상자가 없다. 예외를 던지지 말고 조용히 아무것도 하지 않아야 한다. (Task 5)
2. **점 하나만 찍은 입력** — 경계 상자가 1x1이 되어 축소 비율 계산에서 0 크기가 나올 수 있다. 최소 1px을 보장해야 한다. (Task 5)
3. **캔버스 가장자리에 붙여 그린 획** — 무게중심 정렬을 `np.roll` 식 순환 이동으로 구현하면 획이 반대편 모서리로 넘어가 숫자가 망가진다. 잘린 부분은 0으로 채워야 한다. (Task 5)
4. **`file://` 로 직접 열기** — `fetch`가 차단되어 가중치 로드가 실패한다. 빈 화면 대신 원인과 해결 명령을 안내해야 한다. (Task 6)
5. **터치 기기에서 그리기** — 기본 동작을 막지 않으면 그리는 동안 페이지가 스크롤되어 획이 끊긴다. (Task 7)

---

## File Structure

| 경로 | 책임 |
| --- | --- |
| `.gitignore` | 데이터셋, 캐시, 학습 가중치 제외 |
| `CLAUDE.md` | 두 폴더의 역할과 공통 규칙 |
| `README.md` | 프로젝트 소개와 두 버전 링크 |
| `.github/workflows/pages.yml` | `web_version/` 정적 배포 |
| `desktop_version/**` | 기존 PyTorch + tkinter 앱 (이동만) |
| `web_version/tools/web_model.py` | 웹 전용 경량 CNN 정의 |
| `web_version/tools/손글씨_생성.py` | 글꼴로 손글씨 숫자를 그리는 공용 모듈 (학습/벤치 글꼴 분리) |
| `web_version/tools/train_web.py` | 증강 학습 + 가중치 저장 |
| `web_version/tools/export_weights.py` | `.pt` → `weights.bin` + `weights.json` |
| `web_version/tools/preprocess_ref.py` | 전처리 기준 구현 (파이썬) |
| `web_version/tools/make_fixtures.py` | 테스트 fixture 생성 |
| `web_version/tools/bench.py` | 가혹 손글씨 시뮬레이션 벤치마크 |
| `web_version/js/nn.js` | 텐서 연산 (conv2d, relu, maxPool2d, linear, softmax) |
| `web_version/js/preprocess.js` | 캔버스 픽셀 → 28x28 배열 |
| `web_version/js/model.js` | 가중치 로드 + 순전파 + TTA |
| `web_version/js/app.js` | 캔버스 입력과 화면 갱신 |
| `web_version/index.html` `style.css` | 화면 구조와 스타일 |
| `web_version/tests/*.mjs` | node 동등성 테스트 |

---

## Task 1: 저장소 초기화와 desktop_version 이동

**Files:**
- Create: `.gitignore`, `CLAUDE.md`, `README.md`, `desktop_version/CLAUDE.md`
- Move: `model.py` `train.py` `app.py` `make_icon.py` `mnist_cnn.pt` `app.ico` `data/` `README.md` → `desktop_version/`
- Modify: 바탕 화면 바로가기 `손글씨 숫자 인식기.lnk`

**Interfaces:**
- Consumes: 없음
- Produces: `desktop_version/` 경로 아래에서 동작하는 기존 앱. 이후 작업은 이 폴더를 건드리지 않는다.

- [ ] **Step 1: git 저장소 초기화**

```bash
cd "D:/DSMP/3장 손글씨 앱/study01_MNIST"
git init -b main
git config user.name "somes"
git config user.email "ttotoori@gmail.com"
```

- [ ] **Step 2: .gitignore 작성**

```gitignore
# MNIST 원본 데이터 (64MB, 실행 시 자동 내려받음)
desktop_version/data/
web_version/tools/data/

# 파이썬 캐시
__pycache__/
*.pyc

# 학습 결과 가중치 (웹 배포용 weights.bin 은 제외하지 않는다)
*.pt
```

- [ ] **Step 3: 파일 이동**

```bash
mkdir -p desktop_version
git mv 2>/dev/null || true
mv model.py train.py app.py make_icon.py mnist_cnn.pt app.ico README.md data desktop_version/
rm -rf __pycache__
ls desktop_version
```

- [ ] **Step 4: 이동 후 데스크톱 앱이 동작하는지 확인**

Run:
```bash
cd desktop_version && $PY -c "import app; print('임포트 성공')"
```
Expected: `임포트 성공` (tkinter 창은 뜨지 않는다)

- [ ] **Step 5: 바탕 화면 바로가기 경로 갱신**

```powershell
$프로젝트 = "D:\DSMP\3장 손글씨 앱\study01_MNIST\desktop_version"
$바로가기 = Join-Path ([Environment]::GetFolderPath('Desktop')) "손글씨 숫자 인식기.lnk"
$쉘 = New-Object -ComObject WScript.Shell
$링크 = $쉘.CreateShortcut($바로가기)
$링크.Arguments        = '"' + $프로젝트 + '\app.py"'
$링크.WorkingDirectory = $프로젝트
$링크.IconLocation     = "$프로젝트\app.ico,0"
$링크.Save()
$확인 = $쉘.CreateShortcut($바로가기)
Write-Output $확인.Arguments $확인.WorkingDirectory $확인.IconLocation
```
Expected: 세 값 모두 `desktop_version` 경로를 가리킨다.

- [ ] **Step 6: 바로가기로 앱이 실제로 뜨는지 확인**

```powershell
Start-Process (Join-Path ([Environment]::GetFolderPath('Desktop')) "손글씨 숫자 인식기.lnk")
Start-Sleep -Seconds 15
Get-Process | Where-Object { $_.ProcessName -eq "pythonw" } | Select-Object Id, MainWindowTitle
```
Expected: `MainWindowTitle` 이 `손글씨 숫자 인식기`. 확인 후 `Stop-Process` 로 닫는다.

- [ ] **Step 7: desktop_version/CLAUDE.md 작성**

```markdown
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
```

- [ ] **Step 8: 루트 CLAUDE.md 작성**

```markdown
# 손글씨 숫자 인식기

## 폴더 구성

| 폴더 | 내용 |
| --- | --- |
| `desktop_version/` | PyTorch + tkinter 윈도우 앱 |
| `web_version/` | 순수 자바스크립트 웹 앱 (GitHub Pages 배포) |

작업을 시작하기 전에 어느 쪽 이야기인지 먼저 확인하고, 해당 폴더의 CLAUDE.md를 읽는다.
두 버전은 코드를 공유하지 않는다. 한쪽을 고쳤다고 다른 쪽이 따라 바뀌지 않는다.

## 공통 규칙

- 모든 코드와 주석, 변수명은 한글로 쓴다.
- 전처리 알고리즘은 두 버전이 같아야 한다. 한쪽을 바꾸면
  `web_version/tests/` 의 동등성 테스트가 깨지므로 양쪽을 함께 고친다.

## 설계 문서

- 스펙: `docs/superpowers/specs/2026-09-24-web-desktop-split-design.md`
- 계획: `docs/superpowers/plans/2026-09-24-web-desktop-split.md`
```

- [ ] **Step 9: 루트 README.md 작성**

```markdown
# 손글씨 숫자 인식기

마우스나 손가락으로 숫자를 그리면 CNN이 0~9 중 무엇인지 알아맞히는 프로그램이다.
같은 문제를 두 가지 방식으로 구현했다.

| 버전 | 실행 환경 | 모델 | 가중치 크기 |
| --- | --- | --- | --- |
| [desktop_version](desktop_version/) | 윈도우 + PyTorch + tkinter | Conv32-Conv64-FC128 | 4.8MB |
| [web_version](web_version/) | 브라우저 (순수 JS) | Conv16-Conv32-FC64 | 414KB |

웹 버전은 외부 라이브러리 없이 자바스크립트로 직접 순전파를 계산하며,
GitHub Pages에 정적으로 배포된다.
```

- [ ] **Step 10: 커밋**

```bash
git add -A
git commit -m "chore: desktop_version 폴더로 기존 코드 이동 및 저장소 초기화

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: 웹 전용 모델 정의와 전처리 기준 구현

**Files:**
- Create: `web_version/tools/web_model.py`, `web_version/tools/preprocess_ref.py`, `web_version/tools/test_tools.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `web_model.웹CNN` — `nn.Module`, 입력 `(N,1,28,28)`, 출력 `(N,10)` 로그 확률
  - `preprocess_ref.전처리(회색배열: np.ndarray[uint8, (H,W)]) -> np.ndarray[float32, (28,28)] | None`
    반환값은 **0~1 범위**이며 정규화는 하지 않는다
  - `preprocess_ref.정규화(배열) -> np.ndarray[float32]` — `(x - 0.1307) / 0.3081`

파이썬 테스트는 표준 라이브러리 `unittest` 로 작성한다. pytest는 설치되어 있지 않고, 설치할 이유도 없다.

- [ ] **Step 1: 실패하는 테스트 작성**

`web_version/tools/test_tools.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run:
```bash
cd web_version/tools && $PY -m unittest test_tools -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocess_ref'`

- [ ] **Step 3: 웹 모델 구현**

`web_version/tools/web_model.py`:

```python
"""
웹 버전이 쓰는 경량 CNN 정의.

데스크톱 모델은 완전연결 층 하나가 파라미터의 98%를 차지해 4.8MB가 된다.
여기서는 풀링을 한 번 더 넣어 완전연결 입력을 9216에서 1568로 줄였다.
파라미터 105,866개, float32로 약 414KB다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class 웹CNN(nn.Module):
    """28x28 흑백 손글씨를 0~9로 분류하는 경량 합성곱 신경망."""

    def __init__(self):
        super().__init__()
        # padding=1 을 주어 합성곱 뒤에도 가로세로 크기가 유지되게 한다
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """순전파. 입력 x의 형태는 (배치, 1, 28, 28)."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # (N, 16, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # (N, 32, 7, 7)
        x = torch.flatten(x, 1)                       # (N, 1568)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)
```

- [ ] **Step 4: 전처리 기준 구현**

`web_version/tools/preprocess_ref.py`:

```python
"""
전처리 기준 구현. 자바스크립트 이식(js/preprocess.js)의 정답지 역할을 한다.

데스크톱 앱과 같은 알고리즘이지만 두 가지가 다르다.
1. 축소 필터로 PIL의 LANCZOS 대신 면적 평균을 쓴다. 브라우저 캔버스의 보간은
   구현마다 결과가 달라 재현이 되지 않기 때문에, 양쪽에서 똑같이 직접 계산한다.
2. 정규화를 여기서 하지 않고 0~1 값을 돌려준다. 추론 시 TTA로 1픽셀씩 밀어 볼 때
   배경이 0이어야 빈 자리를 0으로 채울 수 있기 때문이다.
"""

import numpy as np

임계값 = 20          # 이 값을 넘는 픽셀만 획으로 본다
목표_긴변 = 20       # MNIST와 같이 숫자의 긴 변을 20픽셀로 맞춘다
출력_크기 = 28
평균 = 0.1307        # 학습에 쓴 MNIST 전체 평균
표준편차 = 0.3081


def 정규화(배열):
    """학습 때와 같은 표준화를 적용한다."""
    return ((배열 - 평균) / 표준편차).astype(np.float32)


def _반올림(값):
    """
    0.5를 항상 위로 올린다.
    파이썬 기본 round는 0.5에서 짝수로 내리지만(은행가 반올림) 자바스크립트
    Math.round는 위로 올린다. 두 구현의 결과를 맞추려면 이쪽을 따라야 한다.
    """
    return int(np.floor(값 + 0.5))


def _면적평균_축소(배열, 새높이, 새너비):
    """
    출력 픽셀 하나가 덮는 원본 영역의 평균을 낸다.
    가장자리에서 걸치는 칸은 겹치는 길이만큼만 반영한다.
    자바스크립트도 똑같은 식으로 계산해야 결과가 일치한다.
    """
    높이, 너비 = 배열.shape
    비율x = 너비 / 새너비
    가로 = np.zeros((높이, 새너비), dtype=np.float64)
    for x in range(새너비):
        시작, 끝 = x * 비율x, (x + 1) * 비율x
        for fx in range(int(np.floor(시작)), min(int(np.ceil(끝)), 너비)):
            겹침 = min(끝, fx + 1) - max(시작, fx)
            if 겹침 > 0:
                가로[:, x] += 배열[:, fx] * 겹침
        가로[:, x] /= 비율x

    비율y = 높이 / 새높이
    결과 = np.zeros((새높이, 새너비), dtype=np.float64)
    for y in range(새높이):
        시작, 끝 = y * 비율y, (y + 1) * 비율y
        for fy in range(int(np.floor(시작)), min(int(np.ceil(끝)), 높이)):
            겹침 = min(끝, fy + 1) - max(시작, fy)
            if 겹침 > 0:
                결과[y] += 가로[fy] * 겹침
        결과[y] /= 비율y
    return 결과


def _평행이동(배열, 이동y, 이동x):
    """
    잘려 나간 자리를 0으로 채우며 평행 이동한다.
    순환 이동을 쓰면 가장자리에 그린 획이 반대편 모서리로 넘어가 숫자가 망가진다.
    """
    높이, 너비 = 배열.shape
    결과 = np.zeros_like(배열)
    원본_y0, 대상_y0 = (0, 이동y) if 이동y >= 0 else (-이동y, 0)
    원본_x0, 대상_x0 = (0, 이동x) if 이동x >= 0 else (-이동x, 0)
    높이수 = 높이 - abs(이동y)
    너비수 = 너비 - abs(이동x)
    if 높이수 > 0 and 너비수 > 0:
        결과[대상_y0:대상_y0 + 높이수, 대상_x0:대상_x0 + 너비수] = \
            배열[원본_y0:원본_y0 + 높이수, 원본_x0:원본_x0 + 너비수]
    return 결과


def 전처리(회색배열):
    """
    흰 배경(255) 위에 검은 획(0)으로 그려진 2차원 uint8 배열을 받아
    0~1 범위의 28x28 float32 배열을 돌려준다. 획이 하나도 없으면 None.
    """
    배열 = 255.0 - 회색배열.astype(np.float64)   # 반전: 배경 0, 획 255

    좌표 = np.argwhere(배열 > 임계값)
    if 좌표.size == 0:
        return None                               # 빈 캔버스

    위, 왼 = 좌표.min(axis=0)
    아래, 오른 = 좌표.max(axis=0)
    잘린 = 배열[위:아래 + 1, 왼:오른 + 1]

    높이, 너비 = 잘린.shape
    비율 = 목표_긴변 / max(높이, 너비)
    새높이 = max(1, _반올림(높이 * 비율))          # 점 하나여도 최소 1픽셀
    새너비 = max(1, _반올림(너비 * 비율))
    축소 = _면적평균_축소(잘린, 새높이, 새너비)

    도화지 = np.zeros((출력_크기, 출력_크기), dtype=np.float64)
    y0 = (출력_크기 - 새높이) // 2
    x0 = (출력_크기 - 새너비) // 2
    도화지[y0:y0 + 새높이, x0:x0 + 새너비] = 축소

    총합 = 도화지.sum()
    if 총합 > 0:
        y좌표, x좌표 = np.indices(도화지.shape)
        이동y = _반올림((출력_크기 - 1) / 2 - (y좌표 * 도화지).sum() / 총합)
        이동x = _반올림((출력_크기 - 1) / 2 - (x좌표 * 도화지).sum() / 총합)
        도화지 = _평행이동(도화지, 이동y, 이동x)

    return np.clip(도화지 / 255.0, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 5: 테스트가 통과하는지 확인**

Run:
```bash
cd web_version/tools && $PY -m unittest test_tools -v
```
Expected: 11개 테스트 모두 PASS

- [ ] **Step 6: 커밋**

```bash
git add web_version/tools/web_model.py web_version/tools/preprocess_ref.py web_version/tools/test_tools.py
git commit -m "feat: 웹 전용 경량 CNN과 전처리 기준 구현 추가

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: 증강 학습과 정확도 게이트

**Files:**
- Create: `web_version/tools/train_web.py`, `web_version/tools/bench.py`
- Produces (git 제외): `web_version/tools/web_mnist_cnn.pt`

**Interfaces:**
- Consumes: `web_model.웹CNN`, `preprocess_ref.전처리`, `preprocess_ref.정규화`
- Produces:
  - `web_mnist_cnn.pt` — `웹CNN` 의 `state_dict`. Task 4가 이 파일을 읽는다.
  - `bench.측정(모델, 장치, 표본수=400) -> (맞은개수, 전체개수, 혼동표)`

이 작업은 **정확도 게이트**다. 기준을 넘지 못하면 다음 작업으로 넘어가지 않는다.

- [ ] **Step 1: 학습 스크립트 작성**

`web_version/tools/train_web.py`:

```python
"""
웹 버전용 경량 CNN을 학습시키고 web_mnist_cnn.pt 로 저장한다.

사람이 마우스로 쓴 글씨는 MNIST보다 기울기와 크기 변화가 크다. 그래서 학습할 때
어파인 증강(회전/이동/확대축소/전단)을 걸어 그 변형을 미리 겪게 한다.

실행 예)
    python train_web.py --epochs 15
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from web_model import 웹CNN

가중치_파일 = Path(__file__).with_name("web_mnist_cnn.pt")
데이터_경로 = Path(__file__).with_name("data")


def 데이터로더_만들기(배치크기):
    """학습용에는 증강을, 테스트용에는 정규화만 적용한다."""
    학습_전처리 = transforms.Compose([
        transforms.RandomAffine(
            degrees=12,                  # 삐뚤게 쓴 글씨
            translate=(0.12, 0.12),      # 가운데에서 벗어난 글씨
            scale=(0.85, 1.15),          # 크게 또는 작게 쓴 글씨
            shear=8,                     # 기울여 쓴 글씨
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    테스트_전처리 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    학습셋 = datasets.MNIST(데이터_경로, train=True, download=True, transform=학습_전처리)
    테스트셋 = datasets.MNIST(데이터_경로, train=False, download=True, transform=테스트_전처리)
    return (
        torch.utils.data.DataLoader(학습셋, batch_size=배치크기, shuffle=True),
        torch.utils.data.DataLoader(테스트셋, batch_size=1000, shuffle=False),
    )


def 평가(모델, 장치, 데이터로더):
    """테스트 정확도를 백분율로 돌려준다."""
    모델.eval()
    맞음 = 0
    with torch.no_grad():
        for 이미지, 정답 in 데이터로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            맞음 += 모델(이미지).argmax(dim=1).eq(정답).sum().item()
    return 100.0 * 맞음 / len(데이터로더.dataset)


def main():
    파서 = argparse.ArgumentParser(description="웹 버전 경량 CNN 학습")
    파서.add_argument("--epochs", type=int, default=15)
    파서.add_argument("--batch-size", type=int, default=128)
    파서.add_argument("--lr", type=float, default=1.0)
    파서.add_argument("--seed", type=int, default=1)
    인자 = 파서.parse_args()

    torch.manual_seed(인자.seed)
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {장치}")

    학습로더, 테스트로더 = 데이터로더_만들기(인자.batch_size)
    모델 = 웹CNN().to(장치)
    옵티마이저 = optim.Adadelta(모델.parameters(), lr=인자.lr)
    스케줄러 = StepLR(옵티마이저, step_size=3, gamma=0.7)

    최고정확도 = 0.0
    시작 = time.time()
    for 에폭 in range(1, 인자.epochs + 1):
        모델.train()
        for 이미지, 정답 in 학습로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            옵티마이저.zero_grad()
            손실 = F.nll_loss(모델(이미지), 정답)
            손실.backward()
            옵티마이저.step()

        정확도 = 평가(모델, 장치, 테스트로더)
        표시 = ""
        if 정확도 > 최고정확도:
            # 가장 좋았던 시점의 가중치만 남긴다
            최고정확도 = 정확도
            torch.save(모델.state_dict(), 가중치_파일)
            표시 = "  <- 저장"
        print(f"에폭 {에폭:2d}/{인자.epochs}  테스트 정확도 {정확도:.2f}%{표시}")
        스케줄러.step()

    print(f"최고 정확도 {최고정확도:.2f}% / 소요 {time.time() - 시작:.0f}초")
    print(f"가중치 저장 완료: {가중치_파일}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 학습 실행**

Run:
```bash
cd web_version/tools && $PY train_web.py --epochs 15
```
Expected: 마지막 줄의 최고 정확도가 **99.0% 이상**. 미달이면 `--epochs 25` 로 다시 돌린다.
CPU 기준 15에폭에 15~20분이 걸리므로 백그라운드로 실행한다.

- [ ] **Step 3: 가혹 벤치마크 작성**

`web_version/tools/bench.py`:

```python
"""
채점자가 캔버스에 직접 그리는 상황을 흉내 내어 정확도를 잰다.

손글씨 계열 글꼴로 0~9를 쓰고 손떨림(탄성 왜곡), 획 굵기 변화, 기울기, 회전,
위치 이동을 더한 뒤 실제 전처리를 거쳐 모델에 넣는다.

실행 예)
    python bench.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from preprocess_ref import 전처리, 정규화
from web_model import 웹CNN

가중치_파일 = Path(__file__).with_name("web_mnist_cnn.pt")
목표_정확도 = 98.5
글꼴_폴더 = Path(r"C:\Windows\Fonts")
글꼴들 = [
    글꼴_폴더 / "Inkfree.ttf",
    글꼴_폴더 / "comic.ttf",
    글꼴_폴더 / "comicbd.ttf",
    글꼴_폴더 / "segoepr.ttf",
    글꼴_폴더 / "segoeprb.ttf",
    글꼴_폴더 / "segoesc.ttf",
    글꼴_폴더 / "segoescb.ttf",
    글꼴_폴더 / "calibri.ttf",
    글꼴_폴더 / "georgia.ttf",
]


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


def 한장_그리기(숫자):
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
```

- [ ] **Step 4: 벤치마크 실행 (게이트)**

Run:
```bash
cd web_version/tools && $PY bench.py
```
Expected: `목표 달성` 과 종료 코드 0. 정확도 **98.5% 이상**.

미달이면 순서대로 시도한다.

1. `train_web.py --epochs 25` 로 더 학습시킨다.
2. 그래도 모자라면 `RandomAffine` 의 `degrees` 를 15, `shear` 를 12로 올려 다시 학습한다.
3. 혼동표가 특정 쌍(예: 1과 2)에 몰려 있으면 그 원인을 먼저 확인한다.

Task 7에서 TTA를 더하면 실사용 정확도가 한 번 더 올라가지만, 이 게이트는 TTA 없이 통과해야 한다.

- [ ] **Step 5: 커밋**

```bash
git add web_version/tools/train_web.py web_version/tools/bench.py
git commit -m "feat: 증강 학습 스크립트와 가혹 손글씨 벤치마크 추가

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: 가중치 내보내기와 테스트 fixture 생성

**Files:**
- Create: `web_version/tools/export_weights.py`, `web_version/tools/make_fixtures.py`
- Modify: `web_version/tools/test_tools.py` (내보내기 테스트 추가)
- Produces: `web_version/model/weights.bin`, `web_version/model/weights.json`, `web_version/tests/fixtures.json`

**Interfaces:**
- Consumes: `web_mnist_cnn.pt`, `web_model.웹CNN`, `preprocess_ref.전처리`, `preprocess_ref.정규화`
- Produces:
  - `export_weights.레이어_순서` — 문자열 8개 리스트. `weights.bin` 의 배치 순서와 같다.
  - `export_weights.내보내기(가중치파일, 출력폴더) -> dict` — 저장한 `weights.json` 내용
  - `export_weights.넘파이_순전파(가중치사전, 입력배열) -> np.ndarray(10,)` — 로짓. JS 구현의 기준값
  - `tests/fixtures.json` — 아래 형식. Task 6과 Task 7의 node 테스트가 읽는다.

```json
{
  "전처리": [
    { "폭": 280, "높이": 280, "rle": [255, 1200, 0, 30], "기대": [784개의 0~1 실수] }
  ],
  "추론": [
    { "입력": [784개의 0~1 실수], "기대로짓": [10개 실수], "정답": 7 }
  ]
}
```

`rle` 는 캔버스 회색 픽셀을 행 우선으로 훑으며 `[값, 반복횟수, 값, 반복횟수, ...]` 로
적은 것이다. 배경이 대부분이라 280x280 한 장이 수 KB로 줄어든다.

- [ ] **Step 1: 실패하는 테스트 추가**

`web_version/tools/test_tools.py` 끝에 다음을 추가한다.

```python
import tempfile

from export_weights import 넘파이_순전파, 내보내기, 레이어_순서


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
        입력 = np.random.RandomState(0).rand(28, 28).astype(np.float32)
        기대 = self.모델(torch.from_numpy(입력).unsqueeze(0).unsqueeze(0))
        기대로짓 = 기대.detach().numpy()[0]
        실제 = 넘파이_순전파(
            {층["이름"]: 층 for 층 in self.메타["레이어"]},
            입력,
            (self.출력폴더 / "weights.bin").read_bytes(),
        )
        self.assertLess(float(np.abs(실제 - 기대로짓).max()), 1e-4)
```

파일 상단 import 에 `from pathlib import Path` 를 추가한다.

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run:
```bash
cd web_version/tools && $PY -m unittest test_tools -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'export_weights'`

- [ ] **Step 3: 내보내기 구현**

`web_version/tools/export_weights.py`:

```python
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
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run:
```bash
cd web_version/tools && $PY -m unittest test_tools -v
```
Expected: 15개 테스트 모두 PASS

- [ ] **Step 5: 실제 가중치 내보내기**

Run:
```bash
cd web_version/tools && $PY export_weights.py
```
Expected: `파라미터 105,866개`, `weights.bin 423,464 바이트 (414KB)`

- [ ] **Step 6: fixture 생성 스크립트 작성**

`web_version/tools/make_fixtures.py`:

```python
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
import torch
from torchvision import datasets

from bench import 한장_그리기
from export_weights import 넘파이_순전파, 레이어_순서
from preprocess_ref import 전처리, 정규화
from web_model import 웹CNN

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
```

- [ ] **Step 7: fixture 생성**

Run:
```bash
cd web_version/tools && $PY make_fixtures.py
```
Expected: `전처리 6건 / 추론 20건`, 파일 크기 1MB 미만

- [ ] **Step 8: 커밋**

```bash
git add web_version/tools/export_weights.py web_version/tools/make_fixtures.py web_version/tools/test_tools.py web_version/model web_version/tests/fixtures.json
git commit -m "feat: 가중치 내보내기와 JS 검증용 fixture 생성

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: 순수 JS 텐서 연산 (nn.js)

**Files:**
- Create: `web_version/js/nn.js`, `web_version/tests/test_nn.mjs`

**Interfaces:**
- Consumes: 없음
- Produces (모두 `js/nn.js` 에서 `export`):
  - `텐서(데이터: Float32Array, 형태: [채널, 높이, 너비]) -> {데이터, 형태}`
  - `conv2d(입력: 텐서, 가중치: Float32Array, 편향: Float32Array, 출력채널수: number, 커널크기: number, 패딩: number) -> 텐서`
  - `relu(입력: 텐서) -> 텐서` (제자리 수정 후 같은 객체 반환)
  - `maxPool2d(입력: 텐서, 크기: number) -> 텐서`
  - `linear(입력: Float32Array, 가중치: Float32Array, 편향: Float32Array, 출력수: number) -> Float32Array`
  - `softmax(로짓: Float32Array) -> Float32Array`

가중치 배치는 PyTorch와 같다. 합성곱은 `[출력채널][입력채널][커널y][커널x]`,
완전연결은 `[출력][입력]` 행 우선이다.

테스트는 node 내장 `node:test` 와 `node:assert` 만 쓴다. 외부 패키지를 설치하지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`web_version/tests/test_nn.mjs`:

```javascript
// nn.js 의 텐서 연산이 손으로 계산한 값과 일치하는지 확인한다.
import { test } from "node:test";
import assert from "node:assert/strict";

import { conv2d, linear, maxPool2d, relu, softmax, 텐서 } from "../js/nn.js";

const 근사같음 = (실제, 기대, 허용 = 1e-6) => {
  assert.equal(실제.length, 기대.length, "길이가 다르다");
  for (let i = 0; i < 기대.length; i++) {
    assert.ok(Math.abs(실제[i] - 기대[i]) < 허용,
      `${i}번째: 기대 ${기대[i]}, 실제 ${실제[i]}`);
  }
};

test("conv2d: 패딩 없이 3x3 입력에 3x3 커널이면 1x1 이 나온다", () => {
  const 입력 = 텐서(new Float32Array(9).fill(1), [1, 3, 3]);
  const 가중치 = new Float32Array(9).fill(1);
  const 결과 = conv2d(입력, 가중치, new Float32Array([0]), 1, 3, 0);
  assert.deepEqual(결과.형태, [1, 1, 1]);
  근사같음(결과.데이터, [9]);
});

test("conv2d: 패딩 1이면 크기가 유지되고 모서리는 커널의 일부만 덮는다", () => {
  const 입력 = 텐서(new Float32Array(9).fill(1), [1, 3, 3]);
  const 가중치 = new Float32Array(9).fill(1);
  const 결과 = conv2d(입력, 가중치, new Float32Array([0]), 1, 3, 1);
  assert.deepEqual(결과.형태, [1, 3, 3]);
  // 모서리 4칸, 변 6칸, 가운데 9칸
  근사같음(결과.데이터, [4, 6, 4, 6, 9, 6, 4, 6, 4]);
});

test("conv2d: 편향이 모든 출력에 더해진다", () => {
  const 입력 = 텐서(new Float32Array(9).fill(0), [1, 3, 3]);
  const 결과 = conv2d(입력, new Float32Array(9).fill(1), new Float32Array([2.5]), 1, 3, 0);
  근사같음(결과.데이터, [2.5]);
});

test("conv2d: 입력 채널이 여러 개면 모두 더해진다", () => {
  const 입력 = 텐서(new Float32Array([1, 1, 1, 1, 2, 2, 2, 2]), [2, 2, 2]);
  const 가중치 = new Float32Array(8).fill(1);   // 1개 출력채널 x 2입력채널 x 2x2
  const 결과 = conv2d(입력, 가중치, new Float32Array([0]), 1, 2, 0);
  근사같음(결과.데이터, [4 * 1 + 4 * 2]);
});

test("conv2d: 출력 채널마다 다른 커널이 쓰인다", () => {
  const 입력 = 텐서(new Float32Array([1, 2, 3, 4]), [1, 2, 2]);
  const 가중치 = new Float32Array([1, 1, 1, 1, 0, 0, 0, 2]);  // 2개 출력채널
  const 결과 = conv2d(입력, 가중치, new Float32Array([0, 0]), 2, 2, 0);
  assert.deepEqual(결과.형태, [2, 1, 1]);
  근사같음(결과.데이터, [10, 8]);
});

test("relu: 음수는 0이 되고 양수는 그대로다", () => {
  const 결과 = relu(텐서(new Float32Array([-2, -0.1, 0, 0.5, 3]), [1, 1, 5]));
  근사같음(결과.데이터, [0, 0, 0, 0.5, 3]);
});

test("maxPool2d: 2x2 창에서 최댓값을 고른다", () => {
  const 입력 = 텐서(new Float32Array([
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16,
  ]), [1, 4, 4]);
  const 결과 = maxPool2d(입력, 2);
  assert.deepEqual(결과.형태, [1, 2, 2]);
  근사같음(결과.데이터, [6, 8, 14, 16]);
});

test("maxPool2d: 채널마다 따로 계산한다", () => {
  const 입력 = 텐서(new Float32Array([1, 2, 3, 4, 40, 30, 20, 10]), [2, 2, 2]);
  const 결과 = maxPool2d(입력, 2);
  assert.deepEqual(결과.형태, [2, 1, 1]);
  근사같음(결과.데이터, [4, 40]);
});

test("linear: 행 우선 가중치로 행렬곱을 한다", () => {
  const 입력 = new Float32Array([1, 2, 3]);
  const 가중치 = new Float32Array([1, 0, 0, 0, 1, 1]);   // 2x3
  const 결과 = linear(입력, 가중치, new Float32Array([10, 20]), 2);
  근사같음(결과, [1 + 10, 5 + 20]);
});

test("softmax: 합이 1이 된다", () => {
  const 결과 = softmax(new Float32Array([1, 2, 3]));
  근사같음([결과.reduce((합, 값) => 합 + 값, 0)], [1], 1e-6);
  assert.ok(결과[2] > 결과[1] && 결과[1] > 결과[0]);
});

test("softmax: 값이 아주 커도 NaN 이 나오지 않는다", () => {
  const 결과 = softmax(new Float32Array([1000, 1001, 999]));
  for (const 값 of 결과) assert.ok(Number.isFinite(값), "NaN 또는 Infinity 발생");
  근사같음([결과.reduce((합, 값) => 합 + 값, 0)], [1], 1e-6);
});
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run:
```bash
cd web_version && node --test tests/test_nn.mjs
```
Expected: FAIL — `Cannot find module .../js/nn.js`

- [ ] **Step 3: nn.js 구현**

`web_version/js/nn.js`:

```javascript
/**
 * 외부 라이브러리 없이 구현한 텐서 연산 모음.
 *
 * 모든 데이터는 Float32Array 한 줄이고, 형태는 [채널, 높이, 너비] 순서로 다룬다.
 * 가중치 배치는 파이토치와 같다. 합성곱은 [출력채널][입력채널][커널y][커널x],
 * 완전연결은 [출력][입력] 행 우선이다.
 */

/** 데이터와 형태를 묶어 텐서 하나로 만든다. */
export function 텐서(데이터, 형태) {
  return { 데이터, 형태 };
}

/**
 * 2차원 합성곱. 패딩은 0으로 채운다.
 * 커널 위치를 바깥 반복문에 두어 입력을 순차적으로 읽게 했다.
 */
export function conv2d(입력, 가중치, 편향, 출력채널수, 커널크기, 패딩) {
  const [입력채널수, 높이, 너비] = 입력.형태;
  const 출높이 = 높이 + 2 * 패딩 - 커널크기 + 1;
  const 출너비 = 너비 + 2 * 패딩 - 커널크기 + 1;
  const 출력 = new Float32Array(출력채널수 * 출높이 * 출너비);

  for (let oc = 0; oc < 출력채널수; oc++) {
    const 출력시작 = oc * 출높이 * 출너비;
    출력.fill(편향[oc], 출력시작, 출력시작 + 출높이 * 출너비);

    for (let ic = 0; ic < 입력채널수; ic++) {
      const 입력시작 = ic * 높이 * 너비;
      const 커널시작 = ((oc * 입력채널수) + ic) * 커널크기 * 커널크기;

      for (let ky = 0; ky < 커널크기; ky++) {
        for (let kx = 0; kx < 커널크기; kx++) {
          const 계수 = 가중치[커널시작 + ky * 커널크기 + kx];
          if (계수 === 0) continue;

          for (let oy = 0; oy < 출높이; oy++) {
            const y = oy + ky - 패딩;
            if (y < 0 || y >= 높이) continue;      // 패딩 자리는 0이므로 건너뛴다

            const 입력행 = 입력시작 + y * 너비;
            const 출력행 = 출력시작 + oy * 출너비;
            for (let ox = 0; ox < 출너비; ox++) {
              const x = ox + kx - 패딩;
              if (x < 0 || x >= 너비) continue;
              출력[출력행 + ox] += 계수 * 입력.데이터[입력행 + x];
            }
          }
        }
      }
    }
  }
  return 텐서(출력, [출력채널수, 출높이, 출너비]);
}

/** 음수를 0으로 눌러 준다. 입력 배열을 그대로 고쳐 쓴다. */
export function relu(입력) {
  const 데이터 = 입력.데이터;
  for (let i = 0; i < 데이터.length; i++) {
    if (데이터[i] < 0) 데이터[i] = 0;
  }
  return 입력;
}

/** 크기 x 크기 창에서 최댓값을 고른다. 스트라이드는 창 크기와 같다. */
export function maxPool2d(입력, 크기) {
  const [채널수, 높이, 너비] = 입력.형태;
  const 출높이 = Math.floor(높이 / 크기);
  const 출너비 = Math.floor(너비 / 크기);
  const 출력 = new Float32Array(채널수 * 출높이 * 출너비);

  for (let c = 0; c < 채널수; c++) {
    const 입력시작 = c * 높이 * 너비;
    const 출력시작 = c * 출높이 * 출너비;
    for (let oy = 0; oy < 출높이; oy++) {
      for (let ox = 0; ox < 출너비; ox++) {
        let 최대 = -Infinity;
        for (let dy = 0; dy < 크기; dy++) {
          const 행 = 입력시작 + (oy * 크기 + dy) * 너비 + ox * 크기;
          for (let dx = 0; dx < 크기; dx++) {
            const 값 = 입력.데이터[행 + dx];
            if (값 > 최대) 최대 = 값;
          }
        }
        출력[출력시작 + oy * 출너비 + ox] = 최대;
      }
    }
  }
  return 텐서(출력, [채널수, 출높이, 출너비]);
}

/** 완전연결 층. 가중치는 [출력수][입력수] 행 우선이다. */
export function linear(입력, 가중치, 편향, 출력수) {
  const 입력수 = 입력.length;
  const 출력 = new Float32Array(출력수);
  for (let o = 0; o < 출력수; o++) {
    let 합 = 편향[o];
    const 행시작 = o * 입력수;
    for (let i = 0; i < 입력수; i++) {
      합 += 가중치[행시작 + i] * 입력[i];
    }
    출력[o] = 합;
  }
  return 출력;
}

/** 로짓을 확률로 바꾼다. 최댓값을 빼서 지수 폭주를 막는다. */
export function softmax(로짓) {
  let 최대 = -Infinity;
  for (const 값 of 로짓) if (값 > 최대) 최대 = 값;

  const 결과 = new Float32Array(로짓.length);
  let 합 = 0;
  for (let i = 0; i < 로짓.length; i++) {
    결과[i] = Math.exp(로짓[i] - 최대);
    합 += 결과[i];
  }
  for (let i = 0; i < 결과.length; i++) 결과[i] /= 합;
  return 결과;
}
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run:
```bash
cd web_version && node --test tests/test_nn.mjs
```
Expected: 11개 테스트 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add web_version/js/nn.js web_version/tests/test_nn.mjs
git commit -m "feat: 외부 의존성 없는 JS 텐서 연산 구현

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: 전처리 JS 이식 (preprocess.js)

**Files:**
- Create: `web_version/js/preprocess.js`, `web_version/tests/test_preprocess.mjs`

**Interfaces:**
- Consumes: `tests/fixtures.json` 의 `전처리` 항목
- Produces (`js/preprocess.js`):
  - `전처리(회색배열: Float32Array|Uint8Array, 폭: number, 높이: number) -> Float32Array(784) | null`
    반환값은 **0~1 범위**이며 정규화는 하지 않는다
  - `캔버스에서_회색배열(캔버스컨텍스트, 폭, 높이) -> Uint8ClampedArray` — RGBA를 회색 1채널로
  - `런렝스_풀기(rle: number[], 길이: number) -> Uint8Array` — fixture 복원용
  - `임계값`, `목표_긴변`, `출력_크기` 상수

`tools/preprocess_ref.py` 와 **같은 결과**를 내야 한다. 특히 면적 평균 축소와
0으로 채우는 평행 이동을 똑같이 구현한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`web_version/tests/test_preprocess.mjs`:

```javascript
// 파이썬 기준 구현(tools/preprocess_ref.py)과 결과가 같은지 확인한다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { 런렝스_풀기, 전처리 } from "../js/preprocess.js";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures.json", import.meta.url), "utf-8"));

test("파이썬 기준 구현과 28x28 결과가 일치한다", () => {
  assert.ok(fixture.전처리.length >= 6, "fixture 가 모자라다");
  for (const [번호, 사례] of fixture.전처리.entries()) {
    const 회색 = 런렝스_풀기(사례.rle, 사례.폭 * 사례.높이);
    const 결과 = 전처리(회색, 사례.폭, 사례.높이);
    assert.ok(결과 !== null, `${번호}번 사례가 null 을 돌려줬다`);

    let 최대오차 = 0;
    for (let i = 0; i < 784; i++) {
      최대오차 = Math.max(최대오차, Math.abs(결과[i] - 사례.기대[i]));
    }
    assert.ok(최대오차 < 1e-3, `${번호}번 사례 최대 오차 ${최대오차}`);
  }
});

test("빈 캔버스는 null 을 돌려준다", () => {
  const 회색 = new Uint8Array(280 * 280).fill(255);
  assert.equal(전처리(회색, 280, 280), null);
});

test("점 하나만 찍어도 784 길이를 돌려준다", () => {
  const 회색 = new Uint8Array(280 * 280).fill(255);
  회색[140 * 280 + 140] = 0;
  const 결과 = 전처리(회색, 280, 280);
  assert.ok(결과 !== null);
  assert.equal(결과.length, 784);
});

test("가장자리에 그린 획이 반대편으로 넘어가지 않는다", () => {
  const 회색 = new Uint8Array(280 * 280).fill(255);
  for (let y = 0; y < 30; y++) {
    for (let x = 40; x < 240; x++) 회색[y * 280 + x] = 0;
  }
  const 결과 = 전처리(회색, 280, 280);
  // 획은 28x28 한가운데(13~15행)에 놓인다. 위아래 가장자리가 비어 있어야 하며,
  // 순환 이동이면 밀려난 픽셀이 반대편 가장자리에 나타난다.
  let 위가장자리 = 0, 아래가장자리 = 0;
  for (let i = 0; i < 12 * 28; i++) 위가장자리 += 결과[i];
  for (let i = 17 * 28; i < 784; i++) 아래가장자리 += 결과[i];
  assert.ok(위가장자리 < 1e-5, `위 가장자리에 ${위가장자리} 만큼 새어 나왔다`);
  assert.ok(아래가장자리 < 1e-5, `아래 가장자리에 ${아래가장자리} 만큼 새어 나왔다`);
});

test("결과가 0과 1 사이에 들어온다", () => {
  const 회색 = new Uint8Array(280 * 280).fill(255);
  for (let y = 100; y < 180; y++) {
    for (let x = 130; x < 150; x++) 회색[y * 280 + x] = 0;
  }
  const 결과 = 전처리(회색, 280, 280);
  for (const 값 of 결과) assert.ok(값 >= 0 && 값 <= 1, `범위 밖 값 ${값}`);
});
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run:
```bash
cd web_version && node --test tests/test_preprocess.mjs
```
Expected: FAIL — `Cannot find module .../js/preprocess.js`

- [ ] **Step 3: preprocess.js 구현**

`web_version/js/preprocess.js`:

```javascript
/**
 * 사용자가 그린 그림을 모델 입력(28x28)으로 바꾸는 전처리.
 *
 * tools/preprocess_ref.py 와 같은 알고리즘이며, 두 구현이 어긋나면
 * tests/test_preprocess.mjs 가 깨진다. 한쪽만 고치지 않는다.
 *
 * 축소는 브라우저 캔버스의 보간 대신 면적 평균을 직접 계산한다.
 * drawImage 의 보간 품질은 구현마다 달라 결과가 재현되지 않기 때문이다.
 */

export const 임계값 = 20;      // 이 값을 넘는 픽셀만 획으로 본다
export const 목표_긴변 = 20;   // MNIST처럼 숫자의 긴 변을 20픽셀로 맞춘다
export const 출력_크기 = 28;

/** 캔버스의 RGBA 픽셀을 회색 1채널 배열로 바꾼다. */
export function 캔버스에서_회색배열(컨텍스트, 폭, 높이) {
  const rgba = 컨텍스트.getImageData(0, 0, 폭, 높이).data;
  const 회색 = new Uint8ClampedArray(폭 * 높이);
  for (let i = 0; i < 회색.length; i++) {
    // 흰 배경에 검은 획이므로 빨강 채널 하나만 봐도 충분하다
    회색[i] = rgba[i * 4];
  }
  return 회색;
}

/** [값, 반복횟수, ...] 형태를 원래 배열로 되돌린다. 테스트 fixture 복원용이다. */
export function 런렝스_풀기(rle, 길이) {
  const 결과 = new Uint8Array(길이);
  let 위치 = 0;
  for (let i = 0; i < rle.length; i += 2) {
    결과.fill(rle[i], 위치, 위치 + rle[i + 1]);
    위치 += rle[i + 1];
  }
  return 결과;
}

/**
 * 면적 평균으로 축소한다. 출력 픽셀 하나가 덮는 원본 영역의 평균을 내고,
 * 걸치는 칸은 겹치는 길이만큼만 반영한다. 파이썬 구현과 같은 식이다.
 */
function 면적평균_축소(배열, 높이, 너비, 새높이, 새너비) {
  const 비율x = 너비 / 새너비;
  const 가로 = new Float64Array(높이 * 새너비);
  for (let x = 0; x < 새너비; x++) {
    const 시작 = x * 비율x;
    const 끝 = (x + 1) * 비율x;
    const 첫칸 = Math.floor(시작);
    const 끝칸 = Math.min(Math.ceil(끝), 너비);
    for (let fx = 첫칸; fx < 끝칸; fx++) {
      const 겹침 = Math.min(끝, fx + 1) - Math.max(시작, fx);
      if (겹침 <= 0) continue;
      for (let y = 0; y < 높이; y++) {
        가로[y * 새너비 + x] += 배열[y * 너비 + fx] * 겹침;
      }
    }
    for (let y = 0; y < 높이; y++) 가로[y * 새너비 + x] /= 비율x;
  }

  const 비율y = 높이 / 새높이;
  const 결과 = new Float64Array(새높이 * 새너비);
  for (let y = 0; y < 새높이; y++) {
    const 시작 = y * 비율y;
    const 끝 = (y + 1) * 비율y;
    const 첫칸 = Math.floor(시작);
    const 끝칸 = Math.min(Math.ceil(끝), 높이);
    for (let fy = 첫칸; fy < 끝칸; fy++) {
      const 겹침 = Math.min(끝, fy + 1) - Math.max(시작, fy);
      if (겹침 <= 0) continue;
      for (let x = 0; x < 새너비; x++) {
        결과[y * 새너비 + x] += 가로[fy * 새너비 + x] * 겹침;
      }
    }
    for (let x = 0; x < 새너비; x++) 결과[y * 새너비 + x] /= 비율y;
  }
  return 결과;
}

/**
 * 잘려 나간 자리를 0으로 채우며 평행 이동한다.
 * 순환 이동을 쓰면 가장자리에 그린 획이 반대편 모서리로 넘어가 숫자가 망가진다.
 */
function 평행이동(배열, 크기, 이동y, 이동x) {
  const 결과 = new Float64Array(크기 * 크기);
  for (let y = 0; y < 크기; y++) {
    const 원본y = y - 이동y;
    if (원본y < 0 || 원본y >= 크기) continue;
    for (let x = 0; x < 크기; x++) {
      const 원본x = x - 이동x;
      if (원본x < 0 || 원본x >= 크기) continue;
      결과[y * 크기 + x] = 배열[원본y * 크기 + 원본x];
    }
  }
  return 결과;
}

/**
 * 흰 배경(255)에 검은 획(0)으로 그려진 회색 배열을 28x28 float32 배열로 바꾼다.
 * 획이 하나도 없으면 null 을 돌려준다.
 */
export function 전처리(회색배열, 폭, 높이) {
  // 1) 반전: 배경 0, 획 255
  const 반전 = new Float64Array(폭 * 높이);
  let 위 = 높이, 왼 = 폭, 아래 = -1, 오른 = -1;
  for (let y = 0; y < 높이; y++) {
    for (let x = 0; x < 폭; x++) {
      const 값 = 255 - 회색배열[y * 폭 + x];
      반전[y * 폭 + x] = 값;
      if (값 > 임계값) {                       // 2) 획의 경계 상자를 함께 구한다
        if (y < 위) 위 = y;
        if (y > 아래) 아래 = y;
        if (x < 왼) 왼 = x;
        if (x > 오른) 오른 = x;
      }
    }
  }
  if (아래 < 0) return null;                   // 빈 캔버스

  const 잘린높이 = 아래 - 위 + 1;
  const 잘린너비 = 오른 - 왼 + 1;
  const 잘린 = new Float64Array(잘린높이 * 잘린너비);
  for (let y = 0; y < 잘린높이; y++) {
    for (let x = 0; x < 잘린너비; x++) {
      잘린[y * 잘린너비 + x] = 반전[(위 + y) * 폭 + (왼 + x)];
    }
  }

  // 3) 긴 변을 20픽셀로 맞춰 축소 (점 하나여도 최소 1픽셀)
  const 비율 = 목표_긴변 / Math.max(잘린높이, 잘린너비);
  const 새높이 = Math.max(1, Math.round(잘린높이 * 비율));
  const 새너비 = Math.max(1, Math.round(잘린너비 * 비율));
  const 축소 = 면적평균_축소(잘린, 잘린높이, 잘린너비, 새높이, 새너비);

  // 4) 28x28 한가운데에 붙인다
  let 도화지 = new Float64Array(출력_크기 * 출력_크기);
  const y0 = Math.floor((출력_크기 - 새높이) / 2);
  const x0 = Math.floor((출력_크기 - 새너비) / 2);
  for (let y = 0; y < 새높이; y++) {
    for (let x = 0; x < 새너비; x++) {
      도화지[(y0 + y) * 출력_크기 + (x0 + x)] = 축소[y * 새너비 + x];
    }
  }

  // 5) 픽셀 무게중심을 가운데로 옮긴다
  let 총합 = 0, 무게y = 0, 무게x = 0;
  for (let y = 0; y < 출력_크기; y++) {
    for (let x = 0; x < 출력_크기; x++) {
      const 값 = 도화지[y * 출력_크기 + x];
      총합 += 값;
      무게y += y * 값;
      무게x += x * 값;
    }
  }
  if (총합 > 0) {
    const 가운데 = (출력_크기 - 1) / 2;
    const 이동y = Math.round(가운데 - 무게y / 총합);
    const 이동x = Math.round(가운데 - 무게x / 총합);
    도화지 = 평행이동(도화지, 출력_크기, 이동y, 이동x);
  }

  // 6) 0~1 범위로 맞춘다 (정규화는 model.js 에서 한다)
  const 결과 = new Float32Array(출력_크기 * 출력_크기);
  for (let i = 0; i < 결과.length; i++) {
    결과[i] = Math.min(1, Math.max(0, 도화지[i] / 255));
  }
  return 결과;
}
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run:
```bash
cd web_version && node --test tests/test_preprocess.mjs
```
Expected: 5개 테스트 모두 PASS

반올림은 양쪽 모두 0.5를 위로 올린다. 파이썬 쪽은 Task 2에서 `_반올림()`
(`floor(x + 0.5)`)으로 맞춰 두었고, 자바스크립트는 `Math.round` 가 그대로 그 동작이다.
파이썬 기본 `round` 를 쓰면 0.5에서 짝수로 내려 두 구현이 어긋나므로 되돌리지 않는다.

오차가 1e-3 을 넘으면 축소 비율(`새높이`/`새너비`)과 평행 이동량(`이동y`/`이동x`)을
양쪽에서 찍어 비교한다. 정수 하나가 어긋나면 28x28 전체가 밀린다.

- [ ] **Step 5: 커밋**

```bash
git add web_version/js/preprocess.js web_version/tests/test_preprocess.mjs
git commit -m "feat: 전처리 자바스크립트 이식과 파이썬 동등성 테스트

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: 가중치 로더와 추론 (model.js)

**Files:**
- Create: `web_version/js/model.js`, `web_version/tests/test_model.mjs`

**Interfaces:**
- Consumes: `js/nn.js` 전부, `model/weights.json`, `model/weights.bin`, `tests/fixtures.json` 의 `추론` 항목
- Produces (`js/model.js`):
  - `모델_만들기(메타: object, 버퍼: ArrayBuffer) -> 모델` — 네트워크 접근 없이 모델을 만든다 (테스트용)
  - `모델_불러오기(기준경로 = "./model/") -> Promise<모델>` — `fetch` 로 두 파일을 읽어 `모델_만들기` 를 호출
  - `순전파(모델, 이미지: Float32Array(784)) -> Float32Array(10)` — 파이토치와 같은 **로그 확률**
  - `예측(모델, 이미지: Float32Array(784)) -> Float32Array(10)` — TTA 5회를 평균한 **확률**
  - `모델.가져오기(이름: string) -> Float32Array` — 레이어 하나의 가중치 뷰

`이미지` 는 `preprocess.전처리()` 가 돌려준 0~1 배열이다. 정규화는 `순전파` 안에서 한다.
TTA로 1픽셀씩 밀어 볼 때 빈 자리를 0으로 채워야 하는데, 정규화된 배열에서는 배경이
0이 아니라 -0.424 이기 때문이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`web_version/tests/test_model.mjs`:

```javascript
// 내보낸 가중치로 JS가 파이토치와 같은 로짓을 내는지 확인한다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { 모델_만들기, 순전파, 예측 } from "../js/model.js";

const 메타 = JSON.parse(
  readFileSync(new URL("../model/weights.json", import.meta.url), "utf-8"));
const 원본 = readFileSync(new URL("../model/weights.bin", import.meta.url));
const 버퍼 = 원본.buffer.slice(원본.byteOffset, 원본.byteOffset + 원본.byteLength);
const 모델 = 모델_만들기(메타, 버퍼);

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures.json", import.meta.url), "utf-8"));

test("파이토치와 로짓이 1e-4 안에서 일치한다", () => {
  let 최대오차 = 0;
  for (const 사례 of fixture.추론) {
    const 결과 = 순전파(모델, Float32Array.from(사례.입력));
    for (let i = 0; i < 10; i++) {
      최대오차 = Math.max(최대오차, Math.abs(결과[i] - 사례.기대로짓[i]));
    }
  }
  assert.ok(최대오차 < 1e-4, `최대 오차 ${최대오차}`);
});

test("MNIST 20장을 모두 맞힌다", () => {
  for (const 사례 of fixture.추론) {
    const 확률 = 예측(모델, Float32Array.from(사례.입력));
    let 최고 = 0;
    for (let i = 1; i < 10; i++) if (확률[i] > 확률[최고]) 최고 = i;
    assert.equal(최고, 사례.정답);
  }
});

test("예측이 돌려주는 확률의 합은 1이다", () => {
  const 확률 = 예측(모델, Float32Array.from(fixture.추론[0].입력));
  const 합 = 확률.reduce((누적, 값) => 누적 + 값, 0);
  assert.ok(Math.abs(합 - 1) < 1e-5, `합이 ${합}`);
});

test("레이어 여덟 개를 모두 꺼낼 수 있고 크기가 맞는다", () => {
  const 기대크기 = {
    "conv1.weight": 144, "conv1.bias": 16,
    "conv2.weight": 4608, "conv2.bias": 32,
    "fc1.weight": 100352, "fc1.bias": 64,
    "fc2.weight": 640, "fc2.bias": 10,
  };
  for (const [이름, 크기] of Object.entries(기대크기)) {
    assert.equal(모델.가져오기(이름).length, 크기, `${이름} 크기 불일치`);
  }
});

test("추론 한 번이 50ms 안에 끝난다", () => {
  const 입력 = Float32Array.from(fixture.추론[0].입력);
  예측(모델, 입력);                                   // 워밍업
  const 시작 = performance.now();
  for (let i = 0; i < 10; i++) 예측(모델, 입력);
  const 평균 = (performance.now() - 시작) / 10;
  assert.ok(평균 < 50, `평균 ${평균.toFixed(1)}ms`);
});
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run:
```bash
cd web_version && node --test tests/test_model.mjs
```
Expected: FAIL — `Cannot find module .../js/model.js`

- [ ] **Step 3: model.js 구현**

`web_version/js/model.js`:

```javascript
/**
 * 내보낸 가중치를 읽어 순전파를 계산한다.
 *
 * weights.bin 은 float32를 이어 붙인 한 덩어리이고, weights.json 이 각 레이어의
 * 위치를 알려 준다. ArrayBuffer 위에 Float32Array 부분 뷰만 만들어 쓰므로
 * 가중치를 복사하지 않는다.
 */

import { conv2d, linear, maxPool2d, relu, softmax, 텐서 } from "./nn.js";

// 추론 시 이미지를 1픽셀씩 밀어 여러 번 예측하고 평균한다(TTA).
// 사람이 쓴 글씨는 무게중심 정렬 뒤에도 한두 픽셀씩 어긋나는데, 이 평균이 그 흔들림을 덮어 준다.
const 이동_목록 = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]];

/** 메타데이터와 바이트 버퍼로 모델 객체를 만든다. 네트워크에 접근하지 않는다. */
export function 모델_만들기(메타, 버퍼) {
  const 전체 = new Float32Array(버퍼);
  const 층사전 = new Map(메타.레이어.map((층) => [층.이름, 층]));

  return {
    메타,
    정규화: 메타.정규화,
    가져오기(이름) {
      const 층 = 층사전.get(이름);
      if (!층) throw new Error(`가중치에 ${이름} 레이어가 없습니다`);
      return 전체.subarray(층.오프셋, 층.오프셋 + 층.개수);
    },
  };
}

/**
 * weights.json 과 weights.bin 을 읽어 모델을 만든다.
 * file:// 로 열면 fetch 가 막히므로 그 경우를 알아보기 쉬운 오류로 바꿔 준다.
 */
export async function 모델_불러오기(기준경로 = "./model/") {
  let 메타응답, 가중치응답;
  try {
    [메타응답, 가중치응답] = await Promise.all([
      fetch(`${기준경로}weights.json`),
      fetch(`${기준경로}weights.bin`),
    ]);
  } catch (원인) {
    if (location.protocol === "file:") {
      const 오류 = new Error(
        "파일을 직접 열면 브라우저가 가중치 파일을 읽지 못합니다.\n" +
        "이 폴더에서 python -m http.server 8000 을 실행한 뒤\n" +
        "http://localhost:8000 으로 접속해 주세요.");
      오류.종류 = "file-protocol";
      throw 오류;
    }
    throw 원인;
  }

  if (!메타응답.ok || !가중치응답.ok) {
    const 오류 = new Error(
      `가중치를 내려받지 못했습니다 (${메타응답.status}, ${가중치응답.status}).\n` +
      "model/weights.json 과 model/weights.bin 이 배포되었는지 확인해 주세요.");
    오류.종류 = "not-found";
    throw 오류;
  }

  return 모델_만들기(await 메타응답.json(), await 가중치응답.arrayBuffer());
}

/** 28x28 이미지를 이동시킨다. 빈 자리는 0으로 채운다. */
function 이동시키기(이미지, 이동y, 이동x) {
  if (이동y === 0 && 이동x === 0) return 이미지;
  const 결과 = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    const 원본y = y - 이동y;
    if (원본y < 0 || 원본y >= 28) continue;
    for (let x = 0; x < 28; x++) {
      const 원본x = x - 이동x;
      if (원본x < 0 || 원본x >= 28) continue;
      결과[y * 28 + x] = 이미지[원본y * 28 + 원본x];
    }
  }
  return 결과;
}

/**
 * 0~1 범위의 28x28 이미지를 받아 로그 확률 10개를 돌려준다.
 * 파이토치 모델의 출력(log_softmax)과 같은 값이다.
 */
export function 순전파(모델, 이미지) {
  const { 평균, 표준편차 } = 모델.정규화;
  const 입력 = new Float32Array(784);
  for (let i = 0; i < 784; i++) 입력[i] = (이미지[i] - 평균) / 표준편차;

  let x = 텐서(입력, [1, 28, 28]);
  x = maxPool2d(relu(conv2d(x, 모델.가져오기("conv1.weight"),
                            모델.가져오기("conv1.bias"), 16, 3, 1)), 2);
  x = maxPool2d(relu(conv2d(x, 모델.가져오기("conv2.weight"),
                            모델.가져오기("conv2.bias"), 32, 3, 1)), 2);

  let 벡터 = linear(x.데이터, 모델.가져오기("fc1.weight"), 모델.가져오기("fc1.bias"), 64);
  for (let i = 0; i < 벡터.length; i++) if (벡터[i] < 0) 벡터[i] = 0;   // ReLU
  const 로짓 = linear(벡터, 모델.가져오기("fc2.weight"), 모델.가져오기("fc2.bias"), 10);

  // log_softmax: 최댓값을 빼서 지수 폭주를 막는다
  let 최대 = -Infinity;
  for (const 값 of 로짓) if (값 > 최대) 최대 = 값;
  let 합 = 0;
  for (const 값 of 로짓) 합 += Math.exp(값 - 최대);
  const 로그합 = 최대 + Math.log(합);

  const 결과 = new Float32Array(10);
  for (let i = 0; i < 10; i++) 결과[i] = 로짓[i] - 로그합;
  return 결과;
}

/**
 * TTA를 적용해 확률 10개를 돌려준다.
 * 이미지를 다섯 방향으로 밀어 각각 예측하고 확률을 평균한다.
 */
export function 예측(모델, 이미지) {
  const 누적 = new Float32Array(10);
  for (const [이동y, 이동x] of 이동_목록) {
    const 확률 = softmax(순전파(모델, 이동시키기(이미지, 이동y, 이동x)));
    for (let i = 0; i < 10; i++) 누적[i] += 확률[i];
  }
  for (let i = 0; i < 10; i++) 누적[i] /= 이동_목록.length;
  return 누적;
}
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run:
```bash
cd web_version && node --test tests/
```
Expected: `test_nn.mjs`, `test_preprocess.mjs`, `test_model.mjs` 의 21개 테스트 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add web_version/js/model.js web_version/tests/test_model.mjs
git commit -m "feat: 가중치 로더와 TTA 추론 구현

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: 화면 구성 (index.html, style.css, app.js)

**Files:**
- Create: `web_version/index.html`, `web_version/style.css`, `web_version/js/app.js`

**Interfaces:**
- Consumes: `js/preprocess.js` 의 `전처리`, `캔버스에서_회색배열`; `js/model.js` 의 `모델_불러오기`, `예측`
- Produces: 없음 (최종 화면)

데스크톱 앱과 같은 구성에 모델 입력 28x28 미리보기를 더한다. 마우스와 터치를
Pointer Events 하나로 처리하고, 그리는 동안 페이지가 스크롤되지 않게 막는다.

- [ ] **Step 1: index.html 작성**

`web_version/index.html`:

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>손글씨 숫자 인식기</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main class="바깥틀">
    <h1>손글씨 숫자 인식기</h1>
    <p class="설명">칸 안에 숫자를 크게 그리면 바로 알아맞힙니다.</p>

    <div id="오류" class="오류" hidden></div>

    <div class="본문">
      <section class="왼쪽">
        <canvas id="그림판" width="280" height="280" aria-label="숫자를 그리는 칸"></canvas>
        <button id="지우기단추" type="button">지우기</button>
      </section>

      <section class="오른쪽">
        <div class="결과칸">
          <div id="예측숫자" class="예측숫자">?</div>
          <div id="확신도" class="확신도">확신도 --</div>
        </div>

        <h2 class="소제목">숫자별 확률</h2>
        <ul id="막대목록" class="막대목록"></ul>

        <h2 class="소제목">모델이 보는 그림 (28x28)</h2>
        <canvas id="미리보기" class="미리보기" width="28" height="28"
                aria-label="모델 입력 미리보기"></canvas>
      </section>
    </div>

    <footer class="바닥글">
      외부 라이브러리 없이 자바스크립트로 직접 계산합니다.
      <span id="소요시간"></span>
    </footer>
  </main>

  <script type="module" src="js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: style.css 작성**

`web_version/style.css`:

```css
/* 데스크톱 앱과 같은 파란 계열로 맞췄다. 좁은 화면에서는 세로로 쌓인다. */
:root {
  --주색: #1a56db;
  --연한주색: #9db5e8;
  --글자색: #1f2937;
  --선색: #d1d5db;
  --배경: #f8fafc;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 24px 16px;
  font-family: "맑은 고딕", "Malgun Gothic", system-ui, sans-serif;
  color: var(--글자색);
  background: var(--배경);
}

.바깥틀 { max-width: 720px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
.설명 { margin: 0 0 20px; color: #6b7280; }

.오류 {
  margin-bottom: 16px;
  padding: 12px 14px;
  border: 1px solid #fca5a5;
  border-radius: 8px;
  background: #fef2f2;
  color: #991b1b;
  white-space: pre-wrap;   /* 안내문의 줄바꿈을 살린다 */
}

.본문 { display: flex; flex-wrap: wrap; gap: 24px; }
.왼쪽 { flex: 0 0 auto; }
.오른쪽 { flex: 1 1 260px; min-width: 260px; }

#그림판 {
  display: block;
  width: 280px;
  height: 280px;
  background: #fff;
  border: 1px solid var(--선색);
  border-radius: 8px;
  cursor: crosshair;
  touch-action: none;      /* 그리는 동안 페이지가 스크롤되지 않게 막는다 */
}

#지우기단추 {
  width: 100%;
  margin-top: 8px;
  padding: 10px;
  font: inherit;
  color: #fff;
  background: var(--주색);
  border: 0;
  border-radius: 8px;
  cursor: pointer;
}
#지우기단추:hover { background: #1543ad; }

.결과칸 { display: flex; align-items: baseline; gap: 12px; }
.예측숫자 { font-size: 5rem; font-weight: 700; line-height: 1; color: var(--주색); }
.확신도 { color: #6b7280; }

.소제목 { font-size: 0.95rem; margin: 20px 0 8px; }

.막대목록 { list-style: none; margin: 0; padding: 0; }
.막대목록 li { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
.막대목록 .숫자 { width: 1rem; font-variant-numeric: tabular-nums; }
.막대칸 { flex: 1; height: 12px; background: #eceff3; border-radius: 3px; overflow: hidden; }
.막대 { height: 100%; width: 0; background: var(--연한주색); transition: width 0.12s; }
.막대목록 li.최고 .막대 { background: var(--주색); }
.막대목록 .수치 { width: 3rem; text-align: right; font-variant-numeric: tabular-nums; }

.미리보기 {
  width: 112px;
  height: 112px;
  background: #000;
  border: 1px solid var(--선색);
  border-radius: 4px;
  image-rendering: pixelated;   /* 28x28을 또렷한 네모로 확대한다 */
}

.바닥글 { margin-top: 24px; font-size: 0.85rem; color: #9ca3af; }

@media (max-width: 640px) {
  .본문 { flex-direction: column; }
  #그림판 { width: 100%; height: auto; aspect-ratio: 1; }
}
```

- [ ] **Step 3: app.js 작성**

`web_version/js/app.js`:

```javascript
/**
 * 화면과 추론을 잇는 코드.
 *
 * 캔버스에 그린 그림을 전처리해 모델에 넣고, 예측 숫자와 확률 막대,
 * 모델이 실제로 보는 28x28 그림을 갱신한다.
 */

import { 모델_불러오기, 예측 } from "./model.js";
import { 전처리, 캔버스에서_회색배열 } from "./preprocess.js";

const 붓_굵기 = 18;   // 28x28로 줄였을 때 약 2픽셀이 되도록 맞춘 값

const 그림판 = document.getElementById("그림판");
const 그리기 = 그림판.getContext("2d", { willReadFrequently: true });
const 미리보기 = document.getElementById("미리보기");
const 미리그리기 = 미리보기.getContext("2d");
const 예측숫자 = document.getElementById("예측숫자");
const 확신도 = document.getElementById("확신도");
const 막대목록 = document.getElementById("막대목록");
const 오류칸 = document.getElementById("오류");
const 소요시간 = document.getElementById("소요시간");

let 모델 = null;
let 그리는중 = false;
let 직전 = null;

/** 0~9 확률 막대 열 줄을 만든다. */
const 막대들 = Array.from({ length: 10 }, (_, 숫자) => {
  const 줄 = document.createElement("li");
  줄.innerHTML =
    `<span class="숫자">${숫자}</span>` +
    `<span class="막대칸"><span class="막대"></span></span>` +
    `<span class="수치">0%</span>`;
  막대목록.appendChild(줄);
  return { 줄, 막대: 줄.querySelector(".막대"), 수치: 줄.querySelector(".수치") };
});

function 오류_보이기(메시지) {
  오류칸.textContent = 메시지;
  오류칸.hidden = false;
}

function 그림판_비우기() {
  그리기.fillStyle = "#fff";
  그리기.fillRect(0, 0, 그림판.width, 그림판.height);
  그리기.strokeStyle = "#000";
  그리기.fillStyle = "#000";
  그리기.lineWidth = 붓_굵기;
  그리기.lineCap = "round";
  그리기.lineJoin = "round";
}

function 결과_비우기() {
  예측숫자.textContent = "?";
  확신도.textContent = "확신도 --";
  소요시간.textContent = "";
  미리그리기.clearRect(0, 0, 28, 28);
  for (const { 줄, 막대, 수치 } of 막대들) {
    줄.classList.remove("최고");
    막대.style.width = "0";
    수치.textContent = "0%";
  }
}

/** 캔버스 좌표로 바꾼다. 화면에서 축소되어 있을 수 있어 비율을 반영한다. */
function 좌표(이벤트) {
  const 영역 = 그림판.getBoundingClientRect();
  return {
    x: (이벤트.clientX - 영역.left) * (그림판.width / 영역.width),
    y: (이벤트.clientY - 영역.top) * (그림판.height / 영역.height),
  };
}

function 점_찍기(점) {
  그리기.beginPath();
  그리기.arc(점.x, 점.y, 붓_굵기 / 2, 0, Math.PI * 2);
  그리기.fill();
}

그림판.addEventListener("pointerdown", (이벤트) => {
  이벤트.preventDefault();
  그림판.setPointerCapture(이벤트.pointerId);
  그리는중 = true;
  직전 = 좌표(이벤트);
  점_찍기(직전);
});

그림판.addEventListener("pointermove", (이벤트) => {
  if (!그리는중) return;
  이벤트.preventDefault();
  const 현재 = 좌표(이벤트);
  그리기.beginPath();
  그리기.moveTo(직전.x, 직전.y);
  그리기.lineTo(현재.x, 현재.y);
  그리기.stroke();
  직전 = 현재;
});

for (const 이름 of ["pointerup", "pointercancel", "pointerleave"]) {
  그림판.addEventListener(이름, () => {
    if (!그리는중) return;
    그리는중 = false;
    직전 = null;
    인식();
  });
}

document.getElementById("지우기단추").addEventListener("click", () => {
  그림판_비우기();
  결과_비우기();
});

/** 미리보기 캔버스에 28x28 입력을 흰 글씨/검은 배경으로 그린다. */
function 미리보기_갱신(이미지) {
  const 픽셀 = 미리그리기.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const 밝기 = Math.round(이미지[i] * 255);
    픽셀.data[i * 4] = 밝기;
    픽셀.data[i * 4 + 1] = 밝기;
    픽셀.data[i * 4 + 2] = 밝기;
    픽셀.data[i * 4 + 3] = 255;
  }
  미리그리기.putImageData(픽셀, 0, 0);
}

function 인식() {
  if (!모델) return;
  const 회색 = 캔버스에서_회색배열(그리기, 그림판.width, 그림판.height);
  const 이미지 = 전처리(회색, 그림판.width, 그림판.height);
  if (이미지 === null) return;          // 빈 캔버스면 아무것도 하지 않는다

  const 시작 = performance.now();
  const 확률 = 예측(모델, 이미지);
  const 걸린시간 = performance.now() - 시작;

  let 최고 = 0;
  for (let i = 1; i < 10; i++) if (확률[i] > 확률[최고]) 최고 = i;

  예측숫자.textContent = String(최고);
  확신도.textContent = `확신도 ${(확률[최고] * 100).toFixed(1)}%`;
  소요시간.textContent = `(추론 ${걸린시간.toFixed(1)}ms)`;
  미리보기_갱신(이미지);

  막대들.forEach(({ 줄, 막대, 수치 }, 숫자) => {
    줄.classList.toggle("최고", 숫자 === 최고);
    막대.style.width = `${확률[숫자] * 100}%`;
    수치.textContent = `${Math.round(확률[숫자] * 100)}%`;
  });
}

async function 시작하기() {
  그림판_비우기();
  결과_비우기();
  try {
    모델 = await 모델_불러오기("./model/");
  } catch (오류) {
    오류_보이기(오류.message);
  }
}

시작하기();
```

- [ ] **Step 4: 정적 서버를 띄워 브라우저에서 확인**

Run:
```bash
cd web_version && $PY -m http.server 8000
```
브라우저에서 `http://localhost:8000` 을 열고 확인한다.

- 숫자를 그리면 획을 뗄 때 예측이 바뀐다
- 28x28 미리보기에 흰 글씨가 가운데 정렬되어 나타난다
- 확률 막대에서 예측 숫자만 진한 파랑이다
- `지우기` 를 누르면 화면과 결과가 함께 초기화된다
- 바닥글의 추론 시간이 50ms 미만이다
- 0부터 9까지 한 번씩 그려 모두 맞히는지 본다

- [ ] **Step 5: file:// 안내가 뜨는지 확인**

`web_version/index.html` 을 탐색기에서 더블클릭해 연다.
Expected: 빨간 상자에 `python -m http.server 8000` 안내가 뜬다. 빈 화면이 아니어야 한다.

- [ ] **Step 6: 좁은 화면에서 확인**

브라우저 개발자 도구의 기기 모드를 375x812로 맞춘다.
Expected: 좌우 배치가 세로로 바뀌고, 그림판이 화면 너비에 맞게 줄어들며,
손가락으로 그릴 때 페이지가 스크롤되지 않는다.

- [ ] **Step 7: 커밋**

```bash
git add web_version/index.html web_version/style.css web_version/js/app.js
git commit -m "feat: 손글씨 입력 화면과 추론 연결

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: GitHub Pages 배포와 문서, 최종 검증

**Files:**
- Create: `.github/workflows/pages.yml`, `web_version/CLAUDE.md`, `web_version/README.md`
- Modify: `README.md` (배포 주소 안내 추가)

**Interfaces:**
- Consumes: 앞선 모든 작업의 산출물
- Produces: 없음 (배포 설정과 문서)

- [ ] **Step 1: 배포 워크플로우 작성**

`.github/workflows/pages.yml`:

```yaml
# web_version 폴더만 GitHub Pages 로 배포한다.
# tools/ 와 tests/ 는 오프라인 전용이라 사이트에 올리지 않는다.
name: Pages 배포

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  테스트:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '24'
      - name: 자바스크립트 테스트
        run: node --test web_version/tests/

  배포:
    needs: 테스트
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.배포단계.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4

      - name: 정적 파일만 추려 내기
        run: |
          mkdir -p _사이트
          rsync -a --exclude 'tools/' --exclude 'tests/' web_version/ _사이트/
          echo "--- 배포 대상 ---"
          find _사이트 -type f | sort

      - uses: actions/configure-pages@v5

      - uses: actions/upload-pages-artifact@v3
        with:
          path: _사이트

      - id: 배포단계
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: 워크플로우가 올릴 파일 목록을 로컬에서 미리 확인**

Run:
```bash
cd "D:/DSMP/3장 손글씨 앱/study01_MNIST"
find web_version -type f -not -path 'web_version/tools/*' -not -path 'web_version/tests/*' | sort
```
Expected: `index.html`, `style.css`, `js/*.js` 4개, `model/weights.bin`, `model/weights.json`,
`CLAUDE.md`, `README.md` 만 나온다. `.pt` 파일이나 `.py` 파일이 섞여 있으면 안 된다.

- [ ] **Step 3: web_version/CLAUDE.md 작성**

```markdown
# web_version — 순수 자바스크립트 손글씨 인식 웹 앱

## 이 폴더의 역할

브라우저에서 마우스나 손가락으로 숫자를 그리면 CNN이 인식한다.
GitHub Pages에 정적으로 배포되며 `desktop_version/` 과 코드를 공유하지 않는다.

## 절대 규칙

- **런타임 외부 의존성 0개.** CDN, npm 패키지, 빌드 단계, 번들러 모두 금지다.
  브라우저가 `index.html` 과 ES 모듈을 그대로 읽는다.
- 모든 경로는 상대경로로 쓴다. 사이트가 하위 경로에 배포될 수 있다.
- 모든 코드와 주석, 변수명은 한글로 쓴다.

## 폴더 구성

| 경로 | 역할 | 배포 |
| --- | --- | --- |
| `index.html` `style.css` | 화면 구조와 스타일 | O |
| `js/nn.js` | 텐서 연산 (conv2d, relu, maxPool2d, linear, softmax) | O |
| `js/preprocess.js` | 캔버스 픽셀 → 28x28 배열 | O |
| `js/model.js` | 가중치 로드, 순전파, TTA | O |
| `js/app.js` | 화면과 추론 연결 | O |
| `model/weights.bin` `weights.json` | 학습된 가중치 (414KB) | O |
| `tools/손글씨_생성.py` | 글꼴 손글씨 생성 (학습/벤치 글꼴 분리) | X |
| `tools/*.py` | 학습과 내보내기 (오프라인 전용) | X |
| `tests/*.mjs` | node 동등성 테스트 | X |

## 명령

| 목적 | 명령 |
| --- | --- |
| 로컬 확인 | `python -m http.server 8000` 후 `http://localhost:8000` |
| JS 테스트 | `node --test tests/` |
| 파이썬 테스트 | `cd tools && python -m unittest test_tools` |
| 모델 재학습 | `cd tools && python train_web.py --epochs 15` |
| 가중치 재생성 | `cd tools && python export_weights.py` |
| 정확도 확인 | `cd tools && python bench.py` |

## 주의

- `file://` 로 열면 `fetch` 가 막혀 가중치를 읽지 못한다. 반드시 정적 서버로 연다.
  이 경우 화면에 안내가 뜨도록 되어 있으니, 빈 화면이 보이면 그쪽을 먼저 의심한다.
- 모델 구조를 바꾸면 **세 곳**을 함께 고쳐야 한다.
  `tools/web_model.py`, `tools/export_weights.py` 의 `레이어_순서`, `js/model.js` 의 `순전파`.
  그 뒤 `export_weights.py` 와 `make_fixtures.py` 를 다시 돌린다.
- 전처리를 고치면 `tools/preprocess_ref.py` 와 `js/preprocess.js` 를 함께 고친다.
  한쪽만 고치면 `tests/test_preprocess.mjs` 가 깨진다.
- 파이썬 `round` 는 0.5에서 짝수로 내리고 자바스크립트 `Math.round` 는 위로 올린다.
  두 구현에서 반올림을 쓸 때 이 차이를 기억한다.
- `손글씨_생성.py` 의 `학습_글꼴` 과 `벤치_글꼴` 은 **절대 겹치면 안 된다.**
  벤치마크는 학습에서 본 적 없는 글자꼴에 대한 일반화를 재는 잣대다. 학습 글꼴을
  벤치마크에 넣으면 게이트가 스스로를 채점하게 되어 의미를 잃는다.
- MNIST에는 밑변 세리프가 달린 넓적한 '1'(Segoe Script 계열) 같은 글자꼴이 없다.
  그래서 글꼴로 그린 숫자 20,000장을 실제 전처리까지 통과시켜 MNIST에 25% 비중으로
  섞어 학습한다. 표본은 `data/글꼴표본.npz` 에 캐시되며, 지우면 다시 생성된다(약 6분).
```

- [ ] **Step 4: web_version/README.md 작성**

```markdown
# 손글씨 숫자 인식기 (웹 버전)

브라우저에서 숫자를 그리면 CNN이 0~9 중 무엇인지 알아맞힌다.
외부 라이브러리 없이 자바스크립트로 직접 순전파를 계산한다.

## 실행

```bash
python -m http.server 8000
```

브라우저에서 `http://localhost:8000` 을 연다.
`index.html` 을 더블클릭해 열면 브라우저가 가중치 파일을 읽지 못한다.

## 동작 방식

1. 캔버스에 그린 그림을 읽는다
2. 획만 잘라내 긴 변을 20픽셀로 줄이고, 28x28 한가운데에 무게중심을 맞춘다
3. 414KB짜리 가중치로 합성곱 신경망 순전파를 계산한다
4. 1픽셀씩 민 다섯 장을 각각 예측해 확률을 평균한다 (TTA)

## 모델

| 항목 | 값 |
| --- | --- |
| 구조 | Conv16 → Pool → Conv32 → Pool → FC64 → FC10 |
| 파라미터 | 105,866개 |
| 가중치 크기 | 414KB (float32) |
| 추론 연산량 | 약 110만 MAC |

## 가중치를 다시 만들려면

```bash
cd tools
python train_web.py --epochs 15    # web_mnist_cnn.pt 생성
python export_weights.py           # ../model/weights.bin, weights.json 생성
python make_fixtures.py            # ../tests/fixtures.json 갱신
python bench.py                    # 정확도 확인
```

## 테스트

```bash
node --test tests/
```
```

- [ ] **Step 5: 루트 README.md 에 배포 안내 추가**

`README.md` 끝에 다음을 붙인다.

```markdown
## GitHub Pages 배포

`main` 에 push하면 `.github/workflows/pages.yml` 이 `web_version/` 을 배포한다.
`tools/` 와 `tests/` 는 제외된다.

처음 한 번은 GitHub에서 직접 설정해야 한다.

1. GitHub에 저장소를 만들고 `git remote add origin <주소>` 후 push
2. 저장소 Settings → Pages → Source 를 **GitHub Actions** 로 바꾼다
3. Actions 탭에서 `Pages 배포` 워크플로우가 초록불이 되면 주소가 나온다
```

- [ ] **Step 6: 전체 테스트 실행**

Run:
```bash
cd web_version && node --test tests/
cd tools && $PY -m unittest test_tools -v
```
Expected: JS 21개, 파이썬 15개 모두 PASS

- [ ] **Step 7: 정확도 게이트 재확인**

Run:
```bash
cd web_version/tools && $PY bench.py
```
Expected: `목표 달성`, 98.5% 이상

- [ ] **Step 8: 가중치 크기 확인**

Run:
```bash
ls -la web_version/model/
```
Expected: `weights.bin` 이 423,464 바이트 (500KB 이하)

- [ ] **Step 9: 데스크톱 버전이 여전히 동작하는지 확인**

```powershell
Start-Process (Join-Path ([Environment]::GetFolderPath('Desktop')) "손글씨 숫자 인식기.lnk")
Start-Sleep -Seconds 15
Get-Process | Where-Object { $_.ProcessName -eq "pythonw" } | Select-Object Id, MainWindowTitle
```
Expected: `손글씨 숫자 인식기` 창이 뜬다. 확인 후 닫는다.

- [ ] **Step 10: 커밋**

```bash
git add .github README.md web_version/CLAUDE.md web_version/README.md
git commit -m "feat: GitHub Pages 배포 워크플로우와 폴더별 안내 문서 추가

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 11: 사용자 확인이 필요한 단계 안내**

GitHub 저장소 생성과 Pages 설정은 계정 권한이 필요해 대신 할 수 없다.
구현이 끝나면 위 루트 README의 3단계를 사용자에게 안내한다.

---

## 작업 순서 요약

| # | 작업 | 산출물 | 게이트 |
| --- | --- | --- | --- |
| 1 | 저장소 초기화와 desktop_version 이동 | 폴더 구조, 바로가기 갱신 | 앱이 새 경로에서 실행됨 |
| 2 | 웹 모델 정의와 전처리 기준 구현 | `web_model.py`, `preprocess_ref.py` | 파이썬 테스트 11개 |
| 3 | 증강 학습과 정확도 게이트 | `web_mnist_cnn.pt` | MNIST 99.0%, 가혹 98.5% |
| 4 | 가중치 내보내기와 fixture | `weights.bin/json`, `fixtures.json` | 파이썬 테스트 15개 |
| 5 | JS 텐서 연산 | `nn.js` | JS 테스트 11개 |
| 6 | 전처리 JS 이식 | `preprocess.js` | 파이썬과 오차 1e-3 이하 |
| 7 | 가중치 로더와 추론 | `model.js` | 파이토치와 오차 1e-4 이하 |
| 8 | 화면 구성 | `index.html`, `style.css`, `app.js` | 브라우저 수동 확인 |
| 9 | 배포와 문서 | `pages.yml`, CLAUDE.md 3개 | 전체 테스트 + 게이트 재확인 |
