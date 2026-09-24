# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 손글씨 숫자 인식기

같은 문제(MNIST 손글씨 숫자 인식)를 두 가지 방식으로 구현한 저장소다.

| 폴더 | 내용 |
| --- | --- |
| `desktop_version/` | PyTorch + tkinter 윈도우 앱 (완성) |
| `web_version/` | 순수 자바스크립트 웹 앱, GitHub Pages 배포 (구축 중) |

작업을 시작하기 전에 어느 쪽 이야기인지 먼저 확인하고 해당 폴더의 CLAUDE.md를 읽는다.
두 버전은 **코드를 공유하지 않는다.** 한쪽을 고쳤다고 다른 쪽이 따라 바뀌지 않는다.

## 공통 규칙

- 모든 코드와 주석, 변수명은 **한글로** 쓴다. 함수명·변수명도 한글이다.
- 파이썬은 전체 경로로 실행한다: `C:\Users\somes\AppData\Local\Programs\Python\Python313\python.exe`
  `python` 만 치면 Microsoft Store 스텁이 잡혀 아무 일도 일어나지 않는다.
- 셸은 Windows PowerShell 5.1이다. `&&`, `??`, 삼항 연산자를 쓸 수 없다.
  한글이 든 `.ps1` 파일은 UTF-8 BOM이 없으면 파싱에 실패한다.
- 콘솔에서 한글이 깨지면 `PYTHONIOENCODING=utf-8` 을 앞에 붙인다. 코드 문제가 아니라
  콘솔 코드페이지(cp949) 문제다.

## 명령

### desktop_version

```bash
cd desktop_version
python train.py --epochs 5     # mnist_cnn.pt 생성
python app.py                  # 앱 실행 (바탕 화면 바로가기로도 실행됨)
python make_icon.py            # app.ico 재생성
```

### web_version

```bash
cd web_version/tools
python -m unittest test_tools -v                    # 파이썬 테스트 전체
python -m unittest test_tools.전처리_테스트          # 클래스 하나만
python -m unittest test_tools.전처리_테스트.test_빈_캔버스는_None을_돌려준다   # 테스트 하나만
python train_web.py --epochs 20                     # 학습 (백그라운드로 돌릴 것, 약 18분)
python export_weights.py                            # .pt -> ../model/weights.bin + weights.json
python make_fixtures.py                             # ../tests/fixtures.json 갱신
python bench.py                                     # 정확도 게이트 (미달 시 종료 코드 1)
```

```bash
cd web_version
node --test tests/                 # 자바스크립트 테스트 전체
node --test tests/test_nn.mjs      # 파일 하나만
python -m http.server 8000         # 로컬 확인 (file:// 로 열면 fetch 가 막힌다)
```

파이썬 테스트는 **반드시 `web_version/tools` 안에서** 실행한다. 임포트가
`from web_model import ...` 처럼 평평해서 다른 위치에서는 실패한다.
pytest는 설치되어 있지 않다. 표준 라이브러리 `unittest` 를 쓴다.

## 구조

### 산출물 파이프라인

웹 버전의 가중치는 여러 단계를 거쳐 만들어진다. 중간에 하나만 고치면 어긋난다.

```
MNIST 60,000장 + 글꼴 숫자 20,000장
        -> train_web.py      -> tools/web_mnist_cnn.pt   (git 제외)
        -> export_weights.py -> model/weights.bin + weights.json  (커밋 대상, 배포됨)
        -> js/model.js 가 fetch 로 읽어 브라우저에서 순전파
```

`make_fixtures.py` 는 이 파이프라인에서 파이토치의 출력을 뽑아 `tests/fixtures.json` 에
굽는다. 자바스크립트 테스트는 그 값과 대조해 두 구현이 같은지 검증한다.

### 함께 고쳐야 하는 것들

여기가 이 저장소에서 가장 다치기 쉬운 지점이다.

- **모델 구조를 바꾸면 세 곳**을 함께 고친다: `tools/web_model.py`,
  `tools/export_weights.py` 의 `레이어_순서`, `js/model.js` 의 `순전파`.
  그 뒤 `export_weights.py` 와 `make_fixtures.py` 를 다시 돌린다.
- **전처리를 바꾸면 두 곳**을 함께 고친다: `tools/preprocess_ref.py` 와 `js/preprocess.js`.
  한쪽만 고치면 `tests/test_preprocess.mjs` 가 깨진다.
- 데스크톱의 `app.py` 에도 같은 알고리즘의 사본이 있지만 웹 테스트가 검증하지 않는다.
  전처리를 바꿀 때 함께 볼 것.

`tools/preprocess_ref.py` 는 자바스크립트 이식의 **정답지**다. 명시적 반복문을
numpy 벡터화로 "개선"하지 마라. 구조가 바뀌면 1:1 이식이 어긋난다.

### 정확도 게이트

최종 사용자 기준은 "사람이 0~9를 여러 형태로 50번 그려 49번 이상"이다.
MNIST 테스트 정확도만으로는 이를 보장하지 못하므로 두 가지를 함께 본다.

| 기준 | 값 |
| --- | --- |
| MNIST 테스트 정확도 | 99.0% 이상 |
| 가혹 손글씨 시뮬레이션 (`bench.py`) | 98.5% 이상 |

`bench.py` 는 글꼴로 숫자를 그린 뒤 손떨림·획 굵기·기울기·회전·이동을 더해 실제
전처리를 통과시켜 측정한다. 미달하면 종료 코드 1을 낸다.

**`손글씨_생성.py` 의 `학습_글꼴`(19종)과 `벤치_글꼴`(5종)은 절대 겹치면 안 된다.**
벤치마크는 학습에서 본 적 없는 글자꼴에 대한 일반화를 재는 잣대다. 겹치면 게이트가
스스로를 채점하게 되어 의미를 잃는다.

MNIST에는 밑변 세리프가 달린 넓적한 '1'(Segoe Script 계열) 같은 글자꼴이 없다.
증강만으로 학습한 모델은 그 형태를 '2'로 읽어 벤치마크가 94.5%에 그쳤다. 그래서
글꼴로 그린 숫자를 실제 전처리까지 통과시켜 MNIST에 25% 비중으로 섞는다.
표본은 `tools/data/글꼴표본.npz` 에 캐시되며 지우면 다시 생성된다(약 6분).

### 웹 런타임 제약

- **외부 의존성 0개.** CDN, npm 패키지, 빌드 단계, 번들러 모두 금지다.
  브라우저가 `index.html` 과 ES 모듈을 그대로 읽는다.
- 모든 경로는 상대경로로 쓴다. 사이트가 하위 경로에 배포될 수 있다.
- `tools/` 와 `tests/` 는 오프라인 전용이라 배포에서 제외된다.

### 언어 간 함정

파이썬 `round()` 는 0.5에서 짝수로 내리고(`round(0.5) == 0`) 자바스크립트
`Math.round` 는 위로 올린다. 전처리는 양쪽에서 `floor(x + 0.5)` 로 통일해 두었다
(`preprocess_ref._반올림`). 파이썬 기본 `round()` 로 되돌리면 축소 비율과 평행 이동량이
정수 하나씩 어긋나 28x28 전체가 밀린다.

## 설계 문서

- 스펙: `docs/superpowers/specs/2026-09-24-web-desktop-split-design.md`
- 계획: `docs/superpowers/plans/2026-09-24-web-desktop-split.md`

웹 버전은 이 계획에 따라 구축 중이다. `web_version/js/`, `model/`, `tests/` 는
아직 없을 수 있다. 무엇이 남았는지는 계획 문서의 작업 순서 요약표를 본다.
