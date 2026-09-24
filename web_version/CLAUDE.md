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
| JS 테스트 | `node --test "tests/*.mjs"` |
| 파이썬 테스트 | `cd tools && python -m unittest test_tools` |
| 모델 재학습 | `cd tools && python train_web.py --epochs 20` |
| 가중치 재생성 | `cd tools && python export_weights.py` |
| 정확도 확인 | `cd tools && python bench.py` |

## 주의

- `file://` 로 열면 `fetch` 가 막혀 가중치를 읽지 못한다. 반드시 정적 서버로 연다.
  이 경우 화면에 안내가 뜨도록 되어 있으니, 빈 화면이 보이면 그쪽을 먼저 의심한다.
- 테스트는 `node --test "tests/*.mjs"` 처럼 **글로브**로 지정한다. `node --test tests/`
  는 동작하지 않는다 — Node 22부터 `--test` 인자가 글로브 기반으로 바뀌어 디렉터리를
  모듈로 취급하고 `ERR_MODULE_NOT_FOUND` 로 죽는다. 파일명 `test_*.mjs` 의 밑줄도
  node 기본 패턴(`test-*`, `*_test`, `*.test.*`)과 맞지 않는다.
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
