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
python train_web.py --epochs 20    # web_mnist_cnn.pt 생성
python export_weights.py           # ../model/weights.bin, weights.json 생성
python make_fixtures.py            # ../tests/fixtures.json 갱신
python bench.py                    # 정확도 확인
```

## 테스트

```bash
node --test "tests/*.mjs"
```
