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
