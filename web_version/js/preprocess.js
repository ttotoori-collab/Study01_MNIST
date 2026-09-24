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
