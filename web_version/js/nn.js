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
