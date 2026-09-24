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
