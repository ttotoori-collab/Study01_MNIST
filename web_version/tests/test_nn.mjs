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
