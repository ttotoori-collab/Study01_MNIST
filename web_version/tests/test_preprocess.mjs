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
