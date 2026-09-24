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
