# 손글씨 숫자 인식기

## 폴더 구성

| 폴더 | 내용 |
| --- | --- |
| `desktop_version/` | PyTorch + tkinter 윈도우 앱 |
| `web_version/` | 순수 자바스크립트 웹 앱 (GitHub Pages 배포) |

작업을 시작하기 전에 어느 쪽 이야기인지 먼저 확인하고, 해당 폴더의 CLAUDE.md를 읽는다.
두 버전은 코드를 공유하지 않는다. 한쪽을 고쳤다고 다른 쪽이 따라 바뀌지 않는다.

## 공통 규칙

- 모든 코드와 주석, 변수명은 한글로 쓴다.
- 전처리 알고리즘은 두 버전이 같아야 한다. 한쪽을 바꾸면
  `web_version/tests/` 의 동등성 테스트가 깨지므로 양쪽을 함께 고친다.

## 설계 문서

- 스펙: `docs/superpowers/specs/2026-09-24-web-desktop-split-design.md`
- 계획: `docs/superpowers/plans/2026-09-24-web-desktop-split.md`
