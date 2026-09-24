# 손글씨 숫자 인식기

마우스나 손가락으로 숫자를 그리면 CNN이 0~9 중 무엇인지 알아맞히는 프로그램이다.
같은 문제를 두 가지 방식으로 구현했다.

| 버전 | 실행 환경 | 모델 | 가중치 크기 |
| --- | --- | --- | --- |
| [desktop_version](desktop_version/) | 윈도우 + PyTorch + tkinter | Conv32-Conv64-FC128 | 4.8MB |
| [web_version](web_version/) | 브라우저 (순수 JS) | Conv16-Conv32-FC64 | 414KB |

웹 버전은 외부 라이브러리 없이 자바스크립트로 직접 순전파를 계산하며,
GitHub Pages에 정적으로 배포된다.

## GitHub Pages 배포

`main` 에 push하면 `.github/workflows/pages.yml` 이 `web_version/` 을 배포한다.
`tools/` 와 `tests/` 는 제외된다.

처음 한 번은 GitHub에서 직접 설정해야 한다.

1. GitHub에 저장소를 만들고 `git remote add origin <주소>` 후 push
2. 저장소 Settings → Pages → Source 를 **GitHub Actions** 로 바꾼다
3. Actions 탭에서 `Pages 배포` 워크플로우가 초록불이 되면 주소가 나온다
