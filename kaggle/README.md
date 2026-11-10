# Kaggle 무료 GPU로 데모 띄우기

발표 데모(`tools/demo_server.py`)를 Kaggle 무료 T4 GPU에서 실행하는 방법입니다. 비용은 0원이고, GPU는 주 30시간까지 무료입니다.

| 파일 | 내용 |
|---|---|
| `okplatform_kaggle.zip` (769 MB, git 제외) | `okg/`, `tools/demo_server.py`, `tools/demo/`, 릴리스 `v20261105-005127-weekly`. `local_secrets.py`와 학습 데이터는 들어 있지 않습니다. |
| `okchat_demo_kaggle.ipynb` | Kaggle에서 위에서부터 차례로 실행하는 노트북(셀 ①~⑥) |

## 1. Kaggle 계정 준비 (최초 1회)
1. kaggle.com에 로그인합니다.
2. 프로필 → **Settings → Phone verification**에서 전화번호 인증을 합니다. 인증해야 GPU와 Internet을 켤 수 있습니다.

## 2. 데이터셋 올리기 (최초 1회, 비공개)
1. **Datasets → + New Dataset**을 누르고 `kaggle/okplatform_kaggle.zip`을 끌어다 놓습니다.
2. 제목은 `okplatform-kaggle`로 하고, **Visibility는 반드시 Private**로 둡니다.
   릴리스의 `memory/`에는 비상업·라이선스 검토 중인 데이터의 원문이 들어 있어서 공개하면 안 됩니다.
3. **Create**를 누르고 업로드와 처리가 끝날 때까지 기다립니다(몇 분 걸림). zip은 Kaggle이 자동으로 풉니다.

## 3. 노트북 열기
1. **Code → + New Notebook**을 엽니다.
2. **File → Import Notebook**에서 `kaggle/okchat_demo_kaggle.ipynb`를 고릅니다.
3. 오른쪽 **Settings** 패널을 다음과 같이 설정합니다.
   - **Accelerator: GPU T4 x2** (P100은 쓰지 마세요)
   - **Internet: On**
4. **+ Add Input → Your Work → Datasets**에서 `okplatform-kaggle`을 추가합니다.

## 4. 실행 (셀 ①부터 차례로)
| 셀 | 하는 일 | 걸리는 시간 |
|---|---|---|
| ① | 데이터셋을 `/kaggle/working/okplatform`으로 복사하고 Python 버전과 GPU 확인 | 1분 |
| ② | 학습 때와 같은 버전으로 패키지 설치 (Kaggle 기본 torch 사용) | 3~5분 |
| ③ | 데모 서버 시작: Qwen3Guard-Gen-8B(16 GB) 다운로드 후 4-bit 로드. 30초마다 진행 상황 출력 | 10~20분 |
| ④ | 모든 시나리오(6개)를 데모 화면과 같은 방식으로 돌려 턴별 판정·지연시간 출력, 4090 결과와 다르면 표시 | 1~2분 |
| ⑤ | cloudflared 무료 터널로 `https://….trycloudflare.com` 주소 만들기 | 30초 |
| ⑥ | 녹화가 끝나면 서버와 터널 닫기 | 즉시 |

⑤에서 나온 주소를 브라우저로 열면 데모 화면이 나옵니다. **녹화하는 동안 Kaggle 탭은 열어 두세요.** 탭을 닫으면 잠시 뒤 세션이 멈춥니다.
끝나면 ⑥을 실행하고, 오른쪽 위 전원 버튼(**Stop Session**)을 눌러 GPU 사용 시간을 멈추세요.

## RTX 4090(run 2)과 다른 점
- 릴리스와 점수 설정(8B 4-bit NF4, 2,048 토큰, 배치 4)은 같습니다. 다만 T4는 bf16을 하드웨어로 지원하지 않아 더 느리고, 점수가 소수점 단위로 다를 수 있습니다. ④에서 판정을 확인하세요.
- 시간 제한(`guard_timeout_ms`)을 2초에서 15초로 늘렸습니다. T4가 느려서 생기는 시간 초과가 `safe_answer`로 잘못 표시되지 않게 하기 위해서입니다. 점수는 바뀌지 않고, 실제 지연시간은 화면과 ④에 그대로 나옵니다. 발표에서 지연시간을 말할 때는 4090 측정값과 구분하세요.
- 터널 주소는 아는 사람이면 누구나 접속할 수 있습니다. 녹화용으로만 쓰고 끝나면 ⑥으로 닫으세요.

## 문제가 생기면
- **① "Python 3.11 이상 필요"**: Kaggle 이미지가 오래된 경우입니다. Settings → Environment에서 **Latest**를 고르세요.
- **③ 서버가 종료됨**: 셀에 출력된 `demo.log` 마지막 부분을 확인하세요. `CUDA out of memory`가 나오면 Accelerator가 T4가 맞는지 확인하세요. "릴리스는 8B 4-bit 로 보정됐는데…"가 나오면 bitsandbytes 설치가 실패한 것이니 ②를 다시 실행하세요.
- **⑤ 주소가 안 나옴**: Internet이 On인지 확인하고 ⑤를 다시 실행하세요.
- **코드나 시나리오를 고쳤을 때**: 저장소 루트에서 아래 명령으로 zip을 다시 만들고, 데이터셋에서 **New Version**으로 올리세요.
  ```bash
  rm -rf /tmp/okb && mkdir -p /tmp/okb/okplatform/tools /tmp/okb/okplatform/releases/full /tmp/okb/okplatform/deploy && rsync -a --exclude __pycache__ --exclude .DS_Store --exclude .ipynb_checkpoints okg /tmp/okb/okplatform/ && cp tools/demo_server.py /tmp/okb/okplatform/tools/ && rsync -a --exclude .DS_Store tools/demo /tmp/okb/okplatform/tools/ && cp deploy/requirements.txt /tmp/okb/okplatform/deploy/ && cp releases/full/CURRENT /tmp/okb/okplatform/releases/full/ && rsync -a "releases/full/$(cat releases/full/CURRENT)" /tmp/okb/okplatform/releases/full/ && rm -f kaggle/okplatform_kaggle.zip && (cd /tmp/okb && zip -qr -X "$OLDPWD/kaggle/okplatform_kaggle.zip" okplatform)
  ```
