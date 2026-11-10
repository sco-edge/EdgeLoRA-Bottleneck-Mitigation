# 한국어 안내

이 저장소는 **Mitigating Bottlenecks in Batched Multi-Tenant LoRA Serving on Edge Devices**의 실험 코드입니다. [EdgeLoRA](https://github.com/shenzheyu/EdgeLoRA)를 기반으로, 어댑터 로딩 중의 대기와 LoRA 출력 누적 과정의 불필요한 복사를 줄이는 구현을 담았습니다.

| 구성 | 의미 |
|---|---|
| Baseline | 동기 로딩 + 기존 출력 누적 |
| Async | 비동기 어댑터 준비 + 기존 출력 누적 |
| In-place | 동기 로딩 + 제자리 출력 누적 |
| Combined | 두 방법을 함께 적용 |

Jetson은 서빙과 성능 측정에, RTX 2080 Ti는 작업 어댑터와 라우터 학습에 사용했습니다. Hugging Face에서 기본 모델과 학습 데이터를 확보했고, 학습한 체크포인트를 GGUF로 변환했습니다. 성능 비교에서는 어댑터 ID를 직접 지정하여 라우터 선택 효과와 런타임 효과를 분리했습니다.

## 실행 순서

1. [영문 README](../README.md)의 Quick start에 따라 공식 EdgeLoRA 고정 커밋을 내려받아 연구 패치를 적용하고, `bash scripts/build_runtime.sh`로 런타임을 빌드합니다.
2. [학습·변환 안내](../training/README.md)를 따라 모델 파일을 준비합니다.
3. `configs/runtime.example.json`을 `configs/runtime.json`으로 복사하고 로컬 모델 경로를 설정합니다.
4. `python3 scripts/serve.py --mode combined`로 서버를 실행합니다.
5. [실험 안내](experiments.md)에 따라 비교할 구성별로 서버를 재시작하고 같은 요청 조건을 적용합니다.

실행 예제의 `--mode`와 요청의 `adapter_load_policy`를 함께 맞춰야 합니다. Async와 Combined는 `backend-async`, Baseline과 In-place는 `sync`를 사용합니다.

## 포함 범위

핵심 런타임 변경, 실험 클라이언트와 분석 코드, 학습·변환 도구 및 출처 정보를 포함합니다. 모델 가중치·데이터셋·빌드 파일·대형 로그·프로파일 캡처·장비 접속 정보는 포함하지 않습니다.

논문 수치는 해당 Jetson과 요청 조건에서 측정한 결과입니다. OS 파일 페이지 캐시가 따뜻한 상태와 런타임 어댑터가 메모리에 없는 상태를 구분해야 합니다. 공개용 경로 정리 후 전체 GPU 실험을 다시 수행한 것은 아니며, 구체적인 검증 범위와 남은 재현 조건은 각 안내에 적었습니다.

## 결과 그래프 읽기

[메인 README](../README.md)에는 네 가지 구성의 처리량과 요청 완료 시간 그래프가
있습니다. [실험 문서](experiments.md)에는 Max TBT·TTFT 비교, 네 구성의 ablation,
로딩·출력 누적 병목 그래프를 배치했습니다. 기존 논문 CSV를 다시 그린 것이며,
이번 문서 정리 과정에서 새 GPU 측정을 수행한 결과는 아닙니다.

처리량은 높을수록, 완료 시간·TTFT·Max TBT는 낮을수록 좋습니다. 처리량 그래프의
오차 막대는 세 번 측정한 최솟값과 최댓값입니다. 나머지 CSV에는 분산 정보가 없어
임의로 오차 막대를 만들지 않았습니다. 그래프마다 반복 횟수와 캐시 조건을 확인하세요.

## 장비 없이 확인할 수 있는 항목

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
python experiments/plan_runs.py
```

위 명령은 소스·문서 검사, CPU 회귀 검사, 36회 처리량 실험의 수집 순서 확인입니다.
모델을 내려받거나 GPU 실험을 시작하지 않습니다. 실제 측정 준비에는
[재현 체크리스트](reproduction-checklist.md), 오류 해결에는
[문제 해결 안내](troubleshooting.md)를 참고하세요.

라우터 학습 실행 파일은 `training/adapter-router/train_router.py`입니다.
예전 파일명과의 대응은 [학습 안내](../training/README.md)에 보존했습니다.
