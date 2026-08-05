# deploy/ — 엣지 배포 (TensorRT / Jetson)

> 학습된 `best.pt`(W5: 새 장소 군함 mAP50 0.979)를 **실제 장비에서 스트림을 받아 실시간 탐지·추적·경보**하는 실행 파이프라인.
> 전체 계획·단계 정의는 상위 `W6_엣지배포_계획.md`. 함정 노트는 `스모크테스트_유의사항.md`.

## 파일 지도
| 파일 | 역할 | 어디서 |
|---|---|---|
| `sources.py` | 입력원 추상화: 파일 / 프레임폴더 / USB / CSI / RTSP | 공통 |
| `alert.py` | N-of-M 히스테리시스 경보 + 이벤트 JSONL·스냅샷 | 공통 |
| `stream_infer.py` | **실행 엔트리**: 입력→추론→추적→경보→오버레이·리포트 | 공통 |
| `export_trt.py` | `.pt` → ONNX / TRT FP16 / TRT INT8 (ONNX만 있으면 trtexec 폴백) | x86(탐색) + 보드(최종) |
| `bench_engine.py` | 지연 p50/p95·FPS·**상대 연산비** CSV | x86 + 보드 |
| `eval_engine.py` | 엔진별 **정확도 패리티 표**(군함 mAP·운영점 P/R·오경보율) | x86 + 보드 |
| `eval_op.py` | 운영점 지표(recall@conf·오경보율)를 `predict`로 직접 — **rect 엔진은 이것만 됨** | x86 + 보드 |
| `make_board_bundle.py` | USB 이전 번들(가중치 + 동일 벤치마크 + 데모 + 실행지침) 생성 | 데이터 있는 호스트 |
| `jetson/setup.sh` | 보드 환경 점검(JetPack·TRT·jtop·전력모드) | 보드 |
| `jetson/power_bench.sh` | 전력모드별 × 열스로틀 후 지속 성능 | 보드 |

## 지금 상태 (P0 완료)
- 배관 검증됨: `stream_infer.py --stub` 로 입력→경보→오버레이→리포트 전 구간 로컬(CPU, ultralytics 없이) 통과.
- 경보 규칙 검증됨: `python deploy/alert.py` — 단발 오탐은 경보가 되지 않고, 연속 표적은 6프레임 후 경보 ON.
- 아직 안 한 것: 실제 엔진 변환·정확도 패리티·FPS (= P1, GPU 필요).

```bash
# 배관만 점검(GPU·ultralytics 불필요)
python deploy/stream_infer.py --source demo_I2_S0_C5_0079_in.avi --stub --limit 90 --save runs/deploy/stub.mp4
python deploy/alert.py
```

## P1 — x86(vast RTX)에서 설정 탐색
목표: **"정확도 하한을 통과하는 가장 싼 설정"** 과 **설정 간 상대 연산비**. 이 둘이 P2 기종 선정의 입력.

```bash
# 0) 데이터·가중치 준비는 VAST.md §1~3, §7 과 동일 (datasets/marine_session_spot 권장 = 새 장소 홀드아웃)
pip install -U ultralytics onnx onnxslim onnxruntime-gpu

# 1) 변환 (정사각/rect × FP16/INT8)
python deploy/export_trt.py --best <best.pt> --imgsz 1280     --formats onnx,fp16,int8 --data configs/marine.yaml
python deploy/export_trt.py --best <best.pt> --imgsz 736,1280 --formats onnx,fp16,int8 --data configs/marine.yaml

# 2) 정확도 패리티 (★ 이걸 안 재면 배포 근거 없음)
python deploy/eval_engine.py --weights <best.pt>,engines/best_1280_fp16.engine,engines/best_1280_int8.engine,engines/best_736x1280_fp16.engine,engines/best_736x1280_int8.engine \
    --data configs/marine.yaml --data-root datasets/marine_session_spot --imgsz 1280

# 3) 지연·상대 연산비
python deploy/bench_engine.py --weights <같은 목록> --imgsz-list 1280,736x1280,960,640 \
    --frames datasets/marine_session_spot/images/val --n 200

# 4) end-to-end (전처리·NMS·트래킹·오버레이까지 포함한 실제 FPS)
python deploy/stream_infer.py --source dir:datasets/marine_frames:I2_S0_C5_0079 \
    --model engines/best_736x1280_fp16.engine --imgsz 736,1280 --save runs/deploy/demo_trt.mp4
```

합격선(제안): 군함 mAP50 손실 **FP16 ≤1%p / INT8 ≤2%p**, **군함 recall@conf0.6 ≥ 0.95**, 빈프레임 오경보 ≤0.5%.
→ 통과한 설정 중 상대 연산비가 가장 낮은 것이 배포 설정.

## P3 — 보드 (Jetson Orin Nano Super)

### 0) 옮길 것 만들기 — 데이터셋이 있는 호스트에서
```bash
python deploy/make_board_bundle.py --root datasets/marine_session_spot \
    --weights best_spot.pt --demo demo_I2_S0_C5_0079_in.avi --out /media/usb/jetson_bundle
```
가중치 + **x86 P1과 동일한 벤치마크**(spot val 13,020장) + 데모 + `RUN_ON_BOARD.md`가 한 폴더로 나온다.
복사 전에 `frames=13020 / boxes=14272 / 어선7919·상선1910·군함4443`을 대조하므로, 분할이 어긋나면 그 자리에서 잡힌다.

### 1) 보드에서
```bash
bash setup_bundle.sh                             # 번들 위치로 data yaml path 고정 (1회)
bash deploy/jetson/setup.sh                      # 환경 점검 (컨테이너 권장 안내 포함)

# 🔴 x86에서 만든 .engine 은 안 돈다 → 보드에서 재빌드. **.pt 에서** 빌드할 것:
python3 deploy/export_trt.py --best /bundle/weights/best_spot.pt --imgsz 736,1280 --formats fp16
python3 deploy/export_trt.py --best /bundle/weights/best_spot.pt --imgsz 1280     --formats fp16
bash deploy/jetson/power_bench.sh engines/best_spot_736x1280_fp16.engine 736x1280 <프레임폴더> 0 1 2
```
> **왜 ONNX가 아니라 .pt인가:** ultralytics `Model.export()`는 첫 줄에서 `_check_is_pytorch_model()`을 호출해
> `.onnx`를 넣으면 `TypeError`로 죽는다. ONNX밖에 없으면 `export_trt.py`가 `trtexec`으로 우회하지만,
> 그렇게 만든 엔진엔 클래스명 메타데이터가 없어 라벨이 `class0/1/2`로 뜬다.
>
> **엔진을 두 개 만드는 이유:** `val()`이 rect를 지원하지 않아(§15-6) **mAP 축은 정사각(1280)**,
> **운영점·속도는 rect(736×1280)** 로 나눠 잰다.

## 산출물
- `runs/deploy/events.jsonl` — 경보 이벤트(+`*.jpg` 스냅샷)
- `runs/deploy/report_*.json` — 실행별 FPS·지연 p50/p95·입력 드롭률·경보 요약
- `results/deploy/bench_*.csv`, `results/deploy/parity_*.csv` — 표에 그대로 들어가는 숫자

## 잊기 쉬운 것 3가지
1. **엔진 이식 불가** — 아키텍처·TRT 버전 종속. 보드로는 **`.pt`** 를 옮겨 거기서 빌드한다(ONNX는 ultralytics export가 거부, §15-8).
2. **INT8 캘리브레이션은 train 스플릿으로** — ultralytics 기본값은 `data:` yaml의 val이라 그대로 쓰면 평가셋 누수. `export_trt.py`가 임시 yaml로 우회함.
3. **conf=0.6 val의 mAP는 잘린 값** — 표준 mAP(conf 0.001)와 운영점 P/R을 열로 분리해 보고.
