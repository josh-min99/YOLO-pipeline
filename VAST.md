# vast 실행 런북 — 군함 탐지 (YOLO11)

> 전제: RTX 3090 인스턴스, tmux 안에서 실행(§5-4 SSH 끊김 대비). 끝나면 인스턴스 Destroy(§5-5).
> 함정 노트는 상위 `스모크테스트_유의사항.md` 참고.

## 0. 인스턴스 준비물
- **AI Hub 원본 zip** (Google Drive, 이미지 `TS_*.zip` 포함) 만 받으면 됨.
- 라벨(`labels_train.tgz`, 41,972 JSON)은 **repo에 포함**되어 clone하면 딸려옴.

## 1. 셋업
```bash
cd /workspace
git clone https://github.com/josh-min99/YOLO-pipeline.git
cd YOLO-pipeline
pip install -U ultralytics gdown
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # True 확인
```

## 2. 데이터 내려받기 & 풀기
```bash
# (a) 라벨 — repo에 있음. 풀기만.
mkdir -p labels_train && tar -xzf labels_train.tgz -C labels_train --strip-components=1
ls labels_train | wc -l          # 41972 확인

# (b) 이미지 zip 번들 (AI Hub 원본)
gdown 1gF4Bjgd7MORlyxnXtPl6SC1Lzl-3yNKS -O aihub.zip
mkdir -p aihub && unzip -q aihub.zip -d aihub
find aihub -name 'TS_*.zip'
#   막히면: gdown --fuzzy "https://drive.google.com/uc?id=1gF4Bjgd7MORlyxnXtPl6SC1Lzl-3yNKS" -O aihub.zip
#   (개방데이터에 Validation 폴더가 함께 있으면 그 TS_도 포함해 추출 가능)
```

## 3. 이미지 추출 (flat stem.jpg) + YOLO 데이터셋 빌드
```bash
python scripts/extract_images.py --zip-dir "$(dirname $(find aihub -name 'TS_*.zip' | head -1))" \
    --out datasets/marine_frames
# ~40k 프레임, 손상 ~1% 스킵 로그 확인

python scripts/json_to_yolo.py \
    --labels-dir labels_train \
    --images-dir datasets/marine_frames \
    --splits-dir splits \
    --out datasets/marine --link
# [train] frames=... boxes=... / [val] frames=... boxes=...  (img_missing=0 이어야 정상)
```
`configs/marine.yaml`의 `path`가 `../datasets/marine`이면 repo 루트 기준 맞음(필요시 절대경로로).

## 4. 학습 (tmux 안에서)
```bash
tmux new -s train
bash scripts/train.sh yolo11s.pt 1280 100 16
# Ctrl+B D 로 detach. 재접속: tmux attach -t train
```

## 5. 평가 / 확인
```bash
# best 가중치로 val 재평가 (군함=class 2 mAP 확인)
yolo detect val model=runs/detect/marine_*/weights/best.pt data=configs/marine.yaml imgsz=1280

# 몇 장 예측 시각화(bbox 눈으로 확인)
yolo detect predict model=runs/detect/marine_*/weights/best.pt \
    source=datasets/marine/images/val save=True conf=0.25
```

## 6. 결과 회수 후 Destroy
- `runs/detect/marine_*/weights/best.pt` 를 로컬/Drive로 백업.
- `results.csv`, PR/컨퓨전 이미지도 회수.
- 인스턴스 **Destroy** (과금 정지).

## 7. 🔴 재검증 (W5) — 첫 결과(mAP 0.98)는 leakage로 부풀려짐

> audit 결론: clip 분할은 깨끗하나 **같은 (날짜,지점) 세션이 train/val 양쪽**(val 군함세션 83% 겹침).
> 증거: mAP50이 **epoch 1에 이미 0.977**(1에폭에 val 해결 = 외운 걸 재인식). 정직한 일반화 숫자를 다시 내야 함.

### 7-1. 세션 단위 재분할 후 재학습
```bash
# (로컬에서 이미 생성됨: splits_session/, splits_session_spot/ — repo에 포함)
# 세션 분할로 YOLO 데이터셋 다시 빌드 (이미지는 이미 datasets/marine_frames에 있음)
python scripts/json_to_yolo.py --labels-dir labels_train --images-dir datasets/marine_frames \
    --splits-dir splits_session --out datasets/marine_session --link
sed -i "s#^path:.*#path: $(pwd)/datasets/marine_session#" configs/marine.yaml   # 또는 새 yaml
bash scripts/train.sh yolo11s.pt 1280 100 16
#  → 이번엔 mAP50이 epoch1에 천장 치는지 확인. 낮게/천천히 오르면 leakage 제거된 것.
#  더 엄격히 '새 지점' 보려면 splits_session_spot (spot 9,10,12 홀드아웃) 사용.
```

### 7-2. 군함 오탐율(가짜 경보) 측정 — 배포 핵심 지표
```bash
python scripts/eval_warship_fp.py --best runs/detect/runs/*/weights/best.pt \
    --data-root datasets/marine_session --conf-list 0.25,0.4,0.6 --split val
#  → 군함 없는 프레임에서 conf별 '가짜 군함 경보' 프레임비율. 운영 임계값(F1최적≈0.6) 기준으로 보고.
```

### 7-3. 운영점 보고 (mAP 단일 숫자 대신)
- `yolo val ... conf=0.6` 등 임계값별 P/R. F1최적 conf는 첫 run 기준 **0.604**(0.25 아님).
- BoxPR_curve.png: 군함은 recall ~0.95까지 precision ~0.95 유지, **recall>0.97에서 precision 절벽**.
- 위협탐지는 "recall 0.98일 때 precision", "가짜경보율"을 threshold별 표로 보고할 것.

## 다음(예정)
- frame-level AUC로 재구성 VAD(60%)와 같은 축에서 비교(프레임 점수 = 군함 conf max).
- 실시간 데모용 FPS 측정(yolo11n vs s, imgsz 스윕). W1처럼 batch=1 + sync.
- open-set 헤드(이상 정의 확정 후).
