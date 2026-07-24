"""
데모 영상 생성: 한 클립의 프레임 → mp4 → YOLO 탐지(+추적) 오버레이 영상.

박스·클래스·신뢰도가 그려진 영상이 runs/demo/<clip>/ 에 저장됨.
--track 을 주면 ByteTrack ID까지(프레임 간 같은 객체 추적).

사용(vast, venv 활성):
  # 사용 가능한 클립 목록 보기
  python scripts/make_demo.py --best <best.pt>
  # 특정 클립으로 데모 (군함 클립 추천, 예: I1_S0_C5_0008)
  python scripts/make_demo.py --best <best.pt> --clip I1_S0_C5_0008 --track

필요: ffmpeg (apt-get install -y ffmpeg), ultralytics(venv).
"""
import argparse, subprocess, tempfile, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="datasets/marine_frames")
    ap.add_argument("--clip", help="클립 접두 예: I1_S0_C5_0008. 생략 시 목록만 출력")
    ap.add_argument("--best", required=True, help="best.pt 경로")
    ap.add_argument("--fps", type=int, default=8, help="프레임이 성기므로 8 정도 권장")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--track", action="store_true", help="ByteTrack ID 표시(권장)")
    args = ap.parse_args()

    fdir = Path(args.frames_dir)
    if not args.clip:
        clips = sorted({p.stem[:-3] for p in fdir.glob("*.jpg")})
        print(f"{len(clips)} clips available. 예시 30개:")
        for c in clips[:30]:
            print("  ", c)
        print("\n군함 클립을 고르려면 clips.csv에서 has_warship=1 인 clip_id 참고.")
        return

    frames = sorted(fdir.glob(f"{args.clip}*.jpg"))
    if not frames:
        raise SystemExit(f"프레임 없음: {args.clip} (--frames-dir 확인)")
    print(f"{args.clip}: {len(frames)} frames @ {args.fps}fps")

    # 프레임을 연속 번호로 복사(결번·정렬 문제 회피) → ffmpeg
    tmp = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames):
        shutil.copy(f, tmp / f"{i:06d}.jpg")
    demo_in = Path(f"{args.out}_{args.clip}_in.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", str(tmp / "%06d.jpg"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(demo_in)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)

    from ultralytics import YOLO
    model = YOLO(args.best)
    kw = dict(source=str(demo_in), save=True, conf=args.conf,
              project="runs/demo", name=args.clip, exist_ok=True, line_width=2)
    if args.track:
        model.track(tracker="bytetrack.yaml", **kw)
    else:
        model.predict(**kw)
    print(f"\n입력 영상: {demo_in}")
    print(f"오버레이 영상: runs/demo/{args.clip}/  (Jupyter로 다운로드)")


if __name__ == "__main__":
    main()
