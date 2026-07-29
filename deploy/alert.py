"""
경보 판정 — 프레임 단위 탐지를 '사건(event)'으로 바꾸는 층.

왜 필요한가: 배포에서 중요한 건 프레임 정확도가 아니라 **경보의 품질**이다.
W5 측정으로 빈 프레임 군함 오탐이 0.2%(conf 0.6) 인데, 그걸 프레임마다 그대로
경보로 올리면 30 FPS 스트림에서 분당 ~3.6회 헛경보가 된다. 반대로 진짜 군함은
연속 프레임에서 계속 잡히므로(데모 클립 95/95), **최근 M 프레임 중 N 프레임 규칙**
으로 단발 오탐만 골라 죽일 수 있다.

  규칙(기본값): 트랙별로 최근 10프레임 중 6프레임 이상 conf>=0.6 군함 → 경보 ON
                이후 15프레임 연속 미검출 → 경보 OFF
  → 단발/2연발 오탐은 절대 경보가 되지 않고, 진짜 표적은 약 6프레임(30fps에서 0.2초) 지연 후 경보.

트래커(ByteTrack)가 있으면 트랙 단위로, 없으면 프레임 단위(pseudo track)로 동작한다.
"""
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2


@dataclass
class AlertConfig:
    conf: float = 0.6          # 운영점 (W5 F1최적 0.604)
    n: int = 6                 # M 중 N 프레임 이상 히트하면 ON
    m: int = 10                # 판정 윈도
    off_patience: int = 15     # 연속 미검출 이 정도면 OFF
    target_class: int = 2      # warship
    class_names: tuple = ("fishing_boat", "merchant_ship", "warship")


@dataclass
class _Track:
    tid: int
    window: deque = field(default_factory=deque)   # 최근 m 프레임 히트 여부(1/0)
    on: bool = False
    misses: int = 0
    hits: int = 0
    conf_max: float = 0.0
    best_box: tuple = None
    first_frame: int = -1
    on_frame: int = -1
    on_ts: float = 0.0
    last_seen: int = -1


class AlertEngine:
    """dets 를 받아 경보 상태를 갱신하고 이벤트 리스트를 돌려준다."""

    def __init__(self, cfg: AlertConfig = None):
        self.cfg = cfg or AlertConfig()
        self.tracks = {}
        self.events = []

    @property
    def active_ids(self):
        return {t.tid for t in self.tracks.values() if t.on}

    @property
    def alarm(self):
        return bool(self.active_ids)

    def update(self, frame_idx, ts, dets):
        """dets: [{cls:int, conf:float, xyxy:(x1,y1,x2,y2), track_id:int|None}, ...]"""
        c = self.cfg
        hits = {}
        for d in dets:
            if d["cls"] != c.target_class or d["conf"] < c.conf:
                continue
            tid = d.get("track_id")
            tid = int(tid) if tid is not None else 0   # 트래커 없으면 프레임 단위 단일 트랙
            prev = hits.get(tid)
            if prev is None or d["conf"] > prev["conf"]:
                hits[tid] = d

        out = []
        # 이번 프레임에 히트한 트랙
        for tid, d in hits.items():
            t = self.tracks.get(tid)
            if t is None:
                t = self.tracks[tid] = _Track(tid=tid, window=deque(maxlen=c.m),
                                              first_frame=frame_idx)
            t.window.append(1)
            t.misses = 0
            t.hits += 1
            t.last_seen = frame_idx
            if d["conf"] > t.conf_max:
                t.conf_max, t.best_box = d["conf"], tuple(round(v, 1) for v in d["xyxy"])
            if not t.on and sum(t.window) >= c.n:
                t.on, t.on_frame, t.on_ts = True, frame_idx, ts
                out.append(dict(type="alert_on", track_id=tid, frame=frame_idx, ts=ts,
                                cls=c.class_names[c.target_class], conf=round(d["conf"], 3),
                                bbox=[round(v, 1) for v in d["xyxy"]],
                                delay_frames=frame_idx - t.first_frame))

        # 이번 프레임에 없던 트랙
        for tid, t in list(self.tracks.items()):
            if tid in hits:
                continue
            t.window.append(0)
            t.misses += 1
            if t.on and t.misses >= c.off_patience:
                t.on = False
                out.append(dict(type="alert_off", track_id=tid, frame=frame_idx, ts=ts,
                                duration_frames=frame_idx - t.on_frame,
                                duration_s=round(ts - t.on_ts, 2),
                                conf_max=round(t.conf_max, 3), hits=t.hits))
            if not t.on and t.misses > c.off_patience * 4:
                del self.tracks[tid]          # 오래된 트랙 정리(메모리)

        self.events.extend(out)
        return out

    def finalize(self, frame_idx, ts):
        """스트림 종료 시 열려 있는 경보를 닫는다(리포트 정합성)."""
        out = []
        for t in self.tracks.values():
            if t.on:
                t.on = False
                out.append(dict(type="alert_off", track_id=t.tid, frame=frame_idx, ts=ts,
                                duration_frames=frame_idx - t.on_frame,
                                duration_s=round(ts - t.on_ts, 2),
                                conf_max=round(t.conf_max, 3), hits=t.hits, reason="eos"))
        self.events.extend(out)
        return out

    def summary(self):
        on = [e for e in self.events if e["type"] == "alert_on"]
        off = [e for e in self.events if e["type"] == "alert_off"]
        return dict(alerts=len(on),
                    mean_delay_frames=round(sum(e["delay_frames"] for e in on) / len(on), 1) if on else None,
                    mean_duration_s=round(sum(e["duration_s"] for e in off) / len(off), 2) if off else None)


class EventLogger:
    """이벤트 JSONL + 경보 시작 프레임 스냅샷 저장."""

    def __init__(self, outdir, save_snapshots=True):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self.fh = self.path.open("a", encoding="utf-8")
        self.save_snapshots = save_snapshots
        self.session = time.strftime("%Y%m%d_%H%M%S")

    def write(self, events, frame=None):
        for e in events:
            e = dict(e, session=self.session, wall=time.strftime("%Y-%m-%d %H:%M:%S"))
            if self.save_snapshots and frame is not None and e["type"] == "alert_on":
                snap = self.dir / f"{self.session}_f{e['frame']:06d}_t{e['track_id']}.jpg"
                cv2.imwrite(str(snap), frame)
                e["snapshot"] = snap.name
            self.fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


if __name__ == "__main__":
    # 규칙 자체 검증: 단발 오탐은 경보가 되면 안 되고, 연속 표적은 경보가 되어야 한다.
    cfg = AlertConfig()
    eng = AlertEngine(cfg)
    box = (100.0, 100.0, 200.0, 150.0)
    for i in range(3):   # 단발성 오탐 3장(연속 아님)
        eng.update(i * 20, i * 0.66, [dict(cls=2, conf=0.9, xyxy=box, track_id=None)])
        eng.update(i * 20 + 1, i * 0.7, [])
    assert not eng.alarm and eng.summary()["alerts"] == 0, "단발 오탐이 경보가 됨"
    for i in range(100, 130):   # 진짜 표적 30프레임 연속
        ev = eng.update(i, i / 30, [dict(cls=2, conf=0.85, xyxy=box, track_id=7)])
    assert eng.alarm, "연속 표적이 경보가 안 됨"
    for i in range(130, 150):   # 사라짐
        eng.update(i, i / 30, [])
    assert not eng.alarm, "표적이 사라졌는데 경보가 안 꺼짐"
    print("alert 규칙 OK:", eng.summary())
