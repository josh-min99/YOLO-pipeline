"""웹 모니터링 뷰 — 브라우저에서 실시간 화면 + 경보 상태를 본다.

  python3 deploy/stream_infer.py --source usb:0 --gst --live-direct --web 8080 ...
  → PC 브라우저에서 http://<보드IP>:8080

왜 웹인가: 보드에 모니터를 붙여두지 않아도 되고, 원격에서 감시할 수 있으며,
발표 때 노트북 브라우저로 띄우면 그대로 시연이 된다. 표준 라이브러리만 쓴다(의존성 없음).

🔴 설계 원칙 — 추론 루프를 막지 않는다.
   publish() 는 프레임 **참조만** 바꾸고 즉시 반환한다. JPEG 인코딩은 접속한 브라우저마다
   별도 스레드에서 돈다. 아무도 안 보고 있으면 인코딩 비용이 0이다.
   (Orin Nano 에는 HW 인코더가 없어서 JPEG 인코딩도 CPU 다 — 추론 루프에 넣으면 안 된다.)
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>해상 감시 모니터</title><style>
*{box-sizing:border-box}
body{margin:0;background:#141414;color:#eee;font-family:'Malgun Gothic',system-ui,sans-serif}
.wrap{display:flex;gap:14px;padding:14px;height:100vh}
.left{flex:1;min-width:0;display:flex;flex-direction:column;gap:10px}
.right{width:320px;display:flex;flex-direction:column;gap:10px}
img{width:100%;border-radius:8px;display:block;background:#000}
.card{background:#1e1e1e;border-radius:8px;padding:12px 14px}
h1{font-size:17px;margin:0 0 2px}
.sub{color:#888;font-size:12px}
.banner{padding:14px;border-radius:8px;text-align:center;font-size:22px;font-weight:700;
        letter-spacing:.5px;transition:background .15s}
.ok{background:#20402c;color:#7ee2a8}
.alarm{background:#8c1d17;color:#fff;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:5px 0;border-bottom:1px solid #2c2c2c}
td:last-child{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.ev{font-size:12px;max-height:230px;overflow:auto;margin:0;padding:0;list-style:none}
.ev li{padding:6px 8px;border-left:3px solid #8c1d17;background:#242424;margin-bottom:5px;
       border-radius:0 4px 4px 0}
.ev li.off{border-left-color:#555;color:#999}
.t{color:#888}
</style></head><body><div class="wrap">
<div class="left">
  <div class="card"><h1>해상 감시 모니터</h1>
    <div class="sub" id="src">-</div></div>
  <img src="/stream" alt="live">
</div>
<div class="right">
  <div id="banner" class="banner ok">정상</div>
  <div class="card"><table>
    <tr><td>처리 FPS</td><td id="fps">-</td></tr>
    <tr><td>지연 (p50)</td><td id="lat">-</td></tr>
    <tr><td>프레임</td><td id="frames">-</td></tr>
    <tr><td>입력 드롭</td><td id="drop">-</td></tr>
    <tr><td>활성 트랙</td><td id="tracks">-</td></tr>
    <tr><td>누적 경보</td><td id="alerts">-</td></tr>
  </table></div>
  <div class="card"><div class="sub" style="margin-bottom:8px">최근 이벤트</div>
    <ul class="ev" id="events"></ul></div>
</div></div>
<script>
async function tick(){
  try{
    const s = await (await fetch('/status')).json();
    document.getElementById('src').textContent = s.source || '-';
    document.getElementById('fps').textContent = s.fps.toFixed(1);
    document.getElementById('lat').textContent = s.p50.toFixed(1)+' ms';
    document.getElementById('frames').textContent = s.frames;
    document.getElementById('drop').textContent = (s.drop*100).toFixed(1)+' %';
    document.getElementById('tracks').textContent = s.tracks;
    document.getElementById('alerts').textContent = s.alerts;
    const b = document.getElementById('banner');
    b.className = 'banner ' + (s.alarm ? 'alarm' : 'ok');
    b.textContent = s.alarm ? '⚠ 군함 탐지' : '정상';
    document.getElementById('events').innerHTML = s.events.map(e =>
      `<li class="${e.type==='alert_off'?'off':''}">
         <span class="t">f${e.frame}</span>
         ${e.type==='alert_on'?'경보 발생':'경보 해제'} · 트랙 ${e.track_id}
         ${e.conf?('· conf '+e.conf.toFixed(2)):''}</li>`).join('');
  }catch(err){}
}
setInterval(tick, 500); tick();
</script></body></html>"""


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.status = dict(fps=0.0, p50=0.0, frames=0, drop=0.0, tracks=0,
                           alerts=0, alarm=False, events=[], source="")


class _Handler(BaseHTTPRequestHandler):
    state = None
    jpeg_quality = 70
    max_width = 1280

    def log_message(self, *a):
        pass                                   # 접속 로그로 콘솔을 더럽히지 않는다

    def do_GET(self):
        if self.path.startswith("/status"):
            with self.state.lock:
                body = json.dumps(self.state.status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/stream"):
            self._stream()
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                with self.state.lock:
                    fr, seq = self.state.frame, self.state.seq
                if fr is None or seq == last:
                    time.sleep(0.005)          # 새 프레임이 없으면 다시 보내지 않는다
                    continue
                last = seq
                if fr.shape[1] > self.max_width:      # 전송량 줄이기(추론과 무관)
                    s = self.max_width / fr.shape[1]
                    fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", fr,
                                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    continue
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(buf)).encode() + b"\r\n\r\n" + buf.tobytes()
                                 + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                                # 브라우저가 탭을 닫은 것 — 정상


class WebView:
    """추론 루프에서 publish() 만 호출하면 된다. 인코딩은 브라우저 스레드가 한다."""

    def __init__(self, port=8080, quality=70, max_width=1280, source=""):
        self.state = _State()
        self.state.status["source"] = source
        _Handler.state = self.state
        _Handler.jpeg_quality = quality
        _Handler.max_width = max_width
        self.srv = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        print(f"[web] http://<보드IP>:{port}  (같은 네트워크의 브라우저에서 접속)")

    def publish(self, frame, **status):
        """프레임 참조만 교체하고 즉시 반환 — 추론 루프를 막지 않는다."""
        with self.state.lock:
            if frame is not None:
                self.state.frame = frame
                self.state.seq += 1
            self.state.status.update(status)

    def add_events(self, events, keep=12):
        with self.state.lock:
            ev = self.state.status["events"]
            for e in events:
                ev.insert(0, dict(type=e.get("type"), frame=e.get("frame"),
                                  track_id=e.get("track_id"), conf=e.get("conf")))
            del ev[keep:]

    def close(self):
        self.srv.shutdown()
