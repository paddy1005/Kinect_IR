# app/handlers.py
import os
import json
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from .settings import SECRET_TOKEN, OUT_W, OUT_H, FPS
from .tracks import KinectIRTrack
from .lux import current_lux, register_channel, unregister_channel  # LUX SoT

pcs = set()
ir_track: KinectIRTrack | None = None  # シングルトン運用

# ============ Utilities ============
def _unauthorized():
    return web.json_response({"error": "unauthorized"}, status=401)

# ============ HTTP ============
async def index(_request: web.Request):
    html_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    return web.Response(text=html, content_type="text/html")

async def snapshot(_request: web.Request):
    """現在のレンジ/キャリブ設定を反映したIR静止画（PNG, gray）"""
    global ir_track
    if ir_track is None:
        ir_track = KinectIRTrack(FPS, (OUT_W, OUT_H))

    ir16 = None
    try:
        ir16 = ir_track.source.read_ir16() if ir_track.source is not None else None
    except Exception:
        ir16 = None

    if ir16 is None:
        return web.Response(status=500, text="IR read failed")

    h, w = ir16.shape
    if ir_track.map1 is None or ir_track.map2 is None:
        ir_track.native_size = (w, h)
        try:
            from .kinect import build_undistort_maps_from_device
            ir_track.map1, ir_track.map2, _ = build_undistort_maps_from_device(
                ir_track.source.k4a,
                ir_track.source.calib,
                ir_track.native_size,
                geometry_correct=ir_track.geometry_correct,
            )
        except Exception:
            pass

    if ir_track.map1 is not None:
        ir16 = cv2.remap(
            ir16, ir_track.map1, ir_track.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    ir8 = ir_track._map_to_u8(ir16)
    gray = cv2.resize(ir8, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        return web.Response(status=500, text="encode failed")
    return web.Response(body=buf.tobytes(), content_type="image/png")

# ============ WebRTC ============
async def offer(request: web.Request):
    """
    WebRTC offer → answer。DataChannelで制御＆通知。
    """
    global ir_track

    params = await request.json()
    token = (params.get("token") or "").strip()
    if token != SECRET_TOKEN:
        return _unauthorized()

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        # DataChannel を lux 配信購読者に登録
        register_channel(channel)

        # 初回に現在値を即送（"--"を素早く消す）
        init_v = float(current_lux) if isinstance(current_lux, (int, float)) else 0.0
        try:
            channel.send(json.dumps({"lux": init_v}))
        except Exception:
            pass

        @channel.on("close")
        def _on_close():
            unregister_channel(channel)

        @channel.on("message")
        def on_message(message):
            global ir_track
            # JSON以外は無視
            try:
                data = json.loads(message)
            except Exception:
                return

            # トークン検証
            if (data.get("token") or "").strip() != SECRET_TOKEN:
                return

            cmd = data.get("cmd")
            if not cmd:
                return

            # --- lux 即時ポーリングは IR トラック不要
            if cmd == "lux_poll":
                v = float(current_lux) if isinstance(current_lux, (int, float)) else 0.0
                channel.send(json.dumps({"lux": v}))
                return

            # ここから下は IR トラックが必要なコマンド
            if ir_track is None:
                return

            if cmd == "calib_reload":
                ok = ir_track.reload_calib()
                channel.send(json.dumps({"ok": True, "calib_rebuilt_next": ok}))
                return

            if cmd == "set_geom":
                ok = ir_track.set_geom(bool(data.get("enable", True)))
                channel.send(json.dumps({"ok": True, "geometry_correct": ok}))
                return

            if cmd == "set_range_pct":
                try:
                    lo = float(data.get("lo_pct", ir_track.lo_pct))
                    hi = float(data.get("hi_pct", ir_track.hi_pct))
                    ir_track.set_range_pct(lo, hi)
                    channel.send(json.dumps({
                        "ok": True, "mode": "pct",
                        "lo_pct": ir_track.lo_pct, "hi_pct": ir_track.hi_pct,
                    }))
                except Exception as e:
                    channel.send(json.dumps({"ok": False, "error": str(e)}))
                return

            if cmd == "set_range_abs":
                try:
                    lo = int(data.get("lo_abs", ir_track.lo_abs))
                    hi = int(data.get("hi_abs", ir_track.hi_abs))
                    ir_track.set_range_abs(lo, hi)
                    channel.send(json.dumps({
                        "ok": True, "mode": "abs",
                        "lo_abs": ir_track.lo_abs, "hi_abs": ir_track.hi_abs,
                    }))
                except Exception as e:
                    channel.send(json.dumps({"ok": False, "error": str(e)}))
                return

            if cmd == "pose_start":
                on = ir_track.pose_start()
                err = getattr(ir_track.pose, "err_msg", "") if not on else ""
                channel.send(json.dumps({"ok": True, "pose_on": bool(on), "error": err}))
                return

            if cmd == "pose_stop":
                ir_track.pose_stop()
                channel.send(json.dumps({"ok": True, "pose_on": False}))
                return

            if cmd == "rec_start":
                path = ir_track.start_recording()
                url = f"/recordings/{os.path.basename(path)}" if path else ""
                channel.send(json.dumps({"ok": True, "rec_on": True, "rec_file": url}))
                return

            if cmd == "rec_stop":
                path = ir_track.stop_recording()
                url = f"/recordings/{os.path.basename(path)}" if path else ""
                channel.send(json.dumps({"ok": True, "rec_on": False, "rec_file": url}))
                return

            if cmd == "set_roi":
                pts = data.get("points")
                ok = ir_track.set_roi(pts if isinstance(pts, list) else [])
                channel.send(json.dumps({"ok": bool(ok), "roi_on": bool(ok)}))
                return

            if cmd == "clear_roi":
                ir_track.clear_roi()
                channel.send(json.dumps({"ok": True, "roi_on": False}))
                return

    # SDP交換
    offer_obj = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer_obj)

    if ir_track is None:
        ir_track = KinectIRTrack(FPS, (OUT_W, OUT_H))

    # ★ シンプルに addTrack でOK（クライアントは recvonly を宣言している想定）
    pc.addTrack(ir_track)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def health(_request: web.Request):
    return web.json_response({"ok": True})

async def on_shutdown(_app: web.Application):
    """RTC後始末＋録画停止"""
    global ir_track
    for pc in list(pcs):
        await pc.close()
    pcs.clear()

    if ir_track is not None and ir_track.rec_enabled:
        ir_track.stop_recording()
    ir_track = None
