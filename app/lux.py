# app/lux.py
from __future__ import annotations
import asyncio, contextlib, json, os, shutil
from typing import Optional, Set
from aiohttp import web

# ====== 共有状態 ======
current_lux: float | None = None
_ctrl_channels: Set = set()  # DataChannel 購読者

def register_channel(ch) -> None:
    _ctrl_channels.add(ch)

def unregister_channel(ch) -> None:
    _ctrl_channels.discard(ch)

def get_lux() -> Optional[float]:
    return current_lux

# ====== 設定 ======
TD_USB_EXE    = os.environ.get("TD_USB_EXE") or r"C:\Users\paddy\td-usb-0.3.2-x64\td-usb.exe"
TD_USB_DEVICE = os.environ.get("TD_USB_DEVICE")
POLL_INTERVAL = float(os.environ.get("LUX_POLL_SEC", "1.0"))
EMA_ALPHA     = float(os.environ.get("LUX_EMA", "0.2"))
LUX_SCALE     = float(os.environ.get("LUX_SCALE", "1.0"))
LUX_OFFSET    = float(os.environ.get("LUX_OFFSET", "0.0"))
BASE_CMD = "iws660"

def _exe() -> Optional[str]:
    if TD_USB_EXE and os.path.isfile(TD_USB_EXE):
        return TD_USB_EXE
    return shutil.which("td-usb") or shutil.which("td-usb.exe")

async def _run(args: list[str], *, timeout: float = 5.0) -> str:
    exe = _exe()
    if not exe:
        return ""
    proc = await asyncio.create_subprocess_exec(
        exe, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return ""
    return out.decode("utf-8", "ignore").strip()

def _parse_number(text: str) -> Optional[float]:
    import re
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None

async def read_lux_once() -> float:
    if not _exe():
        print("[lux] td-usb not found; returning 0")
        return 0.0
    args = [BASE_CMD, "get", "--format=json"]
    if TD_USB_DEVICE:
        args = [BASE_CMD, "--device", TD_USB_DEVICE, "get", "--format=json"]
    out = await _run(args)
    if not out:
        return 0.0
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            for k in ("lux", "lx", "value", "Illuminance", "intensity"):
                if k in data:
                    return float(data[k])
    except Exception:
        pass
    v = _parse_number(out)
    return v if v is not None else 0.0

async def lux_reader_loop():
    global current_lux
    ema = None
    while True:
        try:
            v = await read_lux_once()
            v = (LUX_SCALE * v) + LUX_OFFSET
            ema = v if ema is None else (EMA_ALPHA * v + (1 - EMA_ALPHA) * ema)
            current_lux = float(ema)
            msg = json.dumps({"lux": current_lux})
            for ch in list(_ctrl_channels):
                try:
                    ch.send(msg)
                except Exception:
                    pass
        except Exception as e:
            print("[lux_reader_loop] error:", e)
        await asyncio.sleep(POLL_INTERVAL)

# aiohttp hooks
async def on_startup(app: web.Application):
    app["lux_task"] = asyncio.create_task(lux_reader_loop())

async def on_shutdown(app: web.Application):
    task = app.get("lux_task")
    if task:
        task.cancel()
        with contextlib.suppress(Exception):
            await task

async def handle_get_lux(_request: web.Request) -> web.Response:
    val = current_lux if current_lux is not None else 0.0
    return web.json_response({"lux": float(val)})
