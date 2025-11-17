# app/main.py
import os, logging
from aiohttp import web
from .handlers import index, snapshot, offer, health, on_shutdown
from .lux import on_startup as lux_startup, on_shutdown as lux_shutdown, handle_get_lux

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc_kinect_ir")

def create_app():
    app = web.Application()

    # Routes
    app.router.add_get("/", index)
    app.router.add_get("/snapshot", snapshot)
    app.router.add_post("/offer", offer)
    app.router.add_get("/health", health)
    app.router.add_get("/lux", handle_get_lux)  # LUXはlux.pyのハンドラに統一

    # Static
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    rec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
    os.makedirs(rec_dir, exist_ok=True)

    app.router.add_static("/recordings", path=rec_dir, show_index=True)
    app.router.add_static("/static", path=static_dir, show_index=False)

    # Lifecycle
    app.on_startup.append(lux_startup)
    app.on_shutdown.append(lux_shutdown)
    app.on_shutdown.append(on_shutdown)  # RTC後始末

    return app

def main():
    app = create_app()
    port = 8080
    logger.info(f"Serving on http://127.0.0.1:{port}  (LAN: http://<IPv4>:{port})")
    web.run_app(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
