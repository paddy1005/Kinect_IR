# app/tracks.py
import os
import time
from fractions import Fraction
from typing import Optional, Tuple, List

import numpy as np
import cv2
import av
from aiortc import MediaStreamTrack

from .settings import OUT_W, OUT_H, FPS, K4A_DEVICE_INDEX, DEPTH_MODE_STR
from .kinect import AzureKinectIR, build_undistort_maps_from_device
from .pose import PoseEstimator, draw_skeleton


class KinectIRTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, fps: int, out_size: Tuple[int, int]):
        super().__init__()
        self.fps = max(1, int(fps))
        self.out_w, self.out_h = int(out_size[0]), int(out_size[1])
        self.frame_interval = 1.0 / self.fps
        self._last_ts = time.time()
        self._pts = 0

        # センサー（失敗時はNoneで黒画フォールバック）
        try:
            self.source = AzureKinectIR(device_index=K4A_DEVICE_INDEX, depth_mode_str=DEPTH_MODE_STR)
        except Exception as e:
            print("[KinectIRTrack] Azure Kinect init failed:", e)
            self.source = None

        # レンジ設定
        self.range_mode = "pct"   # "pct" or "abs"
        self.lo_pct, self.hi_pct = 1.0, 99.0
        self.lo_abs, self.hi_abs = 0, 65535

        # キャリブ
        self.map1 = self.map2 = None
        self.native_size: Optional[Tuple[int, int]] = None
        self.geometry_correct = True

        # Pose
        self.pose_enabled = False
        self.pose = PoseEstimator()

        # ROI（出力座標系）
        self.roi_poly: Optional[np.ndarray] = None  # shape (N,1,2) int32

        # 録画
        self.rec_enabled = False
        self.rec_writer: Optional[cv2.VideoWriter] = None
        self.rec_path: Optional[str] = None
        os.makedirs("recordings", exist_ok=True)

    # ========= control =========
    def reload_calib(self) -> bool:
        self.map1 = self.map2 = None
        return True

    def set_geom(self, enable: bool) -> bool:
        self.geometry_correct = bool(enable)
        self.map1 = self.map2 = None
        return True

    def set_range_pct(self, lo_pct: float, hi_pct: float):
        lo_pct = float(max(0.0, min(100.0, lo_pct)))
        hi_pct = float(max(0.0, min(100.0, hi_pct)))
        if hi_pct <= lo_pct:
            hi_pct = min(100.0, lo_pct + 0.1)
        self.range_mode = "pct"
        self.lo_pct, self.hi_pct = lo_pct, hi_pct

    def set_range_abs(self, lo_abs: int, hi_abs: int):
        lo_abs = int(max(0, min(65535, lo_abs)))
        hi_abs = int(max(0, min(65535, hi_abs)))
        if hi_abs <= lo_abs:
            hi_abs = min(65535, lo_abs + 1)
        self.range_mode = "abs"
        self.lo_abs, self.hi_abs = lo_abs, hi_abs

    def set_roi(self, points: List[List[float]]) -> bool:
        if not points or len(points) < 3:
            self.roi_poly = None
            return False
        pts = []
        for p in points:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                return False
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            x = max(0, min(self.out_w - 1, x))
            y = max(0, min(self.out_h - 1, y))
            pts.append([x, y])
        self.roi_poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        return True

    def clear_roi(self):
        self.roi_poly = None

    def start_recording(self) -> str:
        if self.rec_enabled and self.rec_writer is not None:
            return self.rec_path or ""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.rec_path = os.path.join("recordings", f"ir_{ts}.mp4")
        self.rec_writer = cv2.VideoWriter(self.rec_path, fourcc, float(self.fps), (self.out_w, self.out_h))
        self.rec_enabled = True
        return self.rec_path

    def stop_recording(self) -> Optional[str]:
        path = self.rec_path
        if self.rec_writer is not None:
            self.rec_writer.release()
        self.rec_writer = None
        self.rec_enabled = False
        return path

    def pose_start(self) -> bool:
        if self.pose and self.pose.is_ready:
            self.pose_enabled = True
            return True
        self.pose_enabled = False
        return False

    def pose_stop(self):
        self.pose_enabled = False

    # ========= helpers =========
    def _map_to_u8(self, ir16: np.ndarray) -> np.ndarray:
        ir = ir16.astype(np.float32)
        if self.range_mode == "abs":
            vmin, vmax = float(self.lo_abs), float(self.hi_abs)
        else:
            vmin = float(np.percentile(ir, self.lo_pct))
            vmax = float(np.percentile(ir, self.hi_pct))
        if vmax <= vmin:
            vmax = vmin + 1.0
        ir = np.clip(ir, vmin, vmax)
        ir = (ir - vmin) / (vmax - vmin) * 255.0
        return ir.astype(np.uint8)

    async def _sleep(self, sec: float):
        import asyncio
        await asyncio.sleep(sec)

    # ========= main loop =========
    async def recv(self):
        # FPS調整
        now = time.time()
        sleep_for = self.frame_interval - (now - self._last_ts)
        if sleep_for > 0:
            await self._sleep(sleep_for)
        self._last_ts = time.time()

        # IR読み出し（source None/失敗時は黒画）
        ir16 = None
        if self.source is not None:
            try:
                ir16 = self.source.read_ir16()
            except Exception as e:
                print("[KinectIRTrack] read_ir16 failed:", e)
                ir16 = None

        if ir16 is None:
            gray = np.zeros((self.out_h, self.out_w), dtype=np.uint8)
            base_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            h, w = ir16.shape
            if self.native_size is None:
                self.native_size = (w, h)
            if self.map1 is None or self.map2 is None:
                try:
                    self.map1, self.map2, _ = build_undistort_maps_from_device(
                        self.source.k4a, self.source.calib, self.native_size,
                        geometry_correct=self.geometry_correct
                    )
                except Exception:
                    self.map1 = self.map2 = None

            if self.map1 is not None and self.map2 is not None:
                ir16_ud = cv2.remap(
                    ir16, self.map1, self.map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                ir16_ud = ir16

            ir_u8 = self._map_to_u8(ir16_ud)
            gray = cv2.resize(ir_u8, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)
            base_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 録画（スケルトン描画前）
        if self.rec_enabled and self.rec_writer is not None:
            try:
                self.rec_writer.write(base_bgr)
            except Exception:
                pass

        # 描画ベース
        bgr = base_bgr.copy()

        # ROI枠表示
        if self.roi_poly is not None and len(self.roi_poly) >= 3:
            cv2.polylines(bgr, [self.roi_poly], isClosed=True, color=(255, 0, 0), thickness=2)

        # Pose
        if self.pose_enabled and self.pose and self.pose.is_ready:
            ih, iw = self.pose.input_size
            infer_input = bgr
            offset_xy = (0.0, 0.0)
            scale_xy = (1.0, 1.0)

            if self.roi_poly is not None and len(self.roi_poly) >= 3:
                x, y, w, h = cv2.boundingRect(self.roi_poly)
                m = max(4, int(0.05 * max(w, h)))
                x0 = max(0, x - m); y0 = max(0, y - m)
                x1 = min(self.out_w, x + w + m); y1 = min(self.out_h, y + h + m)
                crop = bgr[y0:y1, x0:x1]

                ch, cw = crop.shape[:2]
                r = min(iw / max(1, cw), ih / max(1, ch))
                nw, nh = max(1, int(cw * r)), max(1, int(ch * r))
                resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

                letter = np.zeros((ih, iw, 3), dtype=np.uint8)
                lx, ly = (iw - nw) // 2, (ih - nh) // 2
                letter[ly:ly + nh, lx:lx + nw] = resized
                infer_input = letter

                scale_xy = (cw / float(nw), ch / float(nh))
                offset_xy = (x0 - lx * scale_xy[0], y0 - ly * scale_xy[1])

            result = self.pose.infer(infer_input)
            if result is not None:
                kps, confs = result
                if self.roi_poly is not None and len(self.roi_poly) >= 3:
                    kps[:, 0] = kps[:, 0] * scale_xy[0] + offset_xy[0]
                    kps[:, 1] = kps[:, 1] * scale_xy[1] + offset_xy[1]

                if isinstance(kps, np.ndarray) and kps.shape[0] >= 17:
                    if self.roi_poly is not None and len(self.roi_poly) >= 3:
                        from cv2 import pointPolygonTest
                        roi32 = self.roi_poly.astype(np.float32)
                        mask_inside = []
                        for i in range(kps.shape[0]):
                            res = pointPolygonTest(roi32, (float(kps[i, 0]), float(kps[i, 1])), False)
                            mask_inside.append(res >= 0)
                        kps = kps[np.array(mask_inside, dtype=bool)]
                    if len(kps) > 0:
                        draw_skeleton(bgr, kps)

                mean_conf = float(np.mean(confs)) if confs is not None and len(confs) > 0 else 0.0
                cv2.rectangle(bgr, (8, 8), (170, 36), (0, 0, 0), -1)
                cv2.putText(bgr, f"conf: {mean_conf:.2f}", (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        vf = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p")
        vf.pts = getattr(self, "_pts", 0)
        vf.time_base = Fraction(1, self.fps)
        self._pts = getattr(self, "_pts", 0) + 1
        return vf
