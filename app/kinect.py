# app/kinect.py
import os
import numpy as np
import cv2

def _get_depth_calib_struct(pykinect, calib):
    if hasattr(calib, "camera_calibration"):
        return calib.camera_calibration[pykinect.K4A_CALIBRATION_TYPE_DEPTH]
    if hasattr(calib, "calibration") and hasattr(calib.calibration, "camera_calibration"):
        return calib.calibration.camera_calibration[pykinect.K4A_CALIBRATION_TYPE_DEPTH]
    if hasattr(calib, "depth_camera_calibration"):
        return calib.depth_camera_calibration
    return None

def _extract_intrinsics_and_distortion(pykinect, calib):
    cc = _get_depth_calib_struct(pykinect, calib)
    if cc is not None:
        p = cc.intrinsics.parameters.param
        fx, fy, cx, cy = float(p.fx), float(p.fy), float(p.cx), float(p.cy)
        dist = np.array([p.k1, p.k2, p.p1, p.p2, p.k3, p.k4, p.k5, p.k6], dtype=np.float64)
        return fx, fy, cx, cy, dist
    if hasattr(calib, "depth_params"):
        dp = calib.depth_params
        def getp(k, default=0.0):
            return float(dp.get(k, default)) if isinstance(dp, dict) else float(getattr(dp, k, default))
        fx, fy, cx, cy = getp("fx"), getp("fy"), getp("cx"), getp("cy")
        dist = np.array([getp("k1"), getp("k2"), getp("p1"), getp("p2"),
                         getp("k3"), getp("k4"), getp("k5"), getp("k6")], dtype=np.float64)
        return fx, fy, cx, cy, dist
    raise AttributeError("No depth intrinsics found in calibration")

def build_undistort_maps_from_device(pykinect, calib, size_wh, geometry_correct=True):
    w, h = size_wh
    fx, fy, cx, cy, dist = _extract_intrinsics_and_distortion(pykinect, calib)
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy,  cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    if geometry_correct:
        f = 0.5 * (fx + fy)
        newK = np.array([[f, 0.0, cx],
                         [0.0, f,  cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        newK = K.copy()
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1)
    return map1, map2, (fx, fy, cx, cy)

class AzureKinectIR:
    def __init__(self, device_index: int = 0, depth_mode_str: str = "WFOV_2X2BINNED"):
        import pykinect_azure as pykinect
        self.k4a = pykinect
        self.k4a.initialize_libraries()

        dm = depth_mode_str.upper()
        if dm == "WFOV_UNBINNED":
            depth_mode = self.k4a.K4A_DEPTH_MODE_WFOV_UNBINNED
            fps = self.k4a.K4A_FRAMES_PER_SECOND_15
        else:
            depth_mode = self.k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
            fps = self.k4a.K4A_FRAMES_PER_SECOND_30

        cfg = self.k4a.default_configuration
        cfg.color_resolution = self.k4a.K4A_COLOR_RESOLUTION_OFF
        cfg.depth_mode = depth_mode
        cfg.camera_fps = fps
        cfg.synchronized_images_only = False

        # 失敗時に sys.exit(1) を投げることがあるので捕捉して変換
        try:
            self.device = self.k4a.start_device(device_index=device_index, config=cfg)
        except SystemExit as e:
            raise RuntimeError(
                "Azure Kinect open failed (busy / not connected / driver). "
                "Close other apps using the device, replug USB/power, then retry."
            ) from e

        self.calib = self.device.get_calibration(cfg.depth_mode, cfg.color_resolution)
        self.depth_mode_str = dm

    def read_ir16(self):
        cap = self.device.update()
        ret, ir16 = cap.get_ir_image()
        return ir16 if ret and ir16 is not None else None
