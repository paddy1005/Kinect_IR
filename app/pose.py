# app/pose.py
import os, sys, cv2, numpy as np
from typing import Optional, Tuple
from .settings import HRNET_ROOT, HRNET_CFG, HRNET_WEIGHTS
import torch


COCO_FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

if os.path.isdir(HRNET_ROOT):
    sys.path.append(HRNET_ROOT)
lib_dir = os.path.join(HRNET_ROOT, "lib")
if os.path.isdir(lib_dir):
    sys.path.append(lib_dir)

POSE_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
]

def draw_skeleton(bgr: np.ndarray, kps: np.ndarray, conf: Optional[np.ndarray] = None, thr: float = 0.2):
    K = len(kps)
    vis = np.ones((K,), dtype=bool) if conf is None else (conf >= float(thr))
    for i, (x, y) in enumerate(kps):
        if not vis[i]: continue
        cv2.circle(bgr, (int(x), int(y)), 3, (0,255,0), -1)
    for a, b in POSE_PAIRS:
        if a < K and b < K and vis[a] and vis[b]:
            xa, ya = int(kps[a,0]), int(kps[a,1])
            xb, yb = int(kps[b,0]), int(kps[b,1])
            cv2.line(bgr, (xa,ya), (xb,yb), (0,255,0), 2)

class PoseEstimator:
    def __init__(self):
        self.is_ready = False
        self.device = None
        self.net = None
        self.input_size = (384, 288)  # (H, W)
        self.err_msg = ""

        if not (os.path.isfile(HRNET_CFG) and os.path.isfile(HRNET_WEIGHTS) and os.path.isdir(HRNET_ROOT)):
            self.err_msg = "HRNet paths are not ready (cfg/weights/root missing)"
            return

        try:
            import torch
            from types import SimpleNamespace
            from lib.config import cfg as hr_cfg
            from lib.config import update_config as hr_update_config
            from lib.models import pose_hrnet

            for d in (os.path.join(HRNET_ROOT,"models"),
                      os.path.join(HRNET_ROOT,"logs"),
                      os.path.join(HRNET_ROOT,"data")):
                os.makedirs(d, exist_ok=True)

            args = SimpleNamespace(cfg=HRNET_CFG, opts=[], modelDir="", logDir="", dataDir="", prevModelDir="")
            hr_update_config(hr_cfg, args)

            try:
                iw, ih = hr_cfg.MODEL.IMAGE_SIZE
                if isinstance(iw, (tuple, list)): iw, ih = iw
                self.input_size = (int(ih), int(iw))
            except Exception:
                pass

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = pose_hrnet.get_pose_net(hr_cfg, is_train=False)

            def _torch_load_safe(path, map_location):
                try:
                    return __import__("torch").load(path, map_location=map_location, weights_only=True)
                except TypeError:
                    return __import__("torch").load(path, map_location=map_location)

            state = _torch_load_safe(HRNET_WEIGHTS, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {k.replace("module.", ""): v for k, v in state.items()}
                missing, unexpected = self.net.load_state_dict(state, strict=False)
                if missing:   self.err_msg += f" [missing:{len(missing)}]"
                if unexpected:self.err_msg += f" [unexpected:{len(unexpected)}]"
            else:
                self.err_msg = "Invalid HRNet weights format"
                return

            self.net.to(self.device).eval()
            try:
                self.net.to(memory_format=__import__("torch").channels_last)
            except Exception:
                pass

            self.is_ready = True

        except Exception as e:
            self.err_msg = f"HRNet init failed: {e}"

    def _prep(self, bgr: np.ndarray, size_hw) -> "torch.Tensor":
        import torch
        ih, iw = size_hw
        resized = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].copy()
        tens = torch.from_numpy(rgb).to(self.device)
        tens = tens.permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0)
        try:
            tens = tens.to(memory_format=torch.channels_last, non_blocking=True)
        except Exception:
            pass
        return tens

    def infer(self, bgr: np.ndarray, conf_thr: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.is_ready:
            return None
        try:
            import torch
            H, W = bgr.shape[:2]
            ih, iw = self.input_size

            with torch.inference_mode():
                tens = self._prep(bgr, (ih, iw))
                out1 = self.net(tens)
                if isinstance(out1, (list, tuple)):
                    out1 = out1[-1]

                tens_flip = torch.flip(tens, dims=[3])
                out2 = self.net(tens_flip)
                if isinstance(out2, (list, tuple)):
                    out2 = out2[-1]

                if out1.shape[1] == out2.shape[1] == len(COCO_FLIP_INDEX):
                    out2 = out2[:, COCO_FLIP_INDEX, :, :]
                    out2 = torch.flip(out2, dims=[3])
                    out = 0.5 * (out1 + out2)
                else:
                    out = out1

                hm = torch.sigmoid(out)[0].detach().cpu().numpy()  # (K,h,w)

            K, hh, ww = hm.shape
            keypoints = np.zeros((K, 2), dtype=np.float32)
            confs = np.zeros((K,), dtype=np.float32)

            for k in range(K):
                m = hm[k]
                y, x = np.unravel_index(np.argmax(m), m.shape)
                c = float(m[y, x])
                if 1 <= x < ww - 1 and 1 <= y < hh - 1:
                    dx = m[y, x + 1] - m[y, x - 1]
                    dy = m[y + 1, x] - m[y - 1, x]
                    x = x + (0.25 if dx >= 0 else -0.25)
                    y = y + (0.25 if dy >= 0 else -0.25)
                X = (x / ww) * W
                Y = (y / hh) * H
                keypoints[k] = (X, Y)
                confs[k] = c

            if conf_thr > 0:
                mask = confs < conf_thr
                keypoints[mask] = np.nan

            return keypoints, confs

        except Exception as e:
            self.err_msg = f"Pose inference failed: {e}"
            return None
