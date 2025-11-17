# app/settings.py
import os

OUT_W  = int(os.environ.get("OUT_W", "640"))
OUT_H  = int(os.environ.get("OUT_H", "640"))
FPS    = int(os.environ.get("FPS", "20"))
SECRET_TOKEN = os.environ.get("REMOTE_TOKEN") or "changeme"

K4A_DEVICE_INDEX = int(os.environ.get("K4A_DEVICE_INDEX", "0"))
DEPTH_MODE_STR   = os.environ.get("DEPTH_MODE", "WFOV_2X2BINNED").upper().strip()

# HRNet
HRNET_ROOT    = os.environ.get("HRNET_ROOT") or r"C:\Users\paddy\webrtc_remote_cam\deep-high-resolution-net.pytorch"
HRNET_CFG     = os.environ.get("HRNET_CFG")  or rf"{HRNET_ROOT}\experiments\coco\hrnet\w48_384x288_adam_lr1e-3.yaml"
HRNET_WEIGHTS = os.environ.get("HRNET_WEIGHTS") or rf"{HRNET_ROOT}\pose_hrnet_w48_384x288.pth"
