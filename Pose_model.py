# Pose_model.py
# 既存の app/pose.py (PoseEstimator) と同じ手法で HRNet 推論を実行し、
# baby.png に緑のキーポイント＋青の骨格線を描いて mark_baby.png を出力。

import sys
from pathlib import Path
import numpy as np
import cv2

# 既存の自作HRNet実装を利用
from app.pose import PoseEstimator, POSE_PAIRS

def draw_points_green_lines_blue(
    bgr: np.ndarray,
    kps: np.ndarray,
    conf: np.ndarray | None = None,
    thr: float = 0.2,
    point_radius: int = 3,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    点=緑, 線=青 で描画（app.pose と同じ関節順を想定）
    kps: (K,2) 画素座標, conf: (K,) or None
    """
    K = len(kps)
    if conf is None:
        vis = np.ones((K,), dtype=bool)
    else:
        conf = np.asarray(conf, dtype=np.float32)
        vis = conf >= float(thr)

    # NaN安全化
    kps_safe = np.array(kps, dtype=np.float32)
    nan_mask = np.isnan(kps_safe).any(axis=1)
    vis = vis & (~nan_mask)

    # 点（緑）
    for i, (x, y) in enumerate(kps_safe):
        if not vis[i]:
            continue
        cv2.circle(bgr, (int(x), int(y)), point_radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    # 線（青）
    for a, b in POSE_PAIRS:
        if a < K and b < K and vis[a] and vis[b]:
            xa, ya = int(kps_safe[a, 0]), int(kps_safe[a, 1])
            xb, yb = int(kps_safe[b, 0]), int(kps_safe[b, 1])
            cv2.line(bgr, (xa, ya), (xb, yb), (255, 0, 0), line_thickness, lineType=cv2.LINE_AA)

    return bgr


def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("baby.png")
    out_path = Path("mark_baby.png")

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} が見つかりません。")

    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"{in_path} を読み込めませんでした。")

    # あなたの HRNet 初期化（settings.py の HRNET_ROOT/CFG/WEIGHTS を使用）
    est = PoseEstimator()
    if not est.is_ready:
        raise RuntimeError(f"HRNet init error: {est.err_msg}")

    # 推論（あなたの heatmap→argmax 実装）
    result = est.infer(img, conf_thr=0.0)  # 返り値: (keypoints(K,2), conf(K,))
    if result is None:
        raise RuntimeError(f"Pose inference failed: {est.err_msg}")
    kps, conf = result  # (K,2), (K,)

    # 描画（緑の点＋青の線）
    out = draw_points_green_lines_blue(img.copy(), kps, conf=conf, thr=0.2)
    cv2.imwrite(str(out_path), out)
    print(f"保存しました: {out_path}  (K={len(kps)})")

if __name__ == "__main__":
    main()
