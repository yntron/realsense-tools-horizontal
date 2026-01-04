"""
Jump Analyzer

.bagファイルからYOLOv8-Poseを使用して3D姿勢推定を行い、
ジャンプの高さ・距離・軌跡を測定するメインスクリプト
"""

import argparse
import json
import csv
import os
import sys
import time
from pathlib import Path
from collections import deque

try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # GUI不要のバックエンドを使用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォントの設定を試行
try:
    # よく使われる日本語フォントを探す
    jp_fonts = [
        "Hiragino Sans",           # macOS
        "Hiragino Kaku Gothic Pro",  # macOS
        "Hiragino Kaku Gothic ProN",  # macOS
        "Noto Sans CJK JP",
        "TakaoGothic",
        "IPAexGothic",
        "IPAPGothic",
        "VL PGothic",
        "Yu Gothic",               # Windows
    ]
    for font_name in jp_fonts:
        try:
            font = fm.findfont(fm.FontProperties(family=font_name))
            if font and "DejaVu" not in font:  # デフォルトフォント以外が見つかった場合
                plt.rcParams["font.family"] = font_name
                break
        except:
            continue
    # フォントが見つからない場合は、英語ラベルを使用する設定にする
    current_font = plt.rcParams.get("font.family", ["DejaVu Sans"])
    if isinstance(current_font, list):
        current_font = current_font[0] if current_font else "DejaVu Sans"
    if "DejaVu" in current_font:
        USE_JAPANESE_LABELS = False
    else:
        USE_JAPANESE_LABELS = True
except:
    USE_JAPANESE_LABELS = False

from src.realsense_utils import BagFileReader, CUPY_AVAILABLE
from src.yolov8_pose_3d import YOLOv8PoseDetector, COCO_KEYPOINTS
from src.jump_detector import JumpDetector
from src.visualizer import JumpVisualizer, create_3d_keypoint_animation
from src.keypoint_smoother import KeypointSmoother
from src.kalman_filter_3d import KalmanSmoother
from src.floor_detector import FloorDetector

import pyrealsense2 as rs


# ===== デュアルカメラ同期 =====
class FramePairer:
    """最近傍時刻でフレームをペアリング"""

    def __init__(self, tol_ms=25.0):
        self.buf0, self.buf1 = deque(maxlen=5), deque(maxlen=5)
        self.tol = tol_ms

    @staticmethod
    def ts(frame):
        """フレームのタイムスタンプを取得（タプルの場合は最初の要素から取得）"""
        if frame is None:
            return None
        if isinstance(frame, tuple):
            # (color_frame, depth_frame) タプルの場合
            return frame[0].get_timestamp() if frame[0] else None
        return frame.get_timestamp()

    def put_and_match(self, f0=None, f1=None):
        """フレームを追加して、ペアが揃ったら返す"""
        if f0 is not None:
            self.buf0.append(f0)
        if f1 is not None:
            self.buf1.append(f1)

        if not self.buf0 or not self.buf1:
            return None

        f0c = self.buf0[0]
        t0 = self.ts(f0c)
        
        if t0 is None:
            self.buf0.popleft()
            return None

        best = None
        best_dt = 1e9
        best_i = -1

        for i, g in enumerate(self.buf1):
            t1 = self.ts(g)
            if t1 is None:
                continue
            dt = abs(t1 - t0)
            if dt < best_dt:
                best_dt, best, best_i = dt, g, i

        if best_dt <= self.tol and best is not None:
            self.buf0.popleft()
            del self.buf1[best_i]
            return f0c, best, best_dt

        # 古い方を進める
        t0_val = self.ts(self.buf0[0])
        t1_val = self.ts(self.buf1[0])
        if t0_val is not None and t1_val is not None:
            if t0_val < t1_val:
                self.buf0.popleft()
            else:
                self.buf1.popleft()
        elif t0_val is None:
            self.buf0.popleft()
        elif t1_val is None:
            self.buf1.popleft()

        return None


def transform_point_to_cam0_coords(point_3d, T1_to_0):
    """
    cam1座標系の3D点をcam0座標系に変換
    
    Args:
        point_3d: (x, y, z) タプルまたはリスト
        T1_to_0: 4x4変換行列（numpy配列）
    
    Returns:
        (x, y, z) タプル（変換後）
    """
    if point_3d is None or None in point_3d:
        return None
    
    import numpy as np
    # 同次座標に変換
    p = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    # 変換
    p_transformed = T1_to_0 @ p
    return (float(p_transformed[0]), float(p_transformed[1]), float(p_transformed[2]))


def merge_keypoints(keypoints_3d_cam0, keypoints_3d_cam1, confidence_2d_cam0=None, confidence_2d_cam1=None):
    """
    両カメラのキーポイントを統合（信頼度の高い方を選択）
    
    Args:
        keypoints_3d_cam0: cam0の3Dキーポイント辞書
        keypoints_3d_cam1: cam1の3Dキーポイント辞書（cam0座標系に変換済み）
        confidence_2d_cam0: cam0の2D信頼度リスト（オプション）
        confidence_2d_cam1: cam1の2D信頼度リスト（オプション）
    
    Returns:
        統合された3Dキーポイント辞書
    """
    merged = {}
    
    for kp_name in COCO_KEYPOINTS:
        kp0 = keypoints_3d_cam0.get(kp_name)
        kp1 = keypoints_3d_cam1.get(kp_name)
        
        # 両方有効な場合は信頼度で選択
        if kp0 and None not in kp0 and kp1 and None not in kp1:
            # 信頼度が利用可能な場合はそれを使用
            if confidence_2d_cam0 and confidence_2d_cam1:
                idx = COCO_KEYPOINTS.index(kp_name)
                conf0 = confidence_2d_cam0[idx][2] if idx < len(confidence_2d_cam0) and confidence_2d_cam0[idx] else 0.0
                conf1 = confidence_2d_cam1[idx][2] if idx < len(confidence_2d_cam1) and confidence_2d_cam1[idx] else 0.0
                # より高い信頼度のものを選択
                merged[kp_name] = kp0 if conf0 >= conf1 else kp1
            else:
                # デフォルトではcam0を優先
                merged[kp_name] = kp0
        elif kp0 and None not in kp0:
            merged[kp_name] = kp0
        elif kp1 and None not in kp1:
            merged[kp_name] = kp1
        else:
            merged[kp_name] = (None, None, None)
    
    return merged

# CuPyのインポート（CUDA高速化用）
if CUPY_AVAILABLE:
    import cupy as cp

    print("CuPy available: Using CUDA acceleration for image processing")
else:
    print("CuPy not available: Using NumPy (CPU mode)")
    print(
        "  Note: Install CuPy with 'pip install cupy-cuda11x' or 'pip install cupy-cuda12x' for GPU acceleration"
    )


def resize_image(image, new_width, new_height):
    """
    画像をリサイズ

    Args:
        image: 入力画像（NumPy配列）
        new_width: 新しい幅
        new_height: 新しい高さ

    Returns:
        リサイズされた画像（NumPy配列）
    """
    return cv2.resize(image, (new_width, new_height))


def convert_keypoints_to_dict(keypoints_2d, keypoints_3d):
    """
    keypointsを辞書形式に変換

    Args:
        keypoints_2d: 2D keypointsのリスト [(x, y, confidence), ...]
        keypoints_3d: 3D keypointsの辞書 {keypoint_name: (x, y, z), ...}

    Returns:
        dict: keypointsデータの辞書
    """
    result = {}

    for i, keypoint_name in enumerate(COCO_KEYPOINTS):
        if i < len(keypoints_2d):
            kp_2d = keypoints_2d[i]
            kp_3d = keypoints_3d.get(keypoint_name)

            result[keypoint_name] = {
                "2d": {
                    "x": float(kp_2d[0]) if kp_2d[0] is not None else None,
                    "y": float(kp_2d[1]) if kp_2d[1] is not None else None,
                    "confidence": float(kp_2d[2]),
                },
                "3d": {
                    "x": float(kp_3d[0]) if kp_3d and kp_3d[0] is not None else None,
                    "y": float(kp_3d[1]) if kp_3d and kp_3d[1] is not None else None,
                    "z": float(kp_3d[2]) if kp_3d and kp_3d[2] is not None else None,
                },
            }
        else:
            result[keypoint_name] = {
                "2d": {"x": None, "y": None, "confidence": 0.0},
                "3d": {"x": None, "y": None, "z": None},
            }

    return result


def save_json(data, output_path):
    """JSONファイルに保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {output_path}")


def save_csv(statistics, trajectory, output_path):
    """CSVファイルに保存"""
    # 統計情報をCSVに保存
    stats_path = output_path.replace(".csv", "_statistics.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Jumps", statistics.get("total_jumps", 0)])
        writer.writerow(["Vertical Jumps", statistics.get("vertical_jumps", 0)])
        writer.writerow(["Horizontal Jumps", statistics.get("horizontal_jumps", 0)])
        writer.writerow(["Max Height (cm)", statistics.get("max_height", 0) * 100])
        writer.writerow(["Max Distance (cm)", statistics.get("max_distance", 0) * 100])
        writer.writerow(["Avg Height (cm)", statistics.get("avg_height", 0) * 100])
        writer.writerow(["Avg Distance (cm)", statistics.get("avg_distance", 0) * 100])
    print(f"Statistics CSV saved to: {stats_path}")

    # ジャンプ詳細をCSVに保存
    if statistics.get("jumps"):
        jumps_path = output_path.replace(".csv", "_jumps.csv")
        with open(jumps_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 床検出使用時は滞空時間も出力
            headers = [
                "Jump #",
                "Type",
                "Height (cm)",
                "Distance (cm)",
                "Start Frame",
                "Takeoff Frame",
                "End Frame",
                "Duration (frames)",
            ]
            if any("air_time" in jump for jump in statistics["jumps"]):
                headers.append("Air Time (s)")
            writer.writerow(headers)

            for i, jump in enumerate(statistics["jumps"], 1):
                row = [
                    i,
                    jump["jump_type"],
                    jump["height"] * 100,
                    jump["distance"] * 100,
                    jump["frame_start"],
                    jump.get("frame_takeoff", jump["frame_start"]),
                    jump["frame_end"],
                    jump["frame_end"] - jump["frame_start"],
                ]
                if "air_time" in jump:
                    row.append(jump["air_time"])
                writer.writerow(row)
        print(f"Jumps CSV saved to: {jumps_path}")

    # 軌跡データをCSVに保存
    if trajectory:
        trajectory_path = output_path.replace(".csv", "_trajectory.csv")
        with open(trajectory_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Timestamp", "X (m)", "Y (m)", "Z (m)"])
            for point in trajectory:
                pos = point.get("position", (None, None, None))
                writer.writerow(
                    [
                        point.get("frame", ""),
                        point.get("timestamp", ""),
                        pos[0] if pos[0] is not None else "",
                        pos[1] if pos[1] is not None else "",
                        pos[2] if pos[2] is not None else "",
                    ]
                )
        print(f"Trajectory CSV saved to: {trajectory_path}")


def plot_keypoint_coordinate_timeline(all_frames_data, output_dir, floor_detector=None):
    """
    全キーポイントのX, Y, Z座標を時系列でプロット（3つのグラフを生成）

    Args:
        all_frames_data: 全フレームデータのリスト
        output_dir: 出力ディレクトリ
        floor_detector: 床検出器（Noneの場合、カメラ座標系のYを使用）
    """
    if not all_frames_data:
        print("Warning: No frame data available for coordinate timeline plot")
        return

    # タイムスタンプと各キーポイントのX, Y, Z座標を収集
    timestamps = []
    keypoint_x = {kp_name: [] for kp_name in COCO_KEYPOINTS}
    keypoint_y = {kp_name: [] for kp_name in COCO_KEYPOINTS}
    keypoint_z = {kp_name: [] for kp_name in COCO_KEYPOINTS}

    # 最初のタイムスタンプを基準に（秒単位に変換）
    first_timestamp = None

    for frame_data in all_frames_data:
        timestamp = frame_data.get("timestamp")
        if timestamp is None:
            continue

        # 最初のタイムスタンプを記録
        if first_timestamp is None:
            first_timestamp = timestamp

        # 経過時間を計算（ミリ秒か秒かを判定）
        if first_timestamp > 1000000000:  # ミリ秒単位と判定
            elapsed_time = (timestamp - first_timestamp) / 1000.0  # 秒に変換
        else:
            elapsed_time = timestamp - first_timestamp

        timestamps.append(elapsed_time)

        # 各キーポイントのX, Y, Z座標を取得
        keypoints = frame_data.get("keypoints", {})
        for kp_name in COCO_KEYPOINTS:
            kp_data = keypoints.get(kp_name, {})
            kp_3d = kp_data.get("3d", {})

            # X座標
            x = kp_3d.get("x") if kp_3d.get("x") is not None else None
            keypoint_x[kp_name].append(x)

            # Y座標（床からの距離が利用可能な場合はそれを使用）
            if (
                floor_detector
                and "distance_to_floor" in kp_data
                and kp_data["distance_to_floor"] is not None
            ):
                y = kp_data["distance_to_floor"]
            elif kp_3d.get("y") is not None:
                y = kp_3d["y"]
            else:
                y = None
            keypoint_y[kp_name].append(y)

            # Z座標
            z = kp_3d.get("z") if kp_3d.get("z") is not None else None
            keypoint_z[kp_name].append(z)

    if not timestamps:
        print("Warning: No valid timestamps found for coordinate timeline plot")
        return

    # カラーマップを準備（キーポイントごとに異なる色）
    try:
        # matplotlib 3.7以降の新しい方法
        from matplotlib import colormaps

        colors = colormaps.get_cmap("tab20")
    except (AttributeError, ImportError):
        # フォールバック
        try:
            colors = plt.get_cmap("tab20")
        except:
            # さらにフォールバック（古い方法）
            from matplotlib import cm

            colors = cm.get_cmap("tab20")

    # X座標のグラフ
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, kp_name in enumerate(COCO_KEYPOINTS):
        x_values = keypoint_x[kp_name]
        valid_data = [(t, x) for t, x in zip(timestamps, x_values) if x is not None]
        if valid_data:
            valid_times, valid_x = zip(*valid_data)
            ax.plot(
                valid_times,
                valid_x,
                label=kp_name,
                color=colors(i),
                alpha=0.7,
                linewidth=1.5,
            )
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("時間 (秒)", fontsize=12, fontweight="bold")
        ax.set_ylabel("X座標 (m)", fontsize=12, fontweight="bold")
        ax.set_title("全キーポイントのX座標（時系列）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel("X coordinate (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "All Keypoints X Coordinate Timeline", fontsize=14, fontweight="bold"
        )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    x_path = output_dir / "keypoint_x_timeline.png"
    plt.savefig(str(x_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Keypoint X coordinate timeline plot saved to: {x_path}")

    # Y座標のグラフ（高さ）
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, kp_name in enumerate(COCO_KEYPOINTS):
        y_values = keypoint_y[kp_name]
        valid_data = [(t, y) for t, y in zip(timestamps, y_values) if y is not None]
        if valid_data:
            valid_times, valid_y = zip(*valid_data)
            ax.plot(
                valid_times,
                valid_y,
                label=kp_name,
                color=colors(i),
                alpha=0.7,
                linewidth=1.5,
            )
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("時間 (秒)", fontsize=12, fontweight="bold")
        if floor_detector:
            ax.set_ylabel("床からの距離 (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "全キーポイントの床からの距離（時系列）", fontsize=14, fontweight="bold"
            )
        else:
            ax.set_ylabel("Y座標 (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "全キーポイントのY座標（時系列）", fontsize=14, fontweight="bold"
            )
    else:
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        if floor_detector:
            ax.set_ylabel("Distance from Floor (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "All Keypoints Distance from Floor Timeline",
                fontsize=14,
                fontweight="bold",
            )
        else:
            ax.set_ylabel("Y coordinate (m)", fontsize=12, fontweight="bold")
            ax.set_title(
                "All Keypoints Y Coordinate Timeline", fontsize=14, fontweight="bold"
            )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    y_path = output_dir / "keypoint_y_timeline.png"
    plt.savefig(str(y_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Keypoint Y coordinate timeline plot saved to: {y_path}")

    # Z座標のグラフ
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, kp_name in enumerate(COCO_KEYPOINTS):
        z_values = keypoint_z[kp_name]
        valid_data = [(t, z) for t, z in zip(timestamps, z_values) if z is not None]
        if valid_data:
            valid_times, valid_z = zip(*valid_data)
            ax.plot(
                valid_times,
                valid_z,
                label=kp_name,
                color=colors(i),
                alpha=0.7,
                linewidth=1.5,
            )
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("時間 (秒)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z座標 (m)", fontsize=12, fontweight="bold")
        ax.set_title("全キーポイントのZ座標（時系列）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z coordinate (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            "All Keypoints Z Coordinate Timeline", fontsize=14, fontweight="bold"
        )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    z_path = output_dir / "keypoint_z_timeline.png"
    plt.savefig(str(z_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Keypoint Z coordinate timeline plot saved to: {z_path}")


def plot_jump_trajectory(trajectory, statistics, output_dir, floor_detector=None):
    """
    ジャンプ軌跡を可視化（水平面と高さ-時間グラフ）

    Args:
        trajectory: 軌跡データのリスト
        statistics: 統計情報（ジャンプ検出結果を含む）
        output_dir: 出力ディレクトリ
        floor_detector: 床検出器（オプション）
    """
    if not trajectory:
        print("Warning: No trajectory data available for jump trajectory plot")
        return

    # タイムスタンプを取得（秒単位に変換）
    timestamps = []
    positions_x = []
    positions_y = []
    positions_z = []
    frames = []

    for point in trajectory:
        timestamp = point.get("timestamp")
        pos = point.get("position", (None, None, None))
        frame = point.get("frame")

        if pos[0] is not None and pos[1] is not None and pos[2] is not None:
            # タイムスタンプを秒に変換（ミリ秒単位の可能性がある）
            if timestamp is not None:
                if timestamp > 1000000000:  # ミリ秒単位と判定
                    timestamp_sec = timestamp / 1000.0
                else:
                    timestamp_sec = timestamp
                # 最初のタイムスタンプを0秒に基準化
                if not timestamps:
                    base_timestamp = timestamp_sec
                    timestamps.append(0.0)
                else:
                    timestamps.append(timestamp_sec - base_timestamp)
            else:
                timestamps.append(None)

            positions_x.append(pos[0])
            positions_y.append(pos[1])  # Y軸は高さ（RealSense座標系では下が正）
            positions_z.append(pos[2])
            frames.append(frame)

    if not positions_x:
        print("Warning: No valid trajectory positions found for jump trajectory plot")
        return

    # 日本語ラベル使用可否を確認
    USE_JAPANESE_LABELS = True
    try:
        import matplotlib.font_manager as fm

        japanese_fonts = [
            f.name
            for f in fm.fontManager.ttflist
            if "japan" in f.name.lower()
            or "noto" in f.name.lower()
            or "gothic" in f.name.lower()
        ]
        if not japanese_fonts:
            USE_JAPANESE_LABELS = False
    except:
        USE_JAPANESE_LABELS = False

    # カラーマップを準備（複数のグラフで使用）
    try:
        from matplotlib import colormaps

        colors = colormaps.get_cmap("tab10")
    except (AttributeError, ImportError):
        try:
            colors = plt.get_cmap("tab10")
        except:
            from matplotlib import cm

            colors = cm.get_cmap("tab10")

    # 1. 水平面（XZ平面）での軌跡を描画
    fig, ax = plt.subplots(figsize=(12, 10))

    # 全軌跡を描画（薄いグレー）
    ax.plot(
        positions_x,
        positions_z,
        "gray",
        alpha=0.3,
        linewidth=1,
        label="Full trajectory" if not USE_JAPANESE_LABELS else "全軌跡",
    )

    # ジャンプ中の軌跡を強調
    jumps = statistics.get("jumps", [])
    if jumps:
        for i, jump in enumerate(jumps):
            frame_start = jump.get("frame_start")
            frame_takeoff = jump.get("frame_takeoff", frame_start)
            frame_end = jump.get("frame_end")

            # ジャンプ範囲のインデックスを取得
            jump_indices = []
            for j, frame in enumerate(frames):
                if frame_start is not None and frame_end is not None:
                    if frame_start <= frame <= frame_end:
                        jump_indices.append(j)

            if jump_indices:
                jump_x = [positions_x[idx] for idx in jump_indices]
                jump_z = [positions_z[idx] for idx in jump_indices]
                color = colors(i % 10)

                # ジャンプ軌跡を描画
                ax.plot(
                    jump_x,
                    jump_z,
                    color=color,
                    linewidth=2.5,
                    alpha=0.8,
                    label=(
                        f"Jump {i+1}" if not USE_JAPANESE_LABELS else f"ジャンプ {i+1}"
                    ),
                )

                # 開始点、離陸点、着地点をマーク
                if jump_indices:
                    start_idx = jump_indices[0]
                    takeoff_idx = None
                    end_idx = jump_indices[-1]

                    # 離陸点を探す
                    for idx in jump_indices:
                        if frames[idx] == frame_takeoff:
                            takeoff_idx = idx
                            break

                    # 開始点を描画（緑の円）- 最初のジャンプのみ凡例に追加
                    ax.scatter(
                        [positions_x[start_idx]],
                        [positions_z[start_idx]],
                        c="green",
                        s=100,
                        marker="o",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=5,
                        label=(
                            "Start"
                            if not USE_JAPANESE_LABELS
                            else "開始" if i == 0 else ""
                        ),
                    )

                    # 離陸点を描画（オレンジの三角）- 最初のジャンプのみ凡例に追加
                    if takeoff_idx is not None:
                        ax.scatter(
                            [positions_x[takeoff_idx]],
                            [positions_z[takeoff_idx]],
                            c="orange",
                            s=100,
                            marker="^",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Takeoff"
                                if not USE_JAPANESE_LABELS
                                else "離陸" if i == 0 else ""
                            ),
                        )

                    # 着地点を描画（赤の四角）- 最初のジャンプのみ凡例に追加
                    ax.scatter(
                        [positions_x[end_idx]],
                        [positions_z[end_idx]],
                        c="red",
                        s=100,
                        marker="s",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=5,
                        label=(
                            "Landing"
                            if not USE_JAPANESE_LABELS
                            else "着地" if i == 0 else ""
                        ),
                    )

    # 軸ラベルとタイトル
    if USE_JAPANESE_LABELS:
        ax.set_xlabel("X座標 (m) - 左右方向", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z座標 (m) - 前後方向", fontsize=12, fontweight="bold")
        ax.set_title("ジャンプ軌跡（水平面）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("X coordinate (m) - Right", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z coordinate (m) - Forward", fontsize=12, fontweight="bold")
        ax.set_title(
            "Jump Trajectory (Horizontal Plane)", fontsize=14, fontweight="bold"
        )

    ax.grid(True, alpha=0.3, linestyle="--")
    # 凡例をグラフの外側（右側）に配置して重なりを避ける
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    trajectory_horizontal_path = output_dir / "jump_trajectory_horizontal.png"
    plt.savefig(str(trajectory_horizontal_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"Jump trajectory (horizontal plane) plot saved to: {trajectory_horizontal_path}"
    )

    # 2. 高さ（Y座標）と時間の関係を描画
    # RealSense座標系ではY軸が下向きが正なので、反転して上向きが正になるようにする
    fig, ax = plt.subplots(figsize=(14, 8))

    # Y座標を反転（RealSense座標系: 下向きが正 → 表示座標系: 上向きが正）
    # 基準値を求める（床面の高さ）
    if positions_y:
        y_min = min(positions_y)
        y_max = max(positions_y)
        # RealSense座標系ではYが大きいほど下にあるので、最大Y値が床面
        # ただし、max()だと時間範囲によって基準値が変わるため、
        # より安定した基準値として以下を使用:
        # 1. 最初の数フレーム（立位と仮定）のY値の平均
        # 2. または全データの上位10%のY値の中央値（立位時の基準）

        # 方法1: 最初の10フレームの平均（立位と仮定）
        initial_frames = min(10, len(positions_y))
        y_floor_initial = sum(positions_y[:initial_frames]) / initial_frames

        # 方法2: 上位10%のY値（最も低い位置）の中央値
        sorted_y = sorted(positions_y, reverse=True)  # 降順（大きい=低い位置が先）
        top_10_percent = sorted_y[:max(1, len(sorted_y) // 10)]
        y_floor_median = sum(top_10_percent) / len(top_10_percent)

        # より大きい値（より低い位置）を床基準として使用
        y_floor = max(y_floor_initial, y_floor_median)

        # Y座標を反転: 床からの距離として表示（上向きが正、0以上）
        # y_floor - y で計算: yが大きい（下）→0に近い、yが小さい（上）→大きな正の値
        heights = [y_floor - y for y in positions_y]
    else:
        heights = []
        y_floor = 0

    # 有効なタイムスタンプがあるかチェック
    valid_timestamps = [t for t in timestamps if t is not None]
    if valid_timestamps:
        # タイムスタンプを使用
        plot_x = timestamps
        x_label = "Time (seconds)" if not USE_JAPANESE_LABELS else "時間 (秒)"
    else:
        # フレーム番号を使用
        plot_x = frames
        x_label = "Frame number" if not USE_JAPANESE_LABELS else "フレーム番号"

    # 全軌跡を描画（薄いグレー）
    ax.plot(
        plot_x,
        heights,
        "gray",
        alpha=0.3,
        linewidth=1,
        label="Full trajectory" if not USE_JAPANESE_LABELS else "全軌跡",
    )

    # ジャンプ中の軌跡を強調
    if jumps:
        for i, jump in enumerate(jumps):
            frame_start = jump.get("frame_start")
            frame_takeoff = jump.get("frame_takeoff", frame_start)
            frame_end = jump.get("frame_end")
            jump_height = jump.get("height", 0) * 100  # cmに変換

            # ジャンプ範囲のインデックスを取得
            jump_indices = []
            for j, frame in enumerate(frames):
                if frame_start is not None and frame_end is not None:
                    if frame_start <= frame <= frame_end:
                        jump_indices.append(j)

            if jump_indices:
                jump_x = [plot_x[idx] for idx in jump_indices if idx < len(plot_x)]
                jump_heights = [
                    heights[idx] for idx in jump_indices
                ]  # 反転済みの高さを使用
                color = colors(i % 10)

                # ジャンプ軌跡を描画
                ax.plot(
                    jump_x,
                    jump_heights,
                    color=color,
                    linewidth=2.5,
                    alpha=0.8,
                    label=(
                        f"Jump {i+1} ({jump_height:.1f}cm)"
                        if not USE_JAPANESE_LABELS
                        else f"ジャンプ {i+1} ({jump_height:.1f}cm)"
                    ),
                )

                # 開始点、離陸点、着地点をマーク
                if jump_indices:
                    start_idx = jump_indices[0]
                    takeoff_idx = None
                    end_idx = jump_indices[-1]

                    # 離陸点を探す
                    for idx in jump_indices:
                        if frames[idx] == frame_takeoff:
                            takeoff_idx = idx
                            break

                    # 開始点を描画（反転済みの高さを使用）- 最初のジャンプのみ凡例に追加
                    if start_idx < len(plot_x) and start_idx < len(heights):
                        ax.scatter(
                            [plot_x[start_idx]],
                            [heights[start_idx]],
                            c="green",
                            s=100,
                            marker="o",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Start"
                                if not USE_JAPANESE_LABELS
                                else "開始" if i == 0 else ""
                            ),
                        )

                    # 離陸点を描画（反転済みの高さを使用）- 最初のジャンプのみ凡例に追加
                    if (
                        takeoff_idx is not None
                        and takeoff_idx < len(plot_x)
                        and takeoff_idx < len(heights)
                    ):
                        ax.scatter(
                            [plot_x[takeoff_idx]],
                            [heights[takeoff_idx]],
                            c="orange",
                            s=100,
                            marker="^",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Takeoff"
                                if not USE_JAPANESE_LABELS
                                else "離陸" if i == 0 else ""
                            ),
                        )

                    # 着地点を描画（反転済みの高さを使用）- 最初のジャンプのみ凡例に追加
                    if end_idx < len(plot_x) and end_idx < len(heights):
                        ax.scatter(
                            [plot_x[end_idx]],
                            [heights[end_idx]],
                            c="red",
                            s=100,
                            marker="s",
                            edgecolors="black",
                            linewidths=1.5,
                            zorder=5,
                            label=(
                                "Landing"
                                if not USE_JAPANESE_LABELS
                                else "着地" if i == 0 else ""
                            ),
                        )

    # 軸ラベルとタイトル（反転済みなので上向きが正）
    if USE_JAPANESE_LABELS:
        ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("高さ (m) - 床からの距離", fontsize=12, fontweight="bold")
        ax.set_title("ジャンプ軌跡（高さ-時間）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        ax.set_ylabel(
            "Height (m) - Distance from floor", fontsize=12, fontweight="bold"
        )
        ax.set_title("Jump Trajectory (Height-Time)", fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    # 凡例をグラフの外側（右側）に配置して重なりを避ける
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    plt.tight_layout()

    trajectory_height_path = output_dir / "jump_trajectory_height.png"
    plt.savefig(str(trajectory_height_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Jump trajectory (height-time) plot saved to: {trajectory_height_path}")

    # 3. X座標-時間グラフ（Mid-HipのX座標）
    fig, ax = plt.subplots(figsize=(14, 8))

    # 有効なタイムスタンプとX座標のデータのみプロット
    valid_time_x = [(t, x) for t, x in zip(timestamps, positions_x) if t is not None]
    if valid_time_x:
        valid_times, valid_x = zip(*valid_time_x)
        ax.plot(
            valid_times,
            valid_x,
            "b-",
            alpha=0.5,
            linewidth=1,
            label="Full trajectory" if not USE_JAPANESE_LABELS else "全軌跡",
        )

        # ジャンプ中の軌跡を強調
        if jumps:
            for i, jump in enumerate(jumps):
                frame_start = jump.get("frame_start")
                frame_takeoff = jump.get("frame_takeoff", frame_start)
                frame_end = jump.get("frame_end")
                jump_type = jump.get("jump_type", "unknown")

                if frame_start is not None and frame_end is not None:
                    jump_times = []
                    jump_x = []

                    for j, f in enumerate(frames):
                        if f is not None and frame_start <= f <= frame_end:
                            if timestamps[j] is not None:
                                jump_times.append(timestamps[j])
                                jump_x.append(positions_x[j])

                    if jump_times:
                        color = colors(i % 10)
                        jump_label = f"Jump {i+1} ({jump_type})"
                        ax.plot(
                            jump_times,
                            jump_x,
                            color=color,
                            linewidth=2.5,
                            label=jump_label,
                            alpha=0.9,
                        )

                        # 離陸点と着地点をマーク
                        if len(jump_times) > 0:
                            ax.scatter(
                                [jump_times[0]],
                                [jump_x[0]],
                                color=color,
                                s=100,
                                marker="^",
                                zorder=5,
                                edgecolors="black",
                                linewidths=1,
                            )
                            ax.scatter(
                                [jump_times[-1]],
                                [jump_x[-1]],
                                color=color,
                                s=100,
                                marker="v",
                                zorder=5,
                                edgecolors="black",
                                linewidths=1,
                            )

    if USE_JAPANESE_LABELS:
        ax.set_xlabel("時間 (秒)", fontsize=12, fontweight="bold")
        ax.set_ylabel("X座標 (m) - 左右方向", fontsize=12, fontweight="bold")
        ax.set_title("腰の中点 X座標（時系列）", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel("X coordinate (m) - Left/Right", fontsize=12, fontweight="bold")
        ax.set_title("Mid-Hip X Coordinate (Time Series)", fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    plt.tight_layout()

    trajectory_x_path = output_dir / "jump_trajectory_x.png"
    plt.savefig(str(trajectory_x_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Jump trajectory (X-time) plot saved to: {trajectory_x_path}")


def load_config(config_path):
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス（TOML形式）

    Returns:
        dict: 設定辞書、読み込み失敗時はNone
    """
    if not TOML_AVAILABLE:
        print("Warning: toml not installed. Install with: pip install toml")
        return None

    config_file = Path(config_path)
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return config
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}")
        return None


def merge_config_with_args(config, args):
    """
    設定ファイルの値をコマンドライン引数で上書き

    Args:
        config: 設定ファイルの辞書
        args: argparse.Namespaceオブジェクト

    Returns:
        argparse.Namespace: マージされた引数オブジェクト
    """
    if config is None:
        return args

    # 設定ファイルの値をデフォルトとして使用（コマンドライン引数が指定されていない場合）
    # 新旧両方の形式に対応（enable_* 形式を優先、後方互換性のため no_* 形式もサポート）

    # filenameが指定されている場合、inputとoutputを自動生成
    if "filename" in config and config["filename"]:
        filename = config["filename"]
        # inputが明示的に指定されていない場合のみ自動生成
        if "input" not in config or not config["input"]:
            config["input"] = f"record/{filename}.bag"
        # outputが明示的に指定されていない場合のみ自動生成
        if "output" not in config or not config["output"]:
            config["output"] = f"results/{filename}"

    # 新しいenable_*形式をno_*形式に変換（コマンドライン引数との互換性のため）
    if "enable_video" in config:
        config["no_video"] = not config["enable_video"]
    if "enable_3d_animation" in config:
        config["no_3d_animation"] = not config["enable_3d_animation"]
    if "enable_depth_interpolation" in config:
        config["no_depth_interpolation"] = not config["enable_depth_interpolation"]
    if "enable_floor_detection" in config:
        config["no_floor_detection"] = not config["enable_floor_detection"]

    config_to_args_map = {
        "input": "input",
        "output": "output",
        "model_dir": "model_dir",
        "model_name": "model_name",
        "threshold_vertical": "threshold_vertical",
        "threshold_horizontal": "threshold_horizontal",
        "min_jump_height": "min_jump_height",
        "min_air_time": "min_air_time",
        "no_video": "no_video",
        "no_3d_animation": "no_3d_animation",
        "interactive_3d": "interactive_3d",
        "smooth_keypoints": "smooth_keypoints",
        "smooth_window_size": "smooth_window_size",
        "no_depth_interpolation": "no_depth_interpolation",
        "depth_kernel_size": "depth_kernel_size",
        "use_kalman_filter": "use_kalman_filter",
        "kalman_process_noise": "kalman_process_noise",
        "kalman_measurement_noise": "kalman_measurement_noise",
        "no_floor_detection": "no_floor_detection",
        "start_time": "start_time",
        "end_time": "end_time",
        "frame_skip": "frame_skip",
        "resize_factor": "resize_factor",
        "minimal_data": "minimal_data",
        "dual": "dual",
    }

    for config_key, arg_name in config_to_args_map.items():
        if config_key in config and config[config_key] is not None:
            # コマンドライン引数が指定されていない（デフォルト値）場合のみ設定ファイルの値を使用
            # argparseでは、明示的に指定された引数とデフォルト値を区別するのが難しいため、
            # 設定ファイルの値が存在し、かつコマンドライン引数の値がNoneまたはデフォルト値の場合は上書き
            current_value = getattr(args, arg_name, None)

            # smooth_keypointsは特別扱い（設定ファイルではboolean、コマンドラインではwindow_size）
            if config_key == "smooth_keypoints":
                if isinstance(config[config_key], bool):
                    # 設定ファイルでbooleanの場合は、window_sizeを設定
                    if config[config_key]:
                        # smooth_window_sizeが設定されていればそれを使用、なければデフォルト
                        window_size = config.get("smooth_window_size", 5)
                        setattr(args, "smooth_keypoints", window_size)
                    else:
                        setattr(args, "smooth_keypoints", 0)
                else:
                    # 既に整数値の場合はそのまま使用
                    setattr(args, arg_name, config[config_key])
            # start_timeとend_time: 0はNoneとして扱う（最初から/最後まで）
            elif config_key in ["start_time", "end_time"]:
                if config[config_key] == 0:
                    setattr(args, arg_name, None)
                else:
                    setattr(args, arg_name, config[config_key])
            # ブール値の場合は、設定ファイルの値を優先（明示的にFalseでも有効）
            elif config_key in [
                "no_video",
                "no_3d_animation",
                "interactive_3d",
                "no_depth_interpolation",
                "use_kalman_filter",
                "no_floor_detection",
                "minimal_data",
                "dual",
            ]:
                setattr(args, arg_name, config[config_key])
            # Noneがデフォルトの値
            elif current_value is None and config[config_key] is not None:
                setattr(args, arg_name, config[config_key])
            # その他の値は、デフォルト値の可能性がある場合に上書き
            # （簡単のため、設定ファイルの値で上書き。明示的に指定したい場合はコマンドライン引数を使用）
            elif current_value is None or (
                isinstance(current_value, (int, float))
                and current_value == getattr(argparse.Namespace(), arg_name, None)
            ):
                setattr(args, arg_name, config[config_key])

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Jump Analyzer: Analyze jump height, distance, and trajectory from RealSense .bag file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/

  # Use config file
  python jump_analyzer.py --config config.toml
  python jump_analyzer.py --config config.toml --input other_file.bag  # Override input from config

  # Analyze specific time range (seconds)
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --start-time 5.0 --end-time 10.0

  # Specify model directory
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --model-dir models/

  # Adjust jump detection thresholds
  python jump_analyzer.py --input bagdata/my_recording.bag --output results/ --threshold-vertical 0.1 --threshold-horizontal 0.2
        """,
    )

    # 設定ファイル
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (TOML format). If specified, config values will be used as defaults (command line args override config).",
    )

    # 必須引数（設定ファイルで指定可能なためrequired=False、後でチェック）
    parser.add_argument(
        "--input", type=str, required=False, default=None, help="Input .bag file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Output directory for results",
    )

    # オプション引数
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for YOLOv8-Pose model files (default: models)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov8x-pose.pt",
        help="YOLOv8-Pose model name (yolov8n-pose.pt, yolov8s-pose.pt, etc.) (default: yolov8x-pose.pt)",
    )
    parser.add_argument(
        "--threshold-vertical",
        type=float,
        default=0.05,
        help="Vertical jump detection threshold in meters (default: 0.05)",
    )
    parser.add_argument(
        "--threshold-horizontal",
        type=float,
        default=0.1,
        help="Horizontal jump detection threshold in meters (default: 0.1)",
    )
    parser.add_argument(
        "--min-jump-height",
        type=float,
        default=0.10,
        help="Minimum jump height in meters to be considered a valid jump (default: 0.10m = 10cm)",
    )
    parser.add_argument(
        "--min-air-time",
        type=float,
        default=0.20,
        help="Minimum air time in seconds to be considered a valid jump (default: 0.20s = 200ms)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip video visualization output"
    )
    parser.add_argument(
        "--no-3d-animation",
        action="store_true",
        help="Skip 3D keypoint animation generation",
    )
    parser.add_argument(
        "--interactive-3d",
        action="store_true",
        help="Display interactive 3D keypoint animation (can rotate view with mouse)",
    )
    parser.add_argument(
        "--smooth-keypoints",
        type=int,
        default=5,
        metavar="N",
        help="Enable keypoint smoothing with window size N (default: 5, set to 0 to disable)",
    )
    parser.add_argument(
        "--no-depth-interpolation",
        action="store_true",
        help="Disable depth interpolation (use single pixel depth, faster but less accurate)",
    )
    parser.add_argument(
        "--depth-kernel-size",
        type=int,
        default=3,
        metavar="N",
        help="Depth interpolation kernel size (default: 3, must be odd, larger = more smoothing)",
    )
    parser.add_argument(
        "--use-kalman-filter",
        action="store_true",
        help="Use Kalman filter for temporal smoothing (research-grade, more accurate than moving average)",
    )
    parser.add_argument(
        "--kalman-process-noise",
        type=float,
        default=0.01,
        help="Kalman filter process noise (default: 0.01, smaller = smoother but slower response)",
    )
    parser.add_argument(
        "--kalman-measurement-noise",
        type=float,
        default=0.1,
        help="Kalman filter measurement noise (default: 0.1, larger = more trust in predictions)",
    )
    parser.add_argument(
        "--no-floor-detection",
        action="store_true",
        help="Disable floor detection (use traditional height-based detection)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time for playback in seconds (default: start from beginning)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time for playback in seconds (default: play until end)",
    )

    # 高速化オプション
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every N frames (1=all frames, 2=every other frame, etc.) (default: 1)",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=1.0,
        help="Resize image before YOLOv8-Pose inference (1.0=no resize, 0.5=half size) (default: 1.0)",
    )
    parser.add_argument(
        "--minimal-data",
        action="store_true",
        help="Save only jump detection frames in JSON (faster, smaller file)",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Enable dual camera mode: if input is a directory, automatically find *_metadata.json file",
    )

    args = parser.parse_args()

    # 設定ファイルを読み込む
    config = None
    if args.config:
        config = load_config(args.config)
        if config:
            print(f"Loaded config from: {args.config}")
            # 設定ファイルの値をマージ（コマンドライン引数が優先）
            args = merge_config_with_args(config, args)
    elif Path("config.toml").exists():
        # デフォルトで config.toml を探す
        config = load_config("config.toml")
        if config:
            print("Loaded config from: config.toml")
            args = merge_config_with_args(config, args)

    # 必須引数のチェック（設定ファイルでも指定可能）
    if args.input is None:
        print(
            "Error: --input is required (can be specified in config file or command line)",
            file=sys.stderr,
        )
        sys.exit(1)
    # 出力ディレクトリ: 常に「ベースディレクトリ/入力ファイル名(拡張子なし)」に集約
    input_path_for_output = Path(args.input)
    if args.output is None:
        print("Error: --output is required (can be specified in config file or command line)", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # メタデータJSONファイルの確認（2台のカメラ録画の場合）
    bag_file_path = args.input
    metadata = None
    
    # dualモード: ディレクトリ指定の場合、メタデータJSONを自動検出
    if args.dual and os.path.isdir(args.input):
        import glob
        metadata_files = glob.glob(os.path.join(args.input, "*_metadata.json"))
        if metadata_files:
            if len(metadata_files) > 1:
                print(f"[WARN] Multiple metadata files found, using: {metadata_files[0]}")
            args.input = metadata_files[0]
            print(f"[INFO] Dual mode: Found metadata file: {args.input}")
        else:
            print(f"[ERROR] Dual mode enabled but no *_metadata.json found in: {args.input}", file=sys.stderr)
            sys.exit(1)
    
    if args.input.endswith("_metadata.json"):
        # メタデータJSONが指定された場合
        if not os.path.exists(args.input):
            print(f"Error: Metadata file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        try:
            with open(args.input, "r") as f:
                metadata = json.load(f)

            # cam0のbagファイルパスを取得
            metadata_dir = os.path.dirname(args.input)
            if not metadata_dir:
                metadata_dir = "."
            bag_file_path = os.path.join(metadata_dir, metadata["cam0_file"])

            if not os.path.exists(bag_file_path):
                print(
                    f"Error: Bag file not found: {bag_file_path}",
                    file=sys.stderr,
                )
                print(
                    f"  Metadata file: {args.input}",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"[INFO] Dual camera recording detected")
            print(f"[INFO] Metadata file: {args.input}")
            print(f"[INFO] Using camera 0 bag file: {bag_file_path}")
            print(
                f"[INFO] Calibration fitness: {metadata.get('calibration', {}).get('fitness', 'N/A'):.3f}, RMSE: {metadata.get('calibration', {}).get('inlier_rmse', 'N/A'):.4f}m"
            )

        except Exception as e:
            print(
                f"Error: Failed to load metadata file: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # 通常のbagファイルの場合
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("Jump Analyzer - YOLOv8-Pose 3D Analysis")
    print("=" * 60)
    print(f"Input file: {bag_file_path}")
    print(f"Output directory: {args.output}")
    print()

    # モデルディレクトリを作成（存在しない場合）
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # YOLOv8-Poseモデルの読み込み
    print("Loading YOLOv8-Pose model...")
    yolo_pose = YOLOv8PoseDetector(model_name=args.model_name, model_dir=str(model_dir))
    if not yolo_pose.load_model():
        print("Error: Failed to load YOLOv8-Pose model", file=sys.stderr)
        sys.exit(1)

    # バグファイルの読み込み
    bag_reader = None
    bag_reader_cam1 = None
    T1_to_0 = None
    pairer = None
    
    if metadata:
        # デュアルカメラモード
        print("Loading dual camera .bag files...")
        bag_file_path_cam1 = os.path.join(metadata_dir, metadata["cam1_file"])
        
        if not os.path.exists(bag_file_path_cam1):
            print(f"Error: Camera 1 bag file not found: {bag_file_path_cam1}", file=sys.stderr)
            sys.exit(1)
        
        # cam0のbagファイル読み込み
        bag_reader = BagFileReader(
            bag_file_path, start_time=args.start_time, end_time=args.end_time
        )
        if not bag_reader.initialize():
            print("Error: Failed to load camera 0 .bag file", file=sys.stderr)
            sys.exit(1)
        
        # cam1のbagファイル読み込み
        bag_reader_cam1 = BagFileReader(
            bag_file_path_cam1, start_time=args.start_time, end_time=args.end_time
        )
        if not bag_reader_cam1.initialize():
            print("Error: Failed to load camera 1 .bag file", file=sys.stderr)
            sys.exit(1)
        
        # キャリブレーション情報を取得
        calib_data = metadata.get("calibration", {})
        if "T1_to_0" in calib_data:
            T1_to_0 = np.array(calib_data["T1_to_0"], dtype=np.float64)
            print(f"[INFO] Loaded calibration transformation matrix")
        else:
            print("[WARN] Calibration matrix not found in metadata, using camera 0 only", file=sys.stderr)
            bag_reader_cam1 = None
        
        # フレームペアラーを初期化
        align0_obj = None
        align1_obj = None
        use_polling_mode = False  # 初期はwait_for_framesを使用
        if bag_reader_cam1:
            pairing_tol = metadata.get("pairing_tolerance_ms", 25.0)
            pairer = FramePairer(tol_ms=pairing_tol)
            # alignオブジェクトを作成（フレーム処理用）
            align0_obj = rs.align(rs.stream.color)
            align1_obj = rs.align(rs.stream.color)
            print(f"[INFO] Dual camera mode enabled with pairing tolerance: {pairing_tol}ms")
            # 最初の数フレームはwait_for_framesで取得してからpollingに切り替え
            print("[INFO] Initializing dual camera frames...")
            init_success = False
            for _ in range(10):
                try:
                    frames0 = bag_reader.pipeline.wait_for_frames(timeout_ms=1000)
                    frames1 = bag_reader_cam1.pipeline.wait_for_frames(timeout_ms=1000)
                    if frames0 and frames1:
                        aligned0 = align0_obj.process(frames0)
                        aligned1 = align1_obj.process(frames1)
                        cf0 = aligned0.get_color_frame()
                        df0 = aligned0.get_depth_frame()
                        cf1 = aligned1.get_color_frame()
                        df1 = aligned1.get_depth_frame()
                        if cf0 and df0 and cf1 and df1:
                            pairer.put_and_match((cf0, df0), (cf1, df1))
                            init_success = True
                            break
                except Exception:
                    pass
            if init_success:
                print("[INFO] Dual camera frames initialized, switching to polling mode")
                use_polling_mode = True
            else:
                print("[WARN] Failed to initialize dual camera frames, using blocking mode")
    else:
        # 単一カメラモード
        print("Loading .bag file...")
        bag_reader = BagFileReader(
            bag_file_path, start_time=args.start_time, end_time=args.end_time
        )
        if not bag_reader.initialize():
            print("Error: Failed to load .bag file", file=sys.stderr)
            sys.exit(1)

    # メタデータがあれば保存（後で参照可能に）
    if metadata:
        metadata_save_path = output_dir / "recording_metadata.json"
        with open(metadata_save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] Metadata saved to: {metadata_save_path}")

    # 床検出器の初期化（オプション）
    floor_detector = None
    if not args.no_floor_detection:
        floor_detector = FloorDetector(
            floor_threshold=0.03,  # 3cm（より緩い閾値で検出しやすく）
            min_inliers=500,  # より少ない点数で検出可能に
            max_iterations=500,
        )
        print(
            "Floor detection enabled: Using foot-floor contact for precise jump detection"
        )
    else:
        print("Floor detection disabled: Using traditional height-based detection")

    # ジャンプ検出器の初期化
    jump_detector = JumpDetector(
        threshold_vertical=args.threshold_vertical,
        threshold_horizontal=args.threshold_horizontal,
        min_jump_height=args.min_jump_height,
        min_air_time=args.min_air_time,
        floor_detector=floor_detector,
        use_floor_detection=not args.no_floor_detection,
    )

    # 可視化器の初期化
    visualizer = JumpVisualizer()

    # キーポイント平滑化器の初期化（Kalmanフィルタまたは移動平均）
    if args.use_kalman_filter:
        smoother = KalmanSmoother(
            process_noise=args.kalman_process_noise,
            measurement_noise=args.kalman_measurement_noise,
        )
        print(
            f"Kalman filter smoothing enabled: process_noise={args.kalman_process_noise}, measurement_noise={args.kalman_measurement_noise}"
        )
    elif args.smooth_keypoints > 0:
        smoother = KeypointSmoother(
            window_size=args.smooth_keypoints, smoothing_type="moving_average"
        )
        print(f"Moving average smoothing enabled: window_size={args.smooth_keypoints}")
    else:
        smoother = None
        print("Keypoint smoothing disabled")

    # 深度補間の設定を表示
    if not args.no_depth_interpolation:
        print(f"Depth interpolation enabled: kernel_size={args.depth_kernel_size}")
    else:
        print("Depth interpolation disabled (using single pixel depth)")

    # データ保存用のリスト
    all_frames_data = []

    # 可視化動画ライター（メモリ効率化のため逐次書き込み）
    video_writer = None
    video_path = None
    video_fps = config.get("video_fps", 30)  # configから取得、デフォルト30fps

    print("\nProcessing frames...")
    if args.frame_skip > 1:
        print(f"  Frame skipping: Processing every {args.frame_skip} frame(s)")
    if args.resize_factor < 1.0:
        print(f"  Image resize: {args.resize_factor*100:.0f}% of original size")
    if args.minimal_data:
        print("  Minimal data mode: Saving only jump detection frames")
    print()
    print("Note: First frame processing may take longer (model warm-up)...")
    print()

    frame_num = 0
    processed_frame_count = 0
    skipped_frame_count = 0
    pairing_fail_count = 0  # ペアリング失敗カウント（デバッグ用）
    start_time = time.time()
    last_progress_time = start_time
    floor_detection_done = False  # 床検出が完了したかどうか

    try:
        while True:
            # フレームを取得（デュアルモード対応）
            if bag_reader_cam1 and pairer:
                # デュアルカメラモード: 非ブロッキングで両カメラからフレームを取得して同期
                if use_polling_mode:
                    frames0_raw = bag_reader.poll_for_frames()
                    frames1_raw = bag_reader_cam1.poll_for_frames()
                else:
                    # ブロッキングモード（初期化時やpollingが失敗した場合）
                    try:
                        frames0_raw = bag_reader.pipeline.wait_for_frames(timeout_ms=100)
                        frames1_raw = bag_reader_cam1.pipeline.wait_for_frames(timeout_ms=100)
                    except Exception:
                        frames0_raw = None
                        frames1_raw = None
                
                # デバッグ: 最初の数回のみフレーム取得状況を出力
                if frame_num == 0 and pairing_fail_count < 5:
                    print(f"[DEBUG] Frame polling: cam0={frames0_raw is not None}, cam1={frames1_raw is not None}")
                
                if frames0_raw and align0_obj:
                    try:
                        aligned0 = align0_obj.process(frames0_raw)
                        color_frame0 = aligned0.get_color_frame()
                        depth_frame0 = aligned0.get_depth_frame()
                        if color_frame0 and depth_frame0:
                            pairer.put_and_match((color_frame0, depth_frame0), None)
                    except Exception as e:
                        if pairing_fail_count < 5:
                            print(f"[DEBUG] Error processing cam0 frame: {e}")
                
                if frames1_raw and align1_obj:
                    try:
                        aligned1 = align1_obj.process(frames1_raw)
                        color_frame1 = aligned1.get_color_frame()
                        depth_frame1 = aligned1.get_depth_frame()
                        if color_frame1 and depth_frame1:
                            pairer.put_and_match(None, (color_frame1, depth_frame1))
                    except Exception as e:
                        if pairing_fail_count < 5:
                            print(f"[DEBUG] Error processing cam1 frame: {e}")
                
                # ペアリングを試行
                match = pairer.put_and_match(None, None)
                
                if match is None:
                    # ペアが揃わない場合はスキップ（CPU負荷軽減）
                    skipped_frame_count += 1
                    pairing_fail_count += 1
                    
                    # デバッグ情報を出力（最初の数回と定期的に）
                    if pairing_fail_count <= 10 or pairing_fail_count % 500 == 0:
                        buf0_len = len(pairer.buf0) if pairer else 0
                        buf1_len = len(pairer.buf1) if pairer else 0
                        print(f"[DEBUG] Waiting for frame pair... (failed={pairing_fail_count}, buf0={buf0_len}, buf1={buf1_len})")
                    
                    # 失敗が多すぎる場合、cam0のみで続行
                    if pairing_fail_count >= 500:
                        print(f"[WARN] Frame pairing failed {pairing_fail_count} times. Checking if frames are available...")
                        # 両方のカメラからフレームが取得できているか確認
                        test_frames0 = bag_reader.poll_for_frames()
                        test_frames1 = bag_reader_cam1.poll_for_frames()
                        if test_frames0 and not test_frames1:
                            print("[INFO] Camera 1 frames not available, continuing with camera 0 only")
                            bag_reader_cam1 = None
                            pairer = None
                            align0_obj = None
                            align1_obj = None
                            # cam0のフレームを直接取得して続行
                            color_frame, depth_frame = bag_reader.get_frames()
                            if color_frame is None or depth_frame is None:
                                break
                            frame_num += 1
                            use_dual_mode = False
                            color_frame1 = None
                            depth_frame1 = None
                        elif not test_frames0 and test_frames1:
                            print("[INFO] Camera 0 frames not available, exiting")
                            break
                        elif not test_frames0 and not test_frames1:
                            print("[INFO] No frames available from either camera, exiting")
                            break
                        else:
                            # フレームは取得できているがペアリングできない
                            # タイムスタンプの差を確認
                            if pairer.buf0 and pairer.buf1:
                                ts0 = pairer.ts(pairer.buf0[0])
                                ts1 = pairer.ts(pairer.buf1[0])
                                if ts0 and ts1:
                                    dt = abs(ts0 - ts1)
                                    print(f"[WARN] Frame pairing issue: dt={dt:.1f}ms (tol={pairer.tol:.1f}ms)")
                                    # 許容時間を一時的に拡大して試行
                                    if dt < pairer.tol * 10:  # 許容時間の10倍以内なら拡大
                                        pairer.tol = dt * 1.2  # 20%のマージンを追加
                                        print(f"[INFO] Temporarily increased pairing tolerance to {pairer.tol:.1f}ms")
                            pairing_fail_count = 0  # リセットして再試行
                    
                    time.sleep(0.01)  # CPU負荷軽減
                    continue
                
                # ペアリング成功
                pairing_fail_count = 0
                (color_frame0, depth_frame0), (color_frame1, depth_frame1), dt_ms = match
                frame_num += 1
                use_dual_mode = True
                
                # 共通変数に設定
                color_frame = color_frame0
                depth_frame = depth_frame0
                
                # 終了チェック（どちらかのカメラが終了した場合）
                if color_frame0 is None or depth_frame0 is None:
                    break
                if color_frame1 is None or depth_frame1 is None:
                    bag_reader_cam1 = None
                    pairer = None
                    align0_obj = None
                    align1_obj = None
                    print("[INFO] Camera 1 stream ended, continuing with camera 0 only")
                    use_dual_mode = False
                    color_frame1 = None
                    depth_frame1 = None
            else:
                # 単一カメラモード
                color_frame, depth_frame = bag_reader.get_frames()
                if color_frame is None or depth_frame is None:
                    break
                frame_num += 1  # フレーム取得時にカウント
                use_dual_mode = False
                color_frame1 = None
                depth_frame1 = None

            color_image, depth_image = bag_reader.frame_to_numpy(
                color_frame, depth_frame
            )

            if color_image is None or depth_image is None:
                continue

            # 床検出（最初の数フレームで実行、フレームスキッピング前）
            if floor_detector and not floor_detection_done:
                # フレームスキッピングを考慮せず、実際の読み込みフレーム数で判定
                if frame_num <= 100:  # 最初の100フレームで床検出を試行
                    # 床検出を試行（毎フレーム試行、成功したら終了）
                    if floor_detector.detect_floor_from_depth(
                        depth_frame,
                        bag_reader.depth_scale,
                        bag_reader.depth_intrinsics or bag_reader.intrinsics,
                    ):
                        floor_plane, floor_normal, floor_height = (
                            floor_detector.get_floor_plane()
                        )
                        if floor_plane is not None:
                            floor_detection_done = True
                            print(f"\n✓ Floor detected at frame {frame_num}!")
                            print(f"  Floor height (Y): {floor_height:.3f}m")
                            print(
                                f"  Floor normal: ({floor_normal[0]:.3f}, {floor_normal[1]:.3f}, {floor_normal[2]:.3f})"
                            )
                elif frame_num > 100:
                    # 100フレーム試しても検出できない場合は警告し、従来方式に切り替え
                    print("\n⚠ Warning: Floor detection failed after 100 frames.")
                    print("  Continuing with traditional height-based detection.")
                    floor_detection_done = True
                    # 床検出を無効化
                    if jump_detector:
                        jump_detector.use_floor_detection = False
                    floor_detector = None

            # フレームスキッピング
            if frame_num % args.frame_skip != 0:
                skipped_frame_count += 1
                continue

            processed_frame_count += 1

            # 画像リサイズ（YOLOv8-Pose推論前）
            original_shape = color_image.shape[:2]
            inference_image = color_image
            resize_scale_x = 1.0
            resize_scale_y = 1.0

            if args.resize_factor < 1.0:
                new_width = int(color_image.shape[1] * args.resize_factor)
                new_height = int(color_image.shape[0] * args.resize_factor)
                inference_image = resize_image(color_image, new_width, new_height)
                resize_scale_x = color_image.shape[1] / new_width
                resize_scale_y = color_image.shape[0] / new_height

            # 2D keypointsを検出（リサイズ済み画像を使用）
            if processed_frame_count == 1:
                print(f"Processing first frame (frame {frame_num})...")

            # ポーズ推定の時間を測定
            pose_start = time.time()
            
            # cam0のキーポイント検出
            keypoints_2d_cam0 = yolo_pose.detect_keypoints(inference_image)
            pose_time_cam0 = time.time() - pose_start
            
            # デュアルモード: cam1のキーポイント検出
            keypoints_2d_cam1 = None
            pose_time_cam1 = 0.0
            color_image1 = None
            depth_image1 = None
            if use_dual_mode and color_frame1 and depth_frame1:
                color_image1, depth_image1 = bag_reader_cam1.frame_to_numpy(
                    color_frame1, depth_frame1
                )
                if color_image1 is not None:
                    inference_image1 = color_image1
                    resize_scale_x1 = 1.0
                    resize_scale_y1 = 1.0
                    if args.resize_factor < 1.0:
                        new_width1 = int(color_image1.shape[1] * args.resize_factor)
                        new_height1 = int(color_image1.shape[0] * args.resize_factor)
                        inference_image1 = resize_image(color_image1, new_width1, new_height1)
                        resize_scale_x1 = color_image1.shape[1] / new_width1
                        resize_scale_y1 = color_image1.shape[0] / new_height1
                    
                    pose_start_cam1 = time.time()
                    keypoints_2d_cam1 = yolo_pose.detect_keypoints(inference_image1)
                    pose_time_cam1 = time.time() - pose_start_cam1
                    
                    # keypoints座標を元の画像サイズにスケール
                    if args.resize_factor < 1.0 and keypoints_2d_cam1:
                        keypoints_2d_cam1 = [
                            (
                                (kp[0] * resize_scale_x1, kp[1] * resize_scale_y1, kp[2])
                                if kp[0] is not None and kp[1] is not None
                                else kp
                            )
                            for kp in keypoints_2d_cam1
                        ]
            
            # keypoints座標を元の画像サイズにスケール（cam0）
            if args.resize_factor < 1.0 and keypoints_2d_cam0:
                keypoints_2d_cam0 = [
                    (
                        (kp[0] * resize_scale_x, kp[1] * resize_scale_y, kp[2])
                        if kp[0] is not None and kp[1] is not None
                        else kp
                    )
                    for kp in keypoints_2d_cam0
                ]
            
            pose_time = pose_time_cam0 + pose_time_cam1

            if keypoints_2d_cam0 is None:
                continue

            # 3D keypointsを計算（バッチ処理で高速化）
            depth_start = time.time()

            # cam0の3Dキーポイント計算
            image_height, image_width = color_image.shape[:2]
            border_threshold = 8

            valid_data_cam0 = [
                (kp_name, kp_2d[0], kp_2d[1], kp_2d[2])
                for kp_name, kp_2d in zip(COCO_KEYPOINTS, keypoints_2d_cam0)
                if (
                    kp_2d[0] is not None
                    and kp_2d[1] is not None
                    and kp_2d[2] > 0.1
                    and border_threshold <= kp_2d[0] < image_width - border_threshold
                    and border_threshold <= kp_2d[1] < image_height - border_threshold
                )
            ]

            if valid_data_cam0:
                valid_points_cam0 = [(x, y) for _, x, y, _ in valid_data_cam0]
                kp_confidences_cam0 = [conf for _, _, _, conf in valid_data_cam0]
                depths_cam0 = bag_reader.get_depth_at_points_batch(
                    depth_frame,
                    valid_points_cam0,
                    use_interpolation=not args.no_depth_interpolation,
                    kernel_size=(
                        args.depth_kernel_size if args.depth_kernel_size % 2 == 1 else 3
                    ),
                    confidences=kp_confidences_cam0,
                )
                coords_3d_cam0 = bag_reader.pixels_to_3d_batch(valid_points_cam0, depths_cam0)
                valid_coords_dict_cam0 = {
                    kp_name: (
                        coords_3d_cam0[i] if coords_3d_cam0[i] is not None else (None, None, None)
                    )
                    for i, (kp_name, _, _, _) in enumerate(valid_data_cam0)
                }
                keypoints_3d_cam0 = {
                    kp_name: valid_coords_dict_cam0.get(kp_name, (None, None, None))
                    for kp_name in COCO_KEYPOINTS
                }
            else:
                keypoints_3d_cam0 = {
                    kp_name: (None, None, None) for kp_name in COCO_KEYPOINTS
                }
            
            # デュアルモード: cam1の3Dキーポイント計算と変換
            keypoints_3d_cam1 = {kp_name: (None, None, None) for kp_name in COCO_KEYPOINTS}
            if use_dual_mode and keypoints_2d_cam1 and color_image1 is not None:
                image_height1, image_width1 = color_image1.shape[:2]
                valid_data_cam1 = [
                    (kp_name, kp_2d[0], kp_2d[1], kp_2d[2])
                    for kp_name, kp_2d in zip(COCO_KEYPOINTS, keypoints_2d_cam1)
                    if (
                        kp_2d[0] is not None
                        and kp_2d[1] is not None
                        and kp_2d[2] > 0.1
                        and border_threshold <= kp_2d[0] < image_width1 - border_threshold
                        and border_threshold <= kp_2d[1] < image_height1 - border_threshold
                    )
                ]
                
                if valid_data_cam1:
                    valid_points_cam1 = [(x, y) for _, x, y, _ in valid_data_cam1]
                    kp_confidences_cam1 = [conf for _, _, _, conf in valid_data_cam1]
                    depths_cam1 = bag_reader_cam1.get_depth_at_points_batch(
                        depth_frame1,
                        valid_points_cam1,
                        use_interpolation=not args.no_depth_interpolation,
                        kernel_size=(
                            args.depth_kernel_size if args.depth_kernel_size % 2 == 1 else 3
                        ),
                        confidences=kp_confidences_cam1,
                    )
                    coords_3d_cam1_raw = bag_reader_cam1.pixels_to_3d_batch(valid_points_cam1, depths_cam1)
                    
                    # cam1座標系からcam0座標系に変換
                    valid_coords_dict_cam1 = {}
                    for i, (kp_name, _, _, _) in enumerate(valid_data_cam1):
                        if coords_3d_cam1_raw[i] is not None:
                            coords_transformed = transform_point_to_cam0_coords(
                                coords_3d_cam1_raw[i], T1_to_0
                            )
                            valid_coords_dict_cam1[kp_name] = coords_transformed
                        else:
                            valid_coords_dict_cam1[kp_name] = (None, None, None)
                    
                    keypoints_3d_cam1 = {
                        kp_name: valid_coords_dict_cam1.get(kp_name, (None, None, None))
                        for kp_name in COCO_KEYPOINTS
                    }
            
            # キーポイントを統合
            if use_dual_mode and T1_to_0 is not None:
                keypoints_3d = merge_keypoints(
                    keypoints_3d_cam0,
                    keypoints_3d_cam1,
                    keypoints_2d_cam0,
                    keypoints_2d_cam1
                )
                keypoints_2d = keypoints_2d_cam0  # 可視化用はcam0を使用
            else:
                keypoints_3d = keypoints_3d_cam0
                keypoints_2d = keypoints_2d_cam0

            # キーポイント平滑化を適用
            if smoother is not None:
                keypoints_3d = smoother.smooth(keypoints_3d)

            depth_time = time.time() - depth_start

            # 最初の数フレームで時間を表示
            if processed_frame_count <= 5:
                total_time = pose_time + depth_time
                print(
                    f"Frame {processed_frame_count}: "
                    f"Pose={pose_time:.3f}s ({pose_time/total_time*100:.1f}%), "
                    f"3D={depth_time:.3f}s ({depth_time/total_time*100:.1f}%)"
                )

            # ジャンプ検出
            timestamp = color_frame.get_timestamp()
            jump_result = jump_detector.update(
                processed_frame_count, keypoints_3d, timestamp
            )

            # フレームデータを保存（minimal_dataモードではジャンプ検出時のみ）
            should_save = not args.minimal_data or (
                jump_result is not None
                and jump_result.get("state") in ["jump_start", "jump_end", "jumping"]
            )

            if should_save:
                # キーポイントの辞書を作成
                keypoints_dict = convert_keypoints_to_dict(keypoints_2d, keypoints_3d)

                # 床からの距離をすべてのキーポイントについて追加（床検出が有効な場合）
                if floor_detector and floor_detector.floor_plane is not None:
                    for kp_name, kp_data in keypoints_dict.items():
                        if kp_data.get("3d") and kp_data["3d"]["x"] is not None:
                            kp_coords = (
                                kp_data["3d"]["x"],
                                kp_data["3d"]["y"],
                                kp_data["3d"]["z"],
                            )
                            distance = floor_detector.distance_to_floor(kp_coords)
                            kp_data["distance_to_floor"] = distance
                        else:
                            kp_data["distance_to_floor"] = None

                frame_data = {
                    "frame": frame_num,
                    "processed_frame": processed_frame_count,
                    "timestamp": timestamp,
                    "keypoints": keypoints_dict,
                }

                if jump_result:
                    frame_data["jump_result"] = {
                        "state": jump_result.get("state", "unknown"),
                        "height": jump_result.get("height"),
                        "position": jump_result.get("position"),
                        "jump_type": jump_result.get("jump_type"),
                        "jump_height": jump_result.get("jump_height"),
                        "jump_distance": jump_result.get("jump_distance"),
                    }

                all_frames_data.append(frame_data)

            # 可視化フレームを生成
            if not args.no_video:
                # 統計情報を取得
                statistics = jump_detector.get_statistics()

                # 可視化フレームを生成（キーポイントとスケルトンのみ）
                vis_frame = visualizer.draw_frame(
                    color_image, keypoints_2d, jump_result, statistics
                )

                # 動画ライターを初期化
                if video_writer is None:
                    video_path = output_dir / "jump_visualization.mp4"
                    height, width = vis_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, video_fps, (width, height)
                    )
                    print(f"Initialized video writer: {video_path}")

                video_writer.write(vis_frame)

            # 進捗表示
            current_time = time.time()
            if (
                processed_frame_count == 1
                or processed_frame_count % 10 == 0
                or (current_time - last_progress_time) >= 5.0
            ):
                elapsed_time = current_time - start_time
                if processed_frame_count > 0 and elapsed_time > 0:
                    fps = processed_frame_count / elapsed_time
                    print(
                        f"Processed {processed_frame_count} frames (total frames read: {frame_num}, skipped: {skipped_frame_count}) | "
                        f"Speed: {fps:.1f} fps | Elapsed: {elapsed_time:.0f}s"
                    )
                    last_progress_time = current_time

        elapsed_time = time.time() - start_time
        print(
            f"\nFinished processing {processed_frame_count} frames (total: {frame_num}, skipped: {skipped_frame_count})"
        )
        print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        if processed_frame_count > 0:
            print(f"Average speed: {processed_frame_count/elapsed_time:.1f} fps")

        # 統計情報を取得
        statistics = jump_detector.get_statistics()
        trajectory = jump_detector.get_trajectory()

        print("\n" + "=" * 60)
        print("Analysis Results")
        print("=" * 60)
        print(f"Total jumps detected: {statistics['total_jumps']}")
        print(f"  - Vertical jumps: {statistics['vertical_jumps']}")
        print(f"  - Horizontal jumps: {statistics['horizontal_jumps']}")
        print(f"Max height: {statistics['max_height'] * 100:.1f} cm")
        print(f"Max distance: {statistics['max_distance'] * 100:.1f} cm")
        print(f"Average height: {statistics['avg_height'] * 100:.1f} cm")
        print(f"Average distance: {statistics['avg_distance'] * 100:.1f} cm")
        if statistics.get("max_air_time", 0) > 0:
            print(
                f"Max air time: {statistics['max_air_time'] * 1000:.1f} ms ({statistics['max_air_time']:.3f} s)"
            )
        if statistics.get("avg_air_time", 0) > 0:
            print(
                f"Average air time: {statistics.get('avg_air_time', 0) * 1000:.1f} ms ({statistics.get('avg_air_time', 0):.3f} s)"
            )
        print()

        # 検出されたジャンプの詳細情報を表示
        if statistics.get("jumps"):
            print("=" * 60)
            print("Detected Jump Details:")
            print("=" * 60)
            for i, jump in enumerate(statistics["jumps"], 1):
                print(f"\nJump #{i}:")
                print(f"  Type: {jump['jump_type']}")
                print(f"  Height: {jump['height'] * 100:.1f} cm")
                print(f"  Distance: {jump['distance'] * 100:.1f} cm")
                if jump.get("air_time"):
                    print(
                        f"  Air time: {jump['air_time'] * 1000:.1f} ms ({jump['air_time']:.3f} s)"
                    )
                print(
                    f"  Frames: {jump.get('frame_start', 'N/A')} → {jump.get('frame_takeoff', 'N/A')} → {jump.get('frame_end', 'N/A')}"
                )
                if jump.get("start_position") and jump.get("end_position"):
                    start_pos = jump["start_position"]
                    end_pos = jump["end_position"]
                    if (
                        start_pos
                        and end_pos
                        and len(start_pos) >= 2
                        and len(end_pos) >= 2
                    ):
                        # XZ平面での距離を計算
                        x_diff = end_pos[0] - start_pos[0]
                        z_diff = (
                            end_pos[2] - start_pos[2]
                            if len(end_pos) > 2 and len(start_pos) > 2
                            else 0
                        )
                        actual_distance = np.sqrt(x_diff**2 + z_diff**2) * 100
                        print(
                            f"  Start position: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2] if len(start_pos) > 2 else 'N/A'})"
                        )
                        print(
                            f"  End position: ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2] if len(end_pos) > 2 else 'N/A'})"
                        )
                        print(f"  Actual XZ distance: {actual_distance:.1f} cm")
                print(f"  Reported distance: {jump['distance'] * 100:.1f} cm")
            print("=" * 60)
            print()

        # キーポイント変動性分析（床検出が有効な場合）
        if floor_detector and not args.no_floor_detection:
            print("=" * 60)
            print("キーポイント変動性分析")
            print("=" * 60)

            # 最もジャンプ検出に敏感なキーポイントを特定
            sensitive_keypoints = jump_detector.get_most_jump_sensitive_keypoints(
                min_samples=10, top_n=10
            )

            if sensitive_keypoints:
                print(
                    "\nジャンプ検出に最も敏感なキーポイント（歩行時との違いが明確な順）:"
                )
                print(
                    f"{'キーポイント':<15} {'歩行時SD':<10} {'ジャンプ時SD':<12} {'変動比':<10} "
                    f"{'検出感度':<10} {'最大変位(m)':<12} {'サンプル数':<10}"
                )
                print("-" * 95)
                for kp_name, stats in sensitive_keypoints:
                    walking_std_str = (
                        f"{stats['walking_std']:.3f}"
                        if stats["walking_std"] is not None
                        else "N/A"
                    )
                    jumping_std_str = (
                        f"{stats['jumping_std']:.3f}"
                        if stats["jumping_std"] is not None
                        else "N/A"
                    )
                    ratio_str = (
                        f"{stats['jump_walk_ratio']:.2f}"
                        if stats["jump_walk_ratio"] is not None
                        else "N/A"
                    )
                    sensitivity_str = (
                        f"{stats['jump_sensitivity']:.3f}"
                        if stats["jump_sensitivity"] is not None
                        else "N/A"
                    )

                    print(
                        f"{kp_name:<15} {walking_std_str:<10} {jumping_std_str:<12} {ratio_str:<10} "
                        f"{sensitivity_str:<10} {stats['range']:<12.3f} "
                        f"({stats['walking_samples']}/{stats['jumping_samples']})"
                    )

                # CSV出力
                import csv

                variability_csv_path = output_dir / "keypoint_variability.csv"
                with open(variability_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Keypoint",
                            "Mean (m)",
                            "Std (m)",
                            "Min (m)",
                            "Max (m)",
                            "Range (m)",
                            "CV",
                            "Valid Samples",
                            "Walking Mean (m)",
                            "Walking Std (m)",
                            "Walking Samples",
                            "Jumping Mean (m)",
                            "Jumping Std (m)",
                            "Jumping Samples",
                            "Jump/Walk Ratio",
                            "Jump Sensitivity",
                        ]
                    )
                    all_stats = jump_detector.analyze_keypoint_variability()
                    for kp_name in sorted(all_stats.keys()):
                        stats = all_stats[kp_name]
                        writer.writerow(
                            [
                                kp_name,
                                f"{stats['mean']:.4f}",
                                f"{stats['std']:.4f}",
                                f"{stats['min']:.4f}",
                                f"{stats['max']:.4f}",
                                f"{stats['range']:.4f}",
                                f"{stats['cv']:.4f}",
                                stats["valid_samples"],
                                (
                                    f"{stats['walking_mean']:.4f}"
                                    if stats["walking_mean"] is not None
                                    else ""
                                ),
                                (
                                    f"{stats['walking_std']:.4f}"
                                    if stats["walking_std"] is not None
                                    else ""
                                ),
                                stats["walking_samples"],
                                (
                                    f"{stats['jumping_mean']:.4f}"
                                    if stats["jumping_mean"] is not None
                                    else ""
                                ),
                                (
                                    f"{stats['jumping_std']:.4f}"
                                    if stats["jumping_std"] is not None
                                    else ""
                                ),
                                stats["jumping_samples"],
                                (
                                    f"{stats['jump_walk_ratio']:.4f}"
                                    if stats["jump_walk_ratio"] is not None
                                    else ""
                                ),
                                f"{stats['jump_sensitivity']:.4f}",
                            ]
                        )
                print(f"\nキーポイント変動性CSV saved to: {variability_csv_path}")

                # ジャンプ遷移分析（始まりと終わりを捉えるキーポイント）
                print("=" * 60)
                print("ジャンプ遷移分析（始まりと終わりを捉えるキーポイント）")
                print("=" * 60)

                transitions = jump_detector.analyze_jump_transitions()

                if transitions["takeoff_keypoints"]:
                    print("\n【離陸（ジャンプ開始）時に急激に変化するキーポイント】:")
                    print(
                        f"{'キーポイント':<15} {'平均変化量(m)':<15} {'標準偏差(m)':<15} {'スコア':<12} {'サンプル数':<10}"
                    )
                    print("-" * 75)
                    for kp_name, stats in transitions["takeoff_keypoints"][:10]:
                        print(
                            f"{kp_name:<15} {stats['avg_change']:<15.4f} {stats['std_change']:<15.4f} "
                            f"{stats['score']:<12.2f} {stats['samples']:<10}"
                        )
                else:
                    print("\n【離陸時の分析】: データが不足しています")

                if transitions["landing_keypoints"]:
                    print("\n【着地（ジャンプ終了）時に急激に変化するキーポイント】:")
                    print(
                        f"{'キーポイント':<15} {'平均変化量(m)':<15} {'標準偏差(m)':<15} {'スコア':<12} {'サンプル数':<10}"
                    )
                    print("-" * 75)
                    for kp_name, stats in transitions["landing_keypoints"][:10]:
                        print(
                            f"{kp_name:<15} {stats['avg_change']:<15.4f} {stats['std_change']:<15.4f} "
                            f"{stats['score']:<12.2f} {stats['samples']:<10}"
                        )
                else:
                    print("\n【着地時の分析】: データが不足しています")

                print()
            else:
                print("キーポイント変動性分析: 十分なデータがありません")
            print()

        # JSONファイルに保存
        json_output = {"frames": all_frames_data, "statistics": statistics}
        json_path = output_dir / "keypoints_3d.json"
        save_json(json_output, str(json_path))

        # キーポイントのX, Y, Z座標時系列グラフを作成
        plot_keypoint_coordinate_timeline(all_frames_data, output_dir, floor_detector)

        # ジャンプ軌跡の可視化画像を作成
        plot_jump_trajectory(trajectory, statistics, output_dir, floor_detector)

        # CSVファイルに保存
        csv_path = output_dir / "jump_statistics.csv"
        save_csv(statistics, trajectory, str(csv_path))

        # 可視化動画を完成
        if not args.no_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_path}")

        # 3Dキーポイントアニメーションを生成
        if not args.no_3d_animation:
            print("\nGenerating 3D keypoint animation...")
            animation_path = output_dir / "keypoints_3d_animation.gif"
            if args.interactive_3d:
                # インタラクティブモードで表示 + ファイルも保存
                if create_3d_keypoint_animation(
                    str(json_path), str(animation_path), fps=30, interactive=True
                ):
                    if Path(animation_path).exists():
                        print(f"3D keypoint animation saved to: {animation_path}")
                else:
                    print("Warning: Failed to create 3D keypoint animation")
            else:
                # ファイルとして保存
                if create_3d_keypoint_animation(
                    str(json_path), str(animation_path), fps=30, interactive=False
                ):
                    print(f"3D keypoint animation saved to: {animation_path}")
                else:
                    print("Warning: Failed to create 3D keypoint animation")

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {args.output}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        bag_reader.stop()
        if bag_reader_cam1:
            bag_reader_cam1.stop()


if __name__ == "__main__":
    main()
