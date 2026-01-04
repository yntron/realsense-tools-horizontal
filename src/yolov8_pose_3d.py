"""
YOLOv8-Pose 3D Utilities

YOLOv8-Poseモデルの読み込み、2D keypoints検出、スケルトン描画機能を提供
OpenPoseDetectorと同様のインターフェースを提供
"""

import cv2
import numpy as np
import os
import sys

# YOLOv8-Poseを使用
try:
    from ultralytics import YOLO
    import torch

    YOLOV8_AVAILABLE = True
except ImportError:
    YOLOV8_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")


# COCO形式のkeypoints（18 points - OpenPose互換）
# OpenPoseと同じ順序で定義（互換性のため）
COCO_KEYPOINTS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
]

# YOLOv8-Poseの17 keypoints形式（COCO標準）
YOLOV8_KEYPOINTS = [
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
]

# COCO形式の接続ペア（18 keypoints形式 - OpenPose互換）
COCO_PAIRS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],  # 肩と腕
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],  # 体幹と脚
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],  # 顔
]


class YOLOv8PoseDetector:
    """YOLOv8-Pose検出器（ultralytics使用）"""

    def __init__(self, model_name="models/yolov8x-pose.pt", model_dir="models"):
        """
        Args:
            model_name: 使用するモデル名
                - yolov8n-pose.pt: nano（超高速、やや精度低下）
                - yolov8s-pose.pt: small（高速、バランス型）
                - yolov8m-pose.pt: medium（中速、高精度）
                - yolov8l-pose.pt: large（やや低速、高精度）
                - yolov8x-pose.pt: extra large（最高精度） - **推奨（デフォルト）**
            model_dir: モデルファイルを保存するディレクトリ
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.model = None
        self.device = self._get_device()

    def _get_device(self):
        """使用可能なデバイスを取得"""
        try:
            import torch

            if torch.cuda.is_available():
                # CUDAが利用可能な場合、実際にアクセスできるか確認
                try:
                    device = "cuda:0"
                    # CUDAデバイスにアクセスして動作確認
                    test_tensor = torch.zeros(1).to(device)
                    del test_tensor
                    # CUDAデバイス情報を表示
                    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
                    print(f"  CUDA version: {torch.version.cuda}")
                    return device
                except Exception as e:
                    # CUDAアクセスに失敗した場合はCPUにフォールバック
                    print(f"  Warning: CUDA device access failed: {e}")
                    print(f"  Falling back to CPU mode")
                    return "cpu"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon GPU
                try:
                    device = "mps"
                    # MPSデバイスにアクセスして動作確認
                    test_tensor = torch.zeros(1).to(device)
                    del test_tensor
                    print(f"  Apple Silicon GPU (MPS) available")
                    return device
                except Exception as e:
                    # MPSアクセスに失敗した場合はCPUにフォールバック
                    print(f"  Warning: MPS device access failed: {e}")
                    print(f"  Falling back to CPU mode")
                    return "cpu"
            else:
                print(f"  Using CPU mode (no GPU available)")
                return "cpu"
        except Exception as e:
            print(f"  Warning: Device detection failed: {e}")
            print(f"  Falling back to CPU mode")
            return "cpu"

    def load_model(self):
        """YOLOv8-Poseモデルを読み込み"""
        if not YOLOV8_AVAILABLE:
            print("Error: ultralytics is not available.")
            print("Install with: pip install ultralytics")
            return False

        try:
            print(f"Loading YOLOv8-Pose model: {self.model_name}")
            print(f"Using device: {self.device}")

            # モデルディレクトリが存在しない場合は作成
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
                print(f"Created model directory: {self.model_dir}")

            # モデルファイルのパスを決定
            if os.path.exists(self.model_name):
                # フルパスで存在する場合
                model_path = self.model_name
            elif os.path.exists(os.path.join(self.model_dir, self.model_name)):
                # modelsディレクトリ内に存在する場合
                model_path = os.path.join(self.model_dir, self.model_name)
            else:
                # モデルが存在しない場合、modelsディレクトリ内のパスを指定して自動ダウンロード
                # これにより、モデルはmodelsディレクトリ内に保存される
                model_path = os.path.join(self.model_dir, self.model_name)
                print(f"Model not found. Will download to: {model_path}")

            # YOLOv8-Poseモデルを読み込み（存在しない場合は自動ダウンロード）
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                # CUDA/MPSデバイスでのモデルロードに失敗した場合はCPUにフォールバック
                if self.device != "cpu" and ("CUDA" in str(e) or "cuda" in str(e).lower() or "MPS" in str(e) or "mps" in str(e).lower()):
                    print(f"Warning: Failed to load model on {self.device}: {e}")
                    print(f"Falling back to CPU mode")
                    self.device = "cpu"
                    self.model = YOLO(model_path)
                else:
                    raise

            # torch.compileはultralyticsとの互換性問題があるため無効化
            # 代わりに、YOLOv8の推論時にhalf精度とdevice指定で高速化を行う
            # 注意: torch.compileを使用すると'OptimizedModule'オブジェクトが
            # subscriptableでなくなり、ultralytics内部のコードでエラーが発生する

            print(f"YOLOv8-Pose model loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load YOLOv8-Pose model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def detect_keypoints(self, image):
        """
        画像からkeypointsを検出

        Args:
            image: 入力画像（BGR形式）

        Returns:
            list: 検出されたkeypointsのリスト [(x, y, confidence), ...] （18要素）
        """
        if self.model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return None

        h, w = image.shape[:2]

        try:
            # YOLOv8-PoseはRGB形式を期待（BGR -> RGB変換）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 推論実行（最適化オプション）
            # imgsz: 入力画像サイズに応じて最適化（大きな画像は640、小さい画像はそのまま）
            # conf: 信頼度閾値（デフォルト）
            # half: 半精度推論（GPU使用時、約2倍高速化）
            # max_det: 最大検出数（人物が1人の場合は1に制限して高速化）
            is_cuda = self.device.startswith("cuda") if isinstance(self.device, str) else False
            is_mps = self.device == "mps"
            half = is_cuda or is_mps  # GPU/MPS使用時は半精度推論を有効化

            # 入力画像サイズに応じてimgszを動的に調整（推論速度の最適化）
            max_dim = max(h, w)
            # 640ピクセル以下の場合はそのまま、それ以上は640にリサイズ
            imgsz = min(640, max(320, (max_dim // 32) * 32))  # 32の倍数に丸める

            # 推論実行（CUDAエラー時はCPUにフォールバック）
            # 複数人物を検出して右側の人物を選択するため、max_det=5に設定
            try:
                results = self.model(
                    image_rgb,
                    verbose=False,
                    device=self.device,
                    half=half,  # 半精度推論（GPU/MPS使用時、約2倍高速化）
                    imgsz=imgsz,  # 動的に最適化された入力サイズ
                    max_det=5,  # 複数人物を検出（右側の人物を選択するため）
                    conf=0.25,  # 信頼度閾値（デフォルトより低めに設定して検出確率を上げる）
                )
            except Exception as e:
                # CUDA/MPSデバイスでの推論に失敗した場合はCPUにフォールバック
                if self.device != "cpu" and ("CUDA" in str(e) or "cuda" in str(e).lower() or "MPS" in str(e) or "mps" in str(e).lower()):
                    print(f"Warning: Inference failed on {self.device}: {e}")
                    print(f"Falling back to CPU mode for inference")
                    self.device = "cpu"
                    # CPUモードでは半精度推論を無効化
                    results = self.model(
                        image_rgb,
                        verbose=False,
                        device=self.device,
                        half=False,  # CPUでは半精度推論を無効化
                        imgsz=imgsz,
                        max_det=5,
                        conf=0.25,
                    )
                else:
                    raise

            if not results or len(results) == 0:
                return None

            # 最初の検出結果を取得（最大信頼度の人物）
            result = results[0]

            # keypointsを取得
            if result.keypoints is None or len(result.keypoints.data) == 0:
                return None

            # 複数人物が検出された場合は、最大のバウンディングボックスの人物を選択
            keypoints_data = (
                result.keypoints.data.cpu().numpy()
            )  # Shape: (num_people, 17, 3)

            if len(keypoints_data) == 0:
                return None

            # 最初の人物のkeypointsを使用（必要に応じて最大bboxサイズの人物を選択）
            if len(keypoints_data) > 1:
                # 複数人物の場合は、最も右側（X座標が大きい）の人物を選択
                # 各人物の腰（LHip=11, RHip=12）の中点X座標を計算
                person_x_positions = []
                for person_kpts in keypoints_data:
                    # LHip(11)とRHip(12)の中点を計算
                    lhip = person_kpts[11]
                    rhip = person_kpts[12]
                    if lhip[2] > 0 and rhip[2] > 0:
                        mid_x = (lhip[0] + rhip[0]) / 2
                    elif lhip[2] > 0:
                        mid_x = lhip[0]
                    elif rhip[2] > 0:
                        mid_x = rhip[0]
                    else:
                        # 腰が検出されない場合は全キーポイントの平均X座標を使用
                        valid_x = person_kpts[person_kpts[:, 2] > 0, 0]
                        mid_x = np.mean(valid_x) if len(valid_x) > 0 else 0
                    person_x_positions.append(mid_x)
                # 最も右側（X座標が最大）の人物を選択
                best_person_idx = np.argmax(person_x_positions)
                keypoints_17 = keypoints_data[best_person_idx]
            else:
                keypoints_17 = keypoints_data[0]

            # YOLOv8-Poseの出力形式: (17, 3) - (x, y, confidence)
            # 18 keypoints形式（COCO互換）に変換
            keypoints_18 = self._convert_17_to_18(keypoints_17, w, h)

            return keypoints_18

        except Exception as e:
            print(f"Error during YOLOv8-Pose inference: {e}")
            import traceback

            traceback.print_exc()
            return None

    def detect_keypoints_multi(self, image, max_det=5, conf=0.25):
        """
        画像から複数人物のkeypointsを検出（18キーポイント形式）

        Args:
            image: BGR画像
            max_det: 最大検出人数
            conf: 信頼度閾値

        Returns:
            list[list[(x, y, conf)]] または None
        """
        if self.model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return None

        h, w = image.shape[:2]

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            is_cuda = self.device.startswith("cuda") if isinstance(self.device, str) else False
            is_mps = self.device == "mps"
            half = is_cuda or is_mps

            max_dim = max(h, w)
            imgsz = min(640, max(320, (max_dim // 32) * 32))

            try:
                results = self.model(
                    image_rgb,
                    verbose=False,
                    device=self.device,
                    half=half,
                    imgsz=imgsz,
                    max_det=int(max(1, max_det)),
                    conf=conf,
                )
            except Exception as e:
                if self.device != "cpu" and ("cuda" in str(e).lower() or "mps" in str(e).lower()):
                    self.device = "cpu"
                    results = self.model(
                        image_rgb,
                        verbose=False,
                        device=self.device,
                        half=False,
                        imgsz=imgsz,
                        max_det=int(max(1, max_det)),
                        conf=conf,
                    )
                else:
                    raise

            if not results or len(results) == 0:
                return None

            result = results[0]
            if result.keypoints is None or len(result.keypoints.data) == 0:
                return None

            keypoints_data = result.keypoints.data.cpu().numpy()  # (num_people, 17, 3)
            if len(keypoints_data) == 0:
                return None

            persons_18 = []
            for person_kpts in keypoints_data:
                persons_18.append(self._convert_17_to_18(person_kpts, w, h))

            return persons_18 if persons_18 else None

        except Exception as e:
            print(f"Error during YOLOv8-Pose multi-person inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_17_to_18(self, keypoints_17, w, h):
        """
        YOLOv8-Poseの17 keypointsを18 keypoints形式（COCO互換）に変換

        YOLOv8-Pose (17 keypoints - COCO標準):
        0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar,
        5: LShoulder, 6: RShoulder,
        7: LElbow, 8: RElbow,
        9: LWrist, 10: RWrist,
        11: LHip, 12: RHip,
        13: LKnee, 14: RKnee,
        15: LAnkle, 16: RAnkle

        COCO (18 keypoints - OpenPose互換):
        0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
        5: LShoulder, 6: LElbow, 7: LWrist,
        8: RHip, 9: RKnee, 10: RAnkle,
        11: LHip, 12: LKnee, 13: LAnkle,
        14: REye, 15: LEye, 16: REar, 17: LEar
        """
        # 18要素のリストを作成（すべてNone）
        keypoints_18 = [(None, None, 0.0) for _ in range(18)]

        # YOLOv8-Pose 17 -> COCO 18 マッピング
        mapping = [
            (0, 0),  # Nose -> Nose
            (1, 15),  # LEye -> LEye
            (2, 14),  # REye -> REye
            (3, 17),  # LEar -> LEar
            (4, 16),  # REar -> REar
            (5, 5),  # LShoulder -> LShoulder
            (6, 2),  # RShoulder -> RShoulder
            (7, 6),  # LElbow -> LElbow
            (8, 3),  # RElbow -> RElbow
            (9, 7),  # LWrist -> LWrist
            (10, 4),  # RWrist -> RWrist
            (11, 11),  # LHip -> LHip
            (12, 8),  # RHip -> RHip
            (13, 12),  # LKnee -> LKnee
            (14, 9),  # RKnee -> RKnee
            (15, 13),  # LAnkle -> LAnkle
            (16, 10),  # RAnkle -> RAnkle
        ]

        for yolo_idx, coco_idx in mapping:
            if yolo_idx < len(keypoints_17):
                kp = keypoints_17[yolo_idx]
                x = float(kp[0])
                y = float(kp[1])
                confidence = float(kp[2]) if len(kp) > 2 else 1.0

                # 有効なkeypointかチェック（信頼度 > 0）
                if confidence > 0 and 0 <= x < w and 0 <= y < h:
                    keypoints_18[coco_idx] = (x, y, confidence)

        # Neckを計算（両肩の中点）
        if (
            keypoints_18[5][0] is not None
            and keypoints_18[2][0] is not None
            and keypoints_18[5][1] is not None
            and keypoints_18[2][1] is not None
        ):
            neck_x = (keypoints_18[5][0] + keypoints_18[2][0]) / 2
            neck_y = (keypoints_18[5][1] + keypoints_18[2][1]) / 2
            neck_conf = min(keypoints_18[5][2], keypoints_18[2][2])
            keypoints_18[1] = (neck_x, neck_y, neck_conf)

        return keypoints_18

    def draw_skeleton(self, image, keypoints, threshold=0.1):
        """
        スケルトンを描画

        Args:
            image: 入力画像（BGR形式）
            keypoints: keypointsのリスト [(x, y, confidence), ...]
            threshold: keypointを描画する最小信頼度

        Returns:
            numpy.ndarray: スケルトンが描画された画像
        """
        skeleton_image = image.copy()

        if keypoints is None:
            return skeleton_image

        # 接続ペアを描画
        for pair in COCO_PAIRS:
            idx1, idx2 = pair
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kp1 = keypoints[idx1]
                kp2 = keypoints[idx2]

                if (
                    kp1[0] is not None
                    and kp1[1] is not None
                    and kp2[0] is not None
                    and kp2[1] is not None
                    and kp1[2] > threshold
                    and kp2[2] > threshold
                ):
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(skeleton_image, pt1, pt2, (0, 255, 0), 2)

        # keypointsを描画
        for i, kp in enumerate(keypoints):
            if kp[0] is not None and kp[1] is not None and kp[2] > threshold:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(skeleton_image, pt, 5, (0, 0, 255), -1)
                cv2.circle(skeleton_image, pt, 3, (255, 255, 255), -1)

        return skeleton_image
