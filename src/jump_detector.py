"""
Jump Detector and Measurement

ジャンプ検出アルゴリズム、高さ・距離計算、軌跡計算機能を提供
床検出ベースの高精度ジャンプ検出に対応
"""

import numpy as np
from collections import deque


class JumpDetector:
    """ジャンプ検出・測定クラス"""

    def __init__(self, threshold_vertical=0.05, threshold_horizontal=0.1, min_frames=10, floor_detector=None, use_floor_detection=True, min_jump_height=0.10, min_air_time=0.20):
        """
        Args:
            threshold_vertical: 垂直ジャンプ検出の閾値（メートル、床検出不使用時）
            threshold_horizontal: 水平ジャンプ検出の閾値（メートル）
            min_frames: ジャンプと認識する最小フレーム数
            floor_detector: FloorDetectorインスタンス（床検出使用時）
            use_floor_detection: 床検出を使用するかどうか（デフォルト: True）
            min_jump_height: ジャンプとして認識する最小高さ（メートル、デフォルト: 10cm）
            min_air_time: ジャンプとして認識する最小滞空時間（秒、デフォルト: 0.2秒）
        """
        self.threshold_vertical = threshold_vertical
        self.threshold_horizontal = threshold_horizontal
        self.min_frames = min_frames
        self.floor_detector = floor_detector
        self.use_floor_detection = use_floor_detection and (floor_detector is not None)
        self.min_jump_height = min_jump_height  # 最小ジャンプ高さ（10cm）
        self.min_air_time = min_air_time  # 最小滞空時間（0.2秒 = 200ms）

        # 履歴データ
        self.height_history = deque(maxlen=30)  # 過去30フレームの高さ（深度Z、後方互換性のため残す）
        self.vertical_history = deque(maxlen=30)  # 過去30フレームの垂直位置（Y座標、RealSense座標系では下が正）
        self.position_history = deque(maxlen=30)  # 過去30フレームの位置
        self.left_foot_on_floor = deque(maxlen=5)  # 左足が床についているか
        self.right_foot_on_floor = deque(maxlen=5)  # 右足が床についているか
        
        # 足首の高さ履歴（床からの距離）- 立位時の基準値を記録するため
        self.ankle_height_history = deque(maxlen=30)  # 足首の床からの距離の履歴
        self.baseline_ankle_height = None  # 立位時の基準足首高さ
        
        # 全キーポイントの床からの距離履歴（変動性分析用）
        self.keypoint_distance_history = {}  # {keypoint_name: deque(maxlen=1000), ...}
        
        # ジャンプ遷移時（離陸・着地）の変化を記録
        self.takeoff_transitions = []  # [(frame_num, keypoint_distances), ...]
        self.landing_transitions = []  # [(frame_num, keypoint_distances), ...]

        # ジャンプ検出状態
        self.jump_state = "ground"  # "ground", "takeoff", "airborne", "landing"
        self._initial_state = True  # 初期状態フラグ
        self.jump_start_frame = None
        self.jump_start_timestamp = None
        self.jump_start_position = None
        self.jump_takeoff_frame = None
        self.jump_takeoff_timestamp = None
        self.jump_takeoff_position = None
        self.jump_takeoff_height = None  # 離陸時の床からの高さ
        self.jump_end_frame = None
        self.jump_end_timestamp = None
        self.jump_end_position = None
        self.jump_max_height = 0.0
        self.jump_max_distance = 0.0

        # レガシーモード用: Y座標ベースの高さ追跡
        self._jump_min_y = None  # ジャンプ中の最小Y値（最高点）
        self._jump_start_y = None  # ジャンプ開始時のY値

        # 検出されたジャンプのリスト
        self.detected_jumps = []

        # 軌跡データ
        self.trajectory = []

        # クールダウン: 着地後の誤検出防止
        self._cooldown_frames = 0
        self._cooldown_duration = 30  # 着地後30フレーム（約1秒@30fps）は新しいジャンプを検出しない

        # 滞空中の連続フレーム数をカウント（走行中の誤検出防止）
        self._airborne_frame_count = 0
        self._min_airborne_frames = 10  # 最低10フレーム（約0.33秒@30fps）連続で滞空している必要がある

    def update(self, frame_num, keypoints_3d, timestamp=None):
        """
        新しいフレームのデータを更新し、ジャンプを検出

        Args:
            frame_num: フレーム番号
            keypoints_3d: 3D keypointsの辞書 {keypoint_name: (x, y, z), ...}
            timestamp: タイムスタンプ（オプション）

        Returns:
            dict: 検出結果と測定値
        """
        if keypoints_3d is None:
            return None

        # 基準点として腰（MidHip）または足首の平均を使用
        # 腰がない場合は、左右の股関節の中点を使用
        mid_hip = None
        left_hip = keypoints_3d.get("LHip")
        right_hip = keypoints_3d.get("RHip")

        if left_hip and right_hip and left_hip[2] is not None and right_hip[2] is not None:
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2 if left_hip[0] is not None and right_hip[0] is not None else None,
                (left_hip[1] + right_hip[1]) / 2 if left_hip[1] is not None and right_hip[1] is not None else None,
                (left_hip[2] + right_hip[2]) / 2 if left_hip[2] is not None and right_hip[2] is not None else None
            )

        # 足首の平均も計算
        left_ankle = keypoints_3d.get("LAnkle")
        right_ankle = keypoints_3d.get("RAnkle")
        mid_ankle = None

        if left_ankle and right_ankle and left_ankle[2] is not None and right_ankle[2] is not None:
            mid_ankle = (
                (left_ankle[0] + right_ankle[0]) / 2 if left_ankle[0] is not None and right_ankle[0] is not None else None,
                (left_ankle[1] + right_ankle[1]) / 2 if left_ankle[1] is not None and right_ankle[1] is not None else None,
                (left_ankle[2] + right_ankle[2]) / 2 if left_ankle[2] is not None and right_ankle[2] is not None else None
            )

        # 基準点（優先順位: 腰 > 足首）
        reference_point = mid_hip if mid_hip and mid_hip[2] is not None else mid_ankle

        if reference_point is None or reference_point[2] is None:
            return None

        x, y, z = reference_point

        # 床検出使用時: 両足首の床からの距離を記録
        # 注意: 足首のキーポイントは床から10cm程度離れているのが正常（立位時）
        # 実際の接触判定は行わず、足首の高さの「変化」を監視する
        both_feet_on_floor = False
        left_ankle_height = None
        right_ankle_height = None
        avg_ankle_height = None
        
        if self.use_floor_detection and self.floor_detector and self.floor_detector.floor_plane is not None:
            # 両足首の床からの距離を計算（接触判定ではなく距離のみ）
            if left_ankle and left_ankle[0] is not None and left_ankle[1] is not None and left_ankle[2] is not None:
                left_ankle_height = self.floor_detector.distance_to_floor(left_ankle)
                
            if right_ankle and right_ankle[0] is not None and right_ankle[1] is not None and right_ankle[2] is not None:
                right_ankle_height = self.floor_detector.distance_to_floor(right_ankle)
            
            # 両足首の平均高さを計算
            if left_ankle_height is not None and right_ankle_height is not None:
                avg_ankle_height = (left_ankle_height + right_ankle_height) / 2.0
            elif left_ankle_height is not None:
                avg_ankle_height = left_ankle_height
            elif right_ankle_height is not None:
                avg_ankle_height = right_ankle_height
            
            # 足首の高さ履歴に追加
            if avg_ankle_height is not None:
                self.ankle_height_history.append(avg_ankle_height)
            
            # 基準足首高さを設定（最初の数フレームの最低値）
            if self.baseline_ankle_height is None and len(self.ankle_height_history) >= 5:
                # 最初の5フレームの最低値を基準とする
                self.baseline_ankle_height = min(list(self.ankle_height_history)[:5])
            elif self.baseline_ankle_height is not None and len(self.ankle_height_history) > 0:
                # 基準値を更新（より低い値が見つかった場合のみ）
                recent_min = min(list(self.ankle_height_history)[-10:])  # 過去10フレームの最低値
                if recent_min < self.baseline_ankle_height:
                    self.baseline_ankle_height = recent_min
            
            # 足首が「床に近い」状態を判定（基準値から一定範囲内）
            # 立位時は足首が基準値付近にあるはず
            if self.baseline_ankle_height is not None and avg_ankle_height is not None:
                # 基準値から8cm以内なら「床に近い（立位）」と判定（5cm→8cmに緩和、水平移動時の振動を許容）
                floor_threshold = 0.08
                left_on_floor = abs(left_ankle_height - self.baseline_ankle_height) <= floor_threshold if left_ankle_height is not None else False
                right_on_floor = abs(right_ankle_height - self.baseline_ankle_height) <= floor_threshold if right_ankle_height is not None else False
                both_feet_on_floor = left_on_floor and right_on_floor
            else:
                left_on_floor = False
                right_on_floor = False
                both_feet_on_floor = False
            
            self.left_foot_on_floor.append(left_on_floor)
            self.right_foot_on_floor.append(right_on_floor)
            
        elif self.use_floor_detection and (self.floor_detector is None or self.floor_detector.floor_plane is None):
            # 床検出がまだ完了していない場合、従来方式にフォールバック
            self.use_floor_detection = False

        # 履歴を更新
        self.height_history.append(z)  # 深度（後方互換性のため残す）
        self.vertical_history.append(y)  # 垂直位置（RealSense座標系では下が正、上にジャンプするとYが減少）
        self.position_history.append((x, y, z))

        # 床からの距離をすべてのキーポイントについて計算
        keypoint_distances_to_floor = {}
        if self.use_floor_detection and self.floor_detector and self.floor_detector.floor_plane is not None:
            for kp_name, kp_coords in keypoints_3d.items():
                if kp_coords and kp_coords[0] is not None and kp_coords[1] is not None and kp_coords[2] is not None:
                    distance = self.floor_detector.distance_to_floor(kp_coords)
                    keypoint_distances_to_floor[kp_name] = distance
                    
                    # 履歴に追加（変動性分析用）
                    if kp_name not in self.keypoint_distance_history:
                        self.keypoint_distance_history[kp_name] = deque(maxlen=1000)
                    # 距離と現在の状態（ground/jumping）を一緒に保存
                    self.keypoint_distance_history[kp_name].append((distance, self.jump_state))
                else:
                    keypoint_distances_to_floor[kp_name] = None
        
        # 軌跡を記録
        self.trajectory.append({
            "frame": frame_num,
            "timestamp": timestamp,
            "position": (x, y, z),
            "keypoints": keypoints_3d,
            "both_feet_on_floor": both_feet_on_floor if self.use_floor_detection else None,
            "keypoint_distances_to_floor": keypoint_distances_to_floor if self.use_floor_detection else None
        })

        # ジャンプ検出
        if self.use_floor_detection:
            # avg_ankle_heightを渡す
            result = self._detect_jump_with_floor(
                frame_num, x, y, z, timestamp, left_ankle, right_ankle, 
                avg_ankle_height if 'avg_ankle_height' in locals() else None,
                keypoints_3d  # 離陸・着地時のキーポイント記録に使用
            )
        else:
            result = self._detect_jump(frame_num, x, y, z, timestamp)

        return result

    def _detect_jump(self, frame_num, x, y, z, timestamp):
        """ジャンプ検出ロジック（従来モード）"""
        # クールダウン中はカウンタを減らす
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1

        if len(self.vertical_history) < self.min_frames:
            return {
                "state": "initializing",
                "frame": frame_num,
                "timestamp": timestamp,
                "height": z,
                "position": (x, y, z)
            }

        # 現在の垂直位置と過去の平均を比較
        # RealSense座標系: Y軸は下向きが正なので、上にジャンプするとYが減少
        current_vertical = y  # 垂直位置（Y座標）
        avg_vertical = np.mean(list(self.vertical_history)[-self.min_frames:-1])

        # 垂直方向の変化（負の値 = 上昇、正の値 = 下降）
        # 表示用に反転: 正の値 = 上昇（ジャンプ）、負の値 = 下降（着地）
        vertical_change = -(current_vertical - avg_vertical)

        # 水平位置の変化（前フレームと比較）- XZ平面での距離
        if len(self.position_history) >= 2:
            prev_pos = self.position_history[-2]
            horizontal_distance = np.sqrt((x - prev_pos[0])**2 + (z - prev_pos[2])**2)
        else:
            horizontal_distance = 0.0

        result = {
            "frame": frame_num,
            "timestamp": timestamp,
            "height": z,
            "position": (x, y, z),
            "vertical_change": vertical_change,  # 垂直方向の変化（正=上昇）
            "horizontal_distance": horizontal_distance,
            "jump_type": None,
            "jump_height": 0.0,
            "jump_distance": 0.0
        }

        # ジャンプ状態の遷移
        if self.jump_state == "ground":
            # クールダウン中は新しいジャンプを検出しない
            if self._cooldown_frames > 0:
                result["state"] = "cooldown"
                return result

            # 地面からジャンプ開始を検出（垂直方向の上昇を検出）
            if vertical_change > self.threshold_vertical:
                self.jump_state = "jumping"
                self.jump_start_frame = frame_num
                self.jump_start_timestamp = timestamp
                self.jump_start_position = (x, y, z)
                # 最小Y値を追跡（RealSense座標系ではYが小さいほど高い位置）
                self._jump_min_y = y  # ジャンプ中の最小Y値（最高点）
                self._jump_start_y = y  # ジャンプ開始時のY値
                self.jump_max_distance = 0.0
                self._airborne_frame_count = 0
                result["state"] = "jump_start"

        elif self.jump_state == "jumping":
            # 滞空フレーム数をカウント
            self._airborne_frame_count += 1

            # ジャンプ中の最高点（最小Y値）を更新
            # RealSense座標系: Y軸は下向きが正なので、Yが小さいほど高い位置
            if y < self._jump_min_y:
                self._jump_min_y = y

            # 水平距離を更新（XZ平面での距離）
            # RealSense座標系: X=左右, Y=上下（垂直）, Z=前後（深度）
            if self.jump_start_position:
                jump_distance = np.sqrt(
                    (x - self.jump_start_position[0])**2 +
                    (z - self.jump_start_position[2])**2
                )
                if jump_distance > self.jump_max_distance:
                    self.jump_max_distance = jump_distance

            # 着地を検出（垂直方向が下降し、元の高さに戻る）
            # vertical_change < 0 は下降を意味し、現在のYが開始時のY付近に戻ったら着地
            if vertical_change < -self.threshold_vertical and current_vertical >= avg_vertical - 0.02:
                # 最低滞空フレーム数をチェック
                if self._airborne_frame_count < self._min_airborne_frames:
                    print(f"  Skipping short airborne (legacy): {self._airborne_frame_count} frames (min={self._min_airborne_frames})")
                    self.jump_state = "ground"
                    self.jump_start_frame = None
                    self.jump_start_position = None
                    self._cooldown_frames = self._cooldown_duration // 2
                    result["state"] = "ground"
                    return result

                self.jump_state = "landed"
                self.jump_end_frame = frame_num
                self.jump_end_timestamp = timestamp
                result["state"] = "jump_end"

                # ジャンプ高さを計算（Y座標の差、RealSense座標系では上がマイナスなので反転）
                # 開始時のY - 最小Y（最高点）= ジャンプ高さ
                jump_height = self._jump_start_y - self._jump_min_y

                # 滞空時間を計算
                air_time = 0.0
                if self.jump_start_timestamp is not None and self.jump_end_timestamp is not None:
                    timestamp_diff = abs(self.jump_end_timestamp - self.jump_start_timestamp)
                    if self.jump_start_timestamp > 1000000000:
                        air_time = timestamp_diff / 1000.0
                    else:
                        air_time = timestamp_diff
                    if air_time > 10.0:
                        air_time = self._airborne_frame_count / 30.0

                # 有効性チェック
                is_valid_jump = (
                    abs(jump_height) >= self.min_jump_height and
                    air_time >= self.min_air_time
                )

                if not is_valid_jump:
                    print(f"  Skipping invalid jump (legacy): height={abs(jump_height)*100:.1f}cm (min={self.min_jump_height*100:.1f}cm), "
                          f"air_time={air_time*1000:.1f}ms (min={self.min_air_time*1000:.1f}ms)")
                    self.jump_state = "ground"
                    self.jump_start_frame = None
                    self.jump_start_position = None
                    self._cooldown_frames = self._cooldown_duration
                    result["state"] = "ground"
                    return result

                # ジャンプの種類を判定
                if self.jump_max_distance < self.threshold_horizontal:
                    result["jump_type"] = "vertical"
                    result["jump_height"] = jump_height
                    result["jump_distance"] = 0.0
                else:
                    result["jump_type"] = "horizontal"
                    result["jump_height"] = jump_height
                    result["jump_distance"] = self.jump_max_distance

                result["air_time"] = air_time

                # 検出されたジャンプを記録
                jump_data = {
                    "frame_start": self.jump_start_frame,
                    "frame_end": frame_num,
                    "jump_type": result["jump_type"],
                    "height": result["jump_height"],
                    "distance": result["jump_distance"],
                    "air_time": air_time,
                    "max_height_y": self._jump_min_y,  # 最高点のY座標（最小Y値）
                    "start_y": self._jump_start_y,  # 開始時のY座標
                    "start_position": self.jump_start_position,
                    "end_position": (x, y, z)
                }
                self.detected_jumps.append(jump_data)

                # 状態をリセット
                self.jump_state = "ground"
                self.jump_start_frame = None
                self.jump_start_position = None
                self._cooldown_frames = self._cooldown_duration
                self._airborne_frame_count = 0

        elif self.jump_state == "landed":
            # 着地後、少し待ってから地面状態に戻る
            if abs(vertical_change) < 0.01:
                self.jump_state = "ground"
                result["state"] = "ground"

        # ジャンプ中の場合の情報を追加
        if self.jump_state == "jumping":
            result["state"] = "jumping"
            # 現在のジャンプ高さ（開始Y - 現在の最小Y）
            if hasattr(self, '_jump_start_y') and hasattr(self, '_jump_min_y'):
                result["jump_height"] = self._jump_start_y - self._jump_min_y
            else:
                result["jump_height"] = 0.0
            result["jump_distance"] = self.jump_max_distance

        return result
    
    def _detect_jump_with_floor(self, frame_num, x, y, z, timestamp, left_ankle, right_ankle, avg_ankle_height=None, keypoints_3d=None):
        """
        床検出ベースのジャンプ検出ロジック
        両足首の高さ変化を基準に正確なジャンプ開始・終了・滞空時間を検出

        Args:
            frame_num: フレーム番号
            x, y, z: 参照点（腰）の3D座標
            timestamp: タイムスタンプ
            left_ankle: 左足首の3D座標
            right_ankle: 右足首の3D座標
            avg_ankle_height: 両足首の平均高さ（床からの距離、メートル）
        """
        # クールダウン中はカウンタを減らす
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1

        if len(self.height_history) < self.min_frames:
            return {
                "state": "initializing",
                "frame": frame_num,
                "timestamp": timestamp,
                "height": z,
                "position": (x, y, z)
            }
        
        # 両足が床についているかの判定（過去数フレームの履歴から）
        both_feet_on_floor_now = (
            len(self.left_foot_on_floor) > 0 and 
            len(self.right_foot_on_floor) > 0 and
            self.left_foot_on_floor[-1] and 
            self.right_foot_on_floor[-1]
        )
        
        # 安定した接触判定（数フレーム連続で床についている）
        # ジャンプ完了後の着地を検出するため、少し厳しめの条件にする
        both_feet_stable_on_floor = (
            len(self.left_foot_on_floor) >= 5 and
            len(self.right_foot_on_floor) >= 5 and
            all(list(self.left_foot_on_floor)[-5:]) and
            all(list(self.right_foot_on_floor)[-5:])
        )
        
        # 両足が離れているかの判定
        both_feet_off_floor = (
            len(self.left_foot_on_floor) > 0 and 
            len(self.right_foot_on_floor) > 0 and
            not self.left_foot_on_floor[-1] and 
            not self.right_foot_on_floor[-1]
        )
        
        # 床からの高さを計算（腰の位置から床までの距離）
        height_above_floor = None
        if self.floor_detector and self.floor_detector.floor_plane is not None:
            # 腰の位置（reference_point）から床までの距離
            mid_hip = (x, y, z)
            if mid_hip[0] is not None and mid_hip[1] is not None and mid_hip[2] is not None:
                height_above_floor = self.floor_detector.distance_to_floor(mid_hip)
        
        # 床からの高さの履歴を保持（高さ変化検出のため）
        if not hasattr(self, 'height_above_floor_history'):
            self.height_above_floor_history = deque(maxlen=10)
        if height_above_floor is not None:
            self.height_above_floor_history.append(height_above_floor)
        
        result = {
            "frame": frame_num,
            "timestamp": timestamp,
            "height": z,
            "position": (x, y, z),
            "height_above_floor": height_above_floor,
            "both_feet_on_floor": both_feet_on_floor_now,
            "jump_type": None,
            "jump_height": 0.0,
            "jump_distance": 0.0,
            "air_time": 0.0
        }
        
        # ジャンプ状態の遷移
        if self.jump_state == "ground":
            # 地面状態: 両足が安定して床についている、または初期状態
            # 初期状態または床に接触している場合は開始位置を記録（準備段階）
            # 初期位置は常に記録（両足が床についていなくても）
            if self.jump_start_position is None:
                # ジャンプ開始位置を記録（準備段階）
                self.jump_start_frame = frame_num
                self.jump_start_timestamp = timestamp
                self.jump_start_position = (x, y, z)
            elif both_feet_stable_on_floor:
                # 床に安定して接触している場合は開始位置を更新
                self.jump_start_frame = frame_num
                self.jump_start_timestamp = timestamp
                self.jump_start_position = (x, y, z)
            
            # 初期状態を解除（開始位置が記録されてから数フレーム後）
            if self._initial_state and self.jump_start_position is not None:
                if len(self.height_history) >= 5:
                    self._initial_state = False
            
            # 離陸検出（初期状態ではない場合、かつクールダウン中でない場合のみ）
            # 足首の高さ変化ベースの離陸検出
            # 基準足首高さから上昇した場合に離陸と判定
            if not self._initial_state and self._cooldown_frames == 0:
                # 足首の高さが基準値から上昇したかチェック
                ankle_height_increased = False
                if self.baseline_ankle_height is not None and avg_ankle_height is not None:
                    # 基準値から8cm以上上昇したら離陸（5cm→8cmに増加、水平移動時の振動対策）
                    takeoff_threshold = 0.08
                    if avg_ankle_height > self.baseline_ankle_height + takeoff_threshold:
                        ankle_height_increased = True
                    # または過去5フレームで一貫して上昇傾向（3→5フレームに増加）
                    elif len(self.ankle_height_history) >= 5:
                        recent_heights = list(self.ankle_height_history)[-5:]
                        avg_recent = sum(recent_heights) / len(recent_heights)
                        # 全フレームが閾値を超えているかチェック（より厳密な判定）
                        all_above_threshold = all(h > self.baseline_ankle_height + takeoff_threshold for h in recent_heights)
                        if all_above_threshold and avg_recent > self.baseline_ankle_height + takeoff_threshold:
                            ankle_height_increased = True
                
                # 足首の高さが上昇した場合に離陸と判定
                # 注: 腰の高さ上昇は補助判定として使用しない（ジャンプ前のかがむ動作で誤検出の可能性があるため）
                if ankle_height_increased:
                    # 離陸時の全キーポイントの距離を記録
                    takeoff_distances = {}
                    if self.use_floor_detection and self.floor_detector:
                        for kp_name, kp_coords in keypoints_3d.items():
                            if kp_coords and kp_coords[0] is not None and kp_coords[1] is not None and kp_coords[2] is not None:
                                takeoff_distances[kp_name] = self.floor_detector.distance_to_floor(kp_coords)
                    self.takeoff_transitions.append((frame_num, takeoff_distances))

                    # 滞空フレームカウンターを初期化
                    self._airborne_frame_count = 0

                    self.jump_state = "takeoff"
                    self.jump_takeoff_frame = frame_num
                    self.jump_takeoff_timestamp = timestamp
                    self.jump_takeoff_position = (x, y, z)
                    # 離陸時の床からの高さを記録（ジャンプ高さ計算の基準）
                    if height_above_floor is not None:
                        self.jump_takeoff_height = height_above_floor
                        self.jump_max_height = height_above_floor
                    else:
                        self.jump_takeoff_height = z
                        self.jump_max_height = z
                    self.jump_max_distance = 0.0
                    result["state"] = "jump_start"
        
        elif self.jump_state == "takeoff":
            # 離陸直後: 浮上中 - 最大高さ・距離を更新
            if height_above_floor is not None:
                # 床からの高さで計算
                if height_above_floor > self.jump_max_height:
                    self.jump_max_height = height_above_floor
            else:
                # 床検出が使えない場合はZ座標で計算
                if z > self.jump_max_height:
                    self.jump_max_height = z

            # 水平距離を更新（開始位置から、XZ平面での距離）
            # RealSense座標系: X=左右, Y=上下（垂直）, Z=前後（深度）
            # 水平距離はXZ平面での距離であるべき
            if self.jump_start_position:
                jump_distance = np.sqrt(
                    (x - self.jump_start_position[0])**2 +
                    (z - self.jump_start_position[2])**2
                )
                if jump_distance > self.jump_max_distance:
                    self.jump_max_distance = jump_distance

            # 滞空フレーム数をカウント開始
            if both_feet_off_floor:
                self._airborne_frame_count += 1

            # 浮上中状態に移行
            self.jump_state = "airborne"
            result["state"] = "jumping"
            # 現在の高さ・距離を記録
            if height_above_floor is not None and self.jump_takeoff_height is not None:
                result["jump_height"] = height_above_floor - self.jump_takeoff_height
            else:
                result["jump_height"] = 0.0
            result["jump_distance"] = self.jump_max_distance
        
        elif self.jump_state == "airborne":
            # 浮上中: 最大高さ・距離を更新
            if height_above_floor is not None:
                # 床からの高さで計算
                if height_above_floor > self.jump_max_height:
                    self.jump_max_height = height_above_floor
            else:
                # 床検出が使えない場合はZ座標で計算
                if z > self.jump_max_height:
                    self.jump_max_height = z

            # 水平距離を更新（XZ平面での距離）
            # RealSense座標系: X=左右, Y=上下（垂直）, Z=前後（深度）
            if self.jump_start_position:
                jump_distance = np.sqrt(
                    (x - self.jump_start_position[0])**2 +
                    (z - self.jump_start_position[2])**2
                )
                if jump_distance > self.jump_max_distance:
                    self.jump_max_distance = jump_distance

            # 滞空フレーム数をカウント
            if both_feet_off_floor:
                self._airborne_frame_count += 1

            # 着地検出: 両足が床に着地
            if both_feet_stable_on_floor:
                # 最低滞空フレーム数をチェック（走行中の誤検出防止）
                if self._airborne_frame_count >= self._min_airborne_frames:
                    # 着地時の全キーポイントの距離を記録
                    landing_distances = {}
                    if self.use_floor_detection and self.floor_detector:
                        for kp_name, kp_coords in keypoints_3d.items():
                            if kp_coords and kp_coords[0] is not None and kp_coords[1] is not None and kp_coords[2] is not None:
                                landing_distances[kp_name] = self.floor_detector.distance_to_floor(kp_coords)
                    self.landing_transitions.append((frame_num, landing_distances))

                    self.jump_state = "landing"
                    self.jump_end_frame = frame_num
                    self.jump_end_timestamp = timestamp
                    self.jump_end_position = (x, y, z)
                    result["state"] = "jump_end"
                else:
                    # 滞空時間が短すぎる（走行中のステップ）→ジャンプとしてカウントしない
                    print(f"  Skipping short airborne: {self._airborne_frame_count} frames (min={self._min_airborne_frames})")
                    # 状態をリセット
                    self.jump_state = "ground"
                    self.jump_start_frame = None
                    self.jump_start_position = None
                    self.jump_takeoff_frame = None
                    self.jump_takeoff_position = None
                    self.jump_max_height = 0.0
                    self.jump_max_distance = 0.0
                    self._airborne_frame_count = 0
                    self._cooldown_frames = self._cooldown_duration // 2  # 短いクールダウン
                    result["state"] = "ground"
                    return result

            result["state"] = "jumping"
            # 現在の高さ・距離を記録（床からの高さ - 離陸時の高さ）
            if height_above_floor is not None and self.jump_takeoff_height is not None:
                result["jump_height"] = height_above_floor - self.jump_takeoff_height
            else:
                result["jump_height"] = 0.0
            result["jump_distance"] = self.jump_max_distance
        
        elif self.jump_state == "landing":
            # 着地: ジャンプ完了を記録
            # ジャンプ高さを計算（最大高さ - 離陸時の高さ）
            if self.jump_takeoff_height is not None:
                jump_height = self.jump_max_height - self.jump_takeoff_height
            elif self.jump_start_position:
                # フォールバック: 開始位置との差
                jump_height = self.jump_max_height - self.jump_start_position[2]
            else:
                jump_height = 0.0
            
            # 滞空時間を計算（秒）
            air_time = 0.0
            if self.jump_takeoff_timestamp is not None and self.jump_end_timestamp is not None:
                # タイムスタンプがミリ秒単位の可能性があるため、差分を計算
                timestamp_diff = abs(self.jump_end_timestamp - self.jump_takeoff_timestamp)
                # タイムスタンプ自体の値で単位を判定（ミリ秒なら10桁以上の値）
                if self.jump_takeoff_timestamp > 1000000000:  # タイムスタンプがミリ秒単位
                    air_time = timestamp_diff / 1000.0  # ミリ秒→秒
                else:
                    # タイムスタンプが秒単位の場合
                    air_time = timestamp_diff
                # 異常値をチェック（最大10秒の滞空時間を想定）
                if air_time > 10.0:
                    # 誤差が大きい可能性があるので、フレーム数から計算を試みる
                    if self.jump_takeoff_frame is not None and self.jump_end_frame is not None:
                        frame_diff = abs(self.jump_end_frame - self.jump_takeoff_frame)
                        # 30fpsを仮定
                        air_time = frame_diff / 30.0
            
            # ジャンプの種類を判定
            if self.jump_max_distance < self.threshold_horizontal:
                result["jump_type"] = "vertical"
                result["jump_height"] = jump_height
                result["jump_distance"] = 0.0
            else:
                result["jump_type"] = "horizontal"
                result["jump_height"] = jump_height
                result["jump_distance"] = self.jump_max_distance
            
            result["air_time"] = air_time
            
            # ジャンプとして有効かどうかを判定（最小高さ・滞空時間の閾値チェック）
            is_valid_jump = (
                jump_height >= self.min_jump_height and
                air_time >= self.min_air_time
            )
            
            if not is_valid_jump:
                # 無効なジャンプは記録しない（誤検出を除外）
                # ただし、クールダウンは適用して連続誤検出を防ぐ
                print(f"  Skipping invalid jump: height={jump_height*100:.1f}cm (min={self.min_jump_height*100:.1f}cm), "
                      f"air_time={air_time*1000:.1f}ms (min={self.min_air_time*1000:.1f}ms)")
            else:
                # 検出されたジャンプを記録
                jump_data = {
                    "frame_start": self.jump_start_frame,
                    "frame_takeoff": self.jump_takeoff_frame,
                    "frame_end": self.jump_end_frame,
                    "timestamp_start": self.jump_start_timestamp,
                    "timestamp_takeoff": self.jump_takeoff_timestamp,
                    "timestamp_end": self.jump_end_timestamp,
                    "jump_type": result["jump_type"],
                    "height": result["jump_height"],
                    "distance": result["jump_distance"],
                    "air_time": air_time,
                    "max_height": self.jump_max_height,
                    "start_position": self.jump_start_position,
                    "takeoff_position": self.jump_takeoff_position,
                    "end_position": self.jump_end_position
                }
                self.detected_jumps.append(jump_data)
            
            # 状態をリセット
            self.jump_state = "ground"
            self.jump_start_frame = None
            self.jump_start_timestamp = None
            self.jump_start_position = None
            self.jump_takeoff_frame = None
            self.jump_takeoff_timestamp = None
            self.jump_takeoff_position = None
            self.jump_end_frame = None
            self.jump_end_timestamp = None
            self.jump_end_position = None
            # ジャンプ検出関連の変数をリセット
            self.jump_max_height = 0.0
            self.jump_max_distance = 0.0
            self.jump_takeoff_height = None
            self._initial_state = True  # 次のジャンプ検出のために初期状態に戻す

            # クールダウンを開始（着地後しばらくは新しいジャンプを検出しない）
            self._cooldown_frames = self._cooldown_duration
            # 滞空フレームカウンターをリセット
            self._airborne_frame_count = 0

            # 着地後は一定フレーム間、新しいジャンプを検出しない（誤検出を防ぐ）
            # 履歴をクリアして、安定した床接触を再確立
            self.left_foot_on_floor.clear()
            self.right_foot_on_floor.clear()
            # 足首高さの基準値もリセット（次のジャンプのために再計算）
            self.baseline_ankle_height = None
            self.ankle_height_history.clear()
        
        return result

    def get_statistics(self):
        """検出されたジャンプの統計情報を取得"""
        if not self.detected_jumps:
            return {
                "total_jumps": 0,
                "vertical_jumps": 0,
                "horizontal_jumps": 0,
                "max_height": 0.0,
                "max_distance": 0.0,
                "avg_height": 0.0,
                "avg_distance": 0.0,
                "max_air_time": 0.0,
                "avg_air_time": 0.0
            }

        vertical_jumps = [j for j in self.detected_jumps if j["jump_type"] == "vertical"]
        horizontal_jumps = [j for j in self.detected_jumps if j["jump_type"] == "horizontal"]

        all_heights = [j["height"] for j in self.detected_jumps]
        # 距離は全ジャンプから取得（vertical jumpも距離0.0として含める）
        all_distances = [j["distance"] for j in self.detected_jumps]
        # 異常な滞空時間を除外（10秒以上は異常値として扱う）
        all_air_times = [j.get("air_time", 0.0) for j in self.detected_jumps 
                        if "air_time" in j and j.get("air_time", 0.0) < 10.0]

        return {
            "total_jumps": len(self.detected_jumps),
            "vertical_jumps": len(vertical_jumps),
            "horizontal_jumps": len(horizontal_jumps),
            "max_height": max(all_heights) if all_heights else 0.0,
            "max_distance": max(all_distances) if all_distances else 0.0,
            "avg_height": np.mean(all_heights) if all_heights else 0.0,
            "avg_distance": np.mean(all_distances) if all_distances and len(all_distances) > 0 else 0.0,
            "max_air_time": max(all_air_times) if all_air_times else 0.0,
            "avg_air_time": np.mean(all_air_times) if all_air_times else 0.0,
            "jumps": self.detected_jumps
        }
    
    def analyze_keypoint_variability(self):
        """
        全キーポイントの床からの距離の変動性を分析
        歩行時は上下変化が少なく、ジャンプ時は変位が顕著に現れるキーポイントを特定
        
        Returns:
            dict: 各キーポイントの変動性統計情報
                {
                    'keypoint_name': {
                        'mean': 平均距離,
                        'std': 標準偏差,
                        'min': 最小距離,
                        'max': 最大距離,
                        'range': 最大変位（max - min）,
                        'cv': 変動係数（std/mean）,
                        'valid_samples': 有効サンプル数,
                        'walking_mean': 歩行時平均（ground状態のみ）,
                        'walking_std': 歩行時標準偏差,
                        'jumping_mean': ジャンプ時平均（jumping状態のみ）,
                        'jumping_std': ジャンプ時標準偏差,
                        'jump_walk_ratio': ジャンプ時/歩行時の変動比,
                        'jump_sensitivity': ジャンプ検出感度スコア
                    },
                    ...
                }
        """
        variability_stats = {}
        
        for kp_name, distance_history in self.keypoint_distance_history.items():
            # 履歴から距離と状態を分離
            all_distances = []
            walking_distances = []
            jumping_distances = []
            
            for item in distance_history:
                if isinstance(item, tuple):
                    distance, state = item
                else:
                    # 後方互換性のため（古いデータ形式）
                    distance = item
                    state = 'ground'
                
                if distance is not None:
                    all_distances.append(distance)
                    if state == 'ground' or state == 'initializing':
                        walking_distances.append(distance)
                    elif state in ['takeoff', 'airborne', 'jumping', 'landing']:
                        jumping_distances.append(distance)
            
            if len(all_distances) < 2:
                continue
            
            all_distances = np.array(all_distances)
            
            # 全体の統計量を計算
            mean_dist = np.mean(all_distances)
            std_dist = np.std(all_distances)
            min_dist = np.min(all_distances)
            max_dist = np.max(all_distances)
            range_dist = max_dist - min_dist
            cv = std_dist / mean_dist if mean_dist > 0 else 0.0
            
            # 歩行時とジャンプ時の統計を別々に計算
            walking_mean = np.mean(walking_distances) if len(walking_distances) > 0 else None
            walking_std = np.std(walking_distances) if len(walking_distances) > 1 else None
            
            jumping_mean = np.mean(jumping_distances) if len(jumping_distances) > 0 else None
            jumping_std = np.std(jumping_distances) if len(jumping_distances) > 1 else None
            
            # ジャンプ時と歩行時の変動比を計算
            jump_walk_ratio = None
            jump_sensitivity = 0.0
            
            if walking_std is not None and walking_std > 0 and jumping_std is not None:
                # 標準偏差の比（ジャンプ時の変動が歩行時よりどれだけ大きいか）
                jump_walk_ratio = jumping_std / walking_std
                
                # ジャンプ検出感度スコア: ジャンプ時の変動が大きく、歩行時の変動が小さいほど高い
                # 式: (jumping_std - walking_std) / (jumping_std + walking_std + epsilon)
                # これは-1から1の範囲で、1に近いほどジャンプ検出に適している
                epsilon = 1e-6
                if jumping_std + walking_std > 0:
                    jump_sensitivity = (jumping_std - walking_std) / (jumping_std + walking_std + epsilon)
            
            variability_stats[kp_name] = {
                'mean': float(mean_dist),
                'std': float(std_dist),
                'min': float(min_dist),
                'max': float(max_dist),
                'range': float(range_dist),
                'cv': float(cv),
                'valid_samples': len(all_distances),
                'walking_samples': len(walking_distances),
                'jumping_samples': len(jumping_distances),
                'walking_mean': float(walking_mean) if walking_mean is not None else None,
                'walking_std': float(walking_std) if walking_std is not None else None,
                'jumping_mean': float(jumping_mean) if jumping_mean is not None else None,
                'jumping_std': float(jumping_std) if jumping_std is not None else None,
                'jump_walk_ratio': float(jump_walk_ratio) if jump_walk_ratio is not None else None,
                'jump_sensitivity': float(jump_sensitivity)
            }
        
        return variability_stats
    
    def get_most_jump_sensitive_keypoints(self, min_samples=10, top_n=5):
        """
        ジャンプ検出に最も敏感なキーポイントを特定
        変動係数（CV）と最大変位（range）の両方を考慮
        
        Args:
            min_samples: 最小サンプル数（この数を下回るキーポイントは除外）
            top_n: 返すキーポイントの数
        
        Returns:
            list: タプルのリスト [(keypoint_name, stats_dict), ...]
                  変動係数と最大変位の組み合わせスコアでソート
        """
        stats = self.analyze_keypoint_variability()
        
        # フィルタリング（最小サンプル数以上のもののみ）
        filtered_stats = {
            kp_name: stats_dict
            for kp_name, stats_dict in stats.items()
            if stats_dict['valid_samples'] >= min_samples
        }
        
        if len(filtered_stats) == 0:
            return []
        
        # スコアリング: ジャンプ検出感度（jump_sensitivity）を優先的に使用
        # jump_sensitivityが高いほど、歩行時との違いが明確（ジャンプ検出に適している）
        scored_keypoints = []
        for kp_name, stats_dict in filtered_stats.items():
            # ジャンプ検出感度スコアを主要指標として使用
            # 補助的にCVとrangeも考慮
            jump_sensitivity_score = (stats_dict['jump_sensitivity'] + 1.0) / 2.0  # -1~1を0~1に正規化
            
            # 変動係数と最大変位も考慮（補助スコア）
            cv_score = min(stats_dict['cv'], 2.0) / 2.0 if stats_dict['cv'] > 0 else 0.0
            range_score = min(stats_dict['range'] / 0.5, 1.0)
            
            # ジャンプ検出感度を優先しつつ、全体の変動も考慮
            # jump_sensitivityが利用可能な場合はそれを重視
            if stats_dict['jump_sensitivity'] is not None and stats_dict['jumping_samples'] > 0:
                # ジャンプ検出感度70%、変動係数20%、最大変位10%
                combined_score = jump_sensitivity_score * 0.7 + cv_score * 0.2 + range_score * 0.1
            else:
                # ジャンプ状態のデータがない場合は従来の方法で評価
                combined_score = (cv_score * 0.5 + range_score * 0.5)
            
            scored_keypoints.append((kp_name, stats_dict, combined_score))
        
        # スコアでソート（降順）
        scored_keypoints.sort(key=lambda x: x[2], reverse=True)
        
        # top_n個を返す
        return [(kp_name, stats) for kp_name, stats, score in scored_keypoints[:top_n]]

    def get_trajectory(self):
        """軌跡データを取得"""
        return self.trajectory
    
    def analyze_jump_transitions(self):
        """
        ジャンプの始まり（離陸）と終わり（着地）を捉えるためのキーポイントを特定
        離陸時と着地時の急激な変化を示すキーポイントを分析
        
        Returns:
            dict: {
                'takeoff_keypoints': [(keypoint_name, stats_dict), ...],
                'landing_keypoints': [(keypoint_name, stats_dict), ...]
            }
        """
        takeoff_analysis = {}
        landing_analysis = {}
        
        # 離陸時の変化を分析
        if len(self.takeoff_transitions) > 0:
            # 各キーポイントについて、離陸前後の変化を計算
            for kp_name in self.keypoint_distance_history.keys():
                changes = []
                
                for i, (takeoff_frame, takeoff_dists) in enumerate(self.takeoff_transitions):
                    if kp_name not in takeoff_dists or takeoff_dists[kp_name] is None:
                        continue
                    
                    # 離陸直前（3フレーム前）と離陸時の距離を取得
                    # 履歴から該当フレーム付近の値を探す
                    history = list(self.keypoint_distance_history[kp_name])
                    
                    # 離陸時の値を取得
                    takeoff_value = takeoff_dists[kp_name]
                    
                    # 離陸直前の値を取得（履歴から遡って探す）
                    # 履歴は最近の値が最後にあるので、逆順で探す
                    before_value = None
                    for j in range(len(history) - 1, max(0, len(history) - 10), -1):
                        item = history[j]
                        if isinstance(item, tuple):
                            dist, state = item
                        else:
                            dist = item
                            state = 'ground'
                        
                        if dist is not None and state in ['ground', 'initializing']:
                            before_value = dist
                            break
                    
                    if before_value is not None and takeoff_value is not None:
                        # 変化量を計算（離陸時に上昇するので正の値）
                        change = takeoff_value - before_value
                        changes.append(change)
                
                if len(changes) > 0:
                    changes_array = np.array(changes)
                    avg_change = np.mean(changes_array)
                    std_change = np.std(changes_array)
                    # スコア: 平均変化量の絶対値と一貫性（1/std）を考慮
                    score = abs(avg_change) * (1.0 / (std_change + 0.01))  # 一貫性の高い変化ほど高スコア
                    takeoff_analysis[kp_name] = {
                        'avg_change': float(avg_change),
                        'std_change': float(std_change),
                        'score': float(score),
                        'samples': len(changes)
                    }
        
        # 着地時の変化を分析
        if len(self.landing_transitions) > 0:
            for kp_name in self.keypoint_distance_history.keys():
                changes = []
                
                for i, (landing_frame, landing_dists) in enumerate(self.landing_transitions):
                    if kp_name not in landing_dists or landing_dists[kp_name] is None:
                        continue
                    
                    landing_value = landing_dists[kp_name]
                    
                    # 着地直前（空中の状態）の値を取得
                    history = list(self.keypoint_distance_history[kp_name])
                    before_value = None
                    for j in range(len(history) - 1, max(0, len(history) - 10), -1):
                        item = history[j]
                        if isinstance(item, tuple):
                            dist, state = item
                        else:
                            dist = item
                            state = 'ground'
                        
                        if dist is not None and state in ['airborne', 'jumping', 'takeoff']:
                            before_value = dist
                            break
                    
                    if before_value is not None and landing_value is not None:
                        # 変化量を計算（着地時に下降するので負の値）
                        change = landing_value - before_value
                        changes.append(change)
                
                if len(changes) > 0:
                    changes_array = np.array(changes)
                    avg_change = np.mean(changes_array)
                    std_change = np.std(changes_array)
                    # スコア: 平均変化量の絶対値と一貫性を考慮
                    score = abs(avg_change) * (1.0 / (std_change + 0.01))
                    landing_analysis[kp_name] = {
                        'avg_change': float(avg_change),
                        'std_change': float(std_change),
                        'score': float(score),
                        'samples': len(changes)
                    }
        
        # スコアでソート
        takeoff_sorted = sorted(takeoff_analysis.items(), key=lambda x: x[1]['score'], reverse=True)
        landing_sorted = sorted(landing_analysis.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return {
            'takeoff_keypoints': [(name, stats) for name, stats in takeoff_sorted],
            'landing_keypoints': [(name, stats) for name, stats in landing_sorted]
        }

