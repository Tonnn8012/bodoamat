"""
scripts/mission_control.py
===========================
Node ROS2 utama untuk kendali misi autonomous VTOL KRTI 2026.

Peningkatan Desain & Perbaikan Bug:
1. Satu Kamera + Servo: Menggunakan 1 camera capture (index 0 / topic `/camera/image_raw`)
   dengan servo tilting. Servo dikontrol lewat topic `/vtol/camera_servo_pitch`.
2. Transition Delay: Mengabaikan input frame selama 1.5 detik setelah merubah sudut servo
   untuk membiarkan kamera selesai berputar dan frame stabil (tidak ada motion blur/guncangan).
3. Mengatasi Inersia (Overshoot): Menerapkan filter settling time (2.0 detik stabil)
   sebelum melakukan dropping payload atau precision landing.
"""

import sys
import os
import time
import math
# pyrefly: ignore [missing-import]
import rclpy
# pyrefly: ignore [missing-import]
from rclpy.node import Node
# pyrefly: ignore [missing-import]
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Import ROS2 message types
# pyrefly: ignore [missing-import]
from geometry_msgs.msg import Twist, PoseStamped
# pyrefly: ignore [missing-import]
from sensor_msgs.msg import Image
# pyrefly: ignore [missing-import]
from std_msgs.msg import Float64
# pyrefly: ignore [missing-import]
from mavros_msgs.msg import State
# pyrefly: ignore [missing-import]
from mavros_msgs.srv import CommandBool, SetMode

# Import OpenCV bridge
try:
    # pyrefly: ignore [missing-import]
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False

import cv2
import numpy as np

# Tambahkan path root agar bisa import modul
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vtol_detector import VTOLDetector
from scripts.pid_controller import PIDController
from scripts.line_follower import LineFollower


class MissionState:
    WAITING_FOR_FCU = 0
    SET_MODE_GUIDED = 1
    ARMING = 2
    TAKEOFF = 3
    
    # Mission 2: Double Gate
    GOTO_GATE_1 = 4
    CENTERING_GATE_1 = 5
    PASSING_GATE_1 = 6
    GOTO_GATE_2 = 7
    CENTERING_GATE_2 = 8
    PASSING_GATE_2 = 9
    
    # Mission 2: Medical Drop
    GOTO_DROP_ZONE = 10
    CENTERING_RED_BOX = 11
    DESCENDING_FOR_DROP = 12
    RELEASING_PAYLOAD = 13
    ASCENDING_AFTER_DROP = 14
    
    # Mission 3: Triple Gate Corridor
    TRIPLE_GATE_NAVIGATION = 15
    
    # Mission 4: Line Following
    LINE_FOLLOWING = 16
    
    # Mission 5: Final Gate & Landing
    GOTO_FINAL_GATE = 17
    CENTERING_FINAL_GATE = 18
    PASSING_FINAL_GATE = 19
    SEARCHING_LANDING_PAD = 20
    CENTERING_LANDING_PAD = 21
    DESCENDING_FOR_LAND = 22
    LANDED = 23


class VTOLMissionControlNode(Node):
    def __init__(self) -> None:
        super().__init__("vtol_mission_control")
        self.get_logger().info("Menginisialisasi Node Kendali Misi VTOL KRTI 2026 (1 Kamera Servo)...")

        # ------------------------------------------------------------------
        # PARAMETER CONFIGURATION
        # ------------------------------------------------------------------
        # Menggunakan model ONNX karena Orange Pi 4 tidak punya NPU
        self.declare_parameter("model_path", "models/best.onnx")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("use_onnx", True)
        self.declare_parameter("camera_index", 0)  # Hanya ada 1 kamera fisik
        self.declare_parameter("use_direct_camera", True)  # Membaca langsung via OpenCV (Low Latency)
        
        self.model_path = self.get_parameter("model_path").value
        self.conf_threshold = self.get_parameter("conf_threshold").value
        self.use_onnx = self.get_parameter("use_onnx").value
        self.camera_idx = self.get_parameter("camera_index").value
        self.use_direct_cam = self.get_parameter("use_direct_camera").value

        # ------------------------------------------------------------------
        # MODULE INITIALIZATION
        # ------------------------------------------------------------------
        # 1. Detector YOLOv8 ONNX
        self.detector = VTOLDetector(
            model_path=self.model_path,
            conf_threshold=self.conf_threshold,
            use_onnx=self.use_onnx
        )
        
        # 2. Line Follower OpenCV (Kamera Bawah)
        self.line_follower = LineFollower(default_threshold=60, min_area=300.0)

        # 3. PID Controllers (Slew rate limiter & Deadzone sangat krusial redam overshoot)
        # Roll: Geser kanan/kiri (lateral offset)
        self.pid_roll = PIDController(
            kp=0.0012, ki=0.00005, kd=0.0006,  # Diturunkan sedikit Kp dan dinaikkan Kd untuk meredam inersia
            min_output=-0.35, max_output=0.35,
            max_integral=0.08, slew_rate_limit=0.4,
            deadzone=8.0
        )
        # Pitch: Maju/mundur (longitudinal offset)
        self.pid_pitch = PIDController(
            kp=0.0012, ki=0.00005, kd=0.0006,
            min_output=-0.3, max_output=0.3,
            max_integral=0.08, slew_rate_limit=0.4,
            deadzone=8.0
        )
        # Yaw: Rotasi orientasi drone
        self.pid_yaw = PIDController(
            kp=0.012, ki=0.0002, kd=0.004,
            min_output=-0.25, max_output=0.25,
            max_integral=0.08, slew_rate_limit=0.3,
            deadzone=2.0
        )
        # Alt: Vertical rate rate naik/turun
        self.pid_alt = PIDController(
            kp=0.35, ki=0.01, kd=0.08,
            min_output=-0.25, max_output=0.25,
            max_integral=0.08, slew_rate_limit=0.3,
            deadzone=0.05
        )

        # ------------------------------------------------------------------
        # STATE & TELEMETRY
        # ------------------------------------------------------------------
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.mission_state = MissionState.WAITING_FOR_FCU
        self.target_altitude = 1.0  # Ketinggian default jelajah
        self.state_timer = 0.0
        self.bridge = CvBridge() if HAS_CV_BRIDGE else None

        # Aktuator servo drop kit & servo kamera
        self.servo_triggered = False
        self.target_camera_angle = 0.0      # 0.0 = Depan, 90.0 = Bawah
        self.current_camera_angle = 0.0
        self.last_camera_tilt_time = 0.0    # Catat waktu saat servo digerakkan

        # Timer untuk stabilisasi centering (mengurangi overshoot inersia)
        self.centering_start_time = None

        # ------------------------------------------------------------------
        # ROS2 PUBLISHERS & SUBSCRIBERS
        # ------------------------------------------------------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state_sub = self.create_subscription(
            State, "/mavros/state", self.state_cb, qos_profile
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_cb, qos_profile
        )
        
        self.vel_pub = self.create_publisher(
            Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10
        )
        self.pos_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", 10
        )

        # Publisher untuk servo gimbal/tilting kamera (0.0 = depan, 90.0 = bawah)
        self.camera_servo_pub = self.create_publisher(
            Float64, "/vtol/camera_servo_pitch", 10
        )

        # ROS2 Service Clients
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # ------------------------------------------------------------------
        # CAMERA INITIALIZATION
        # ------------------------------------------------------------------
        if self.use_direct_cam:
            self.get_logger().info(f"Membuka kamera index={self.camera_idx} secara langsung...")
            self.cap = cv2.VideoCapture(self.camera_idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.control_timer = self.create_timer(1.0 / 10.0, self.main_control_loop)
        else:
            self.image_sub = self.create_subscription(
                Image, "/camera/image_raw", self.image_cb, 10
            )
            self.latest_frame = None
            self.control_timer = self.create_timer(1.0 / 30.0, self.main_control_loop_ros_img)

        # Status logger periodik
        self.status_timer = self.create_timer(2.0, self.log_mission_status)

        # Posisi servo default: Menghadap depan saat mulai
        self.set_camera_servo(0.0)

    # ------------------------------------------------------------------
    # CALLBACKS & CAMERA HELPERS
    # ------------------------------------------------------------------
    def state_cb(self, msg: State) -> None:
        self.current_state = msg

    def pose_cb(self, msg: PoseStamped) -> None:
        self.current_pose = msg

    def image_cb(self, msg: Image) -> None:
        if self.bridge:
            try:
                self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"Gagal mengonversi gambar ROS: {e}")

    def set_camera_servo(self, angle: float) -> None:
        """Mengirim perintah perubahan sudut servo kamera."""
        self.target_camera_angle = angle
        msg = Float64()
        msg.data = angle
        self.camera_servo_pub.publish(msg)
        
        if abs(self.current_camera_angle - angle) > 1.0:
            self.get_logger().info(f"Merubah posisi servo kamera ke: {angle} derajat.")
            self.current_camera_angle = angle
            self.last_camera_tilt_time = time.time()  # Catat waktu transisi fisik servo

    def main_control_loop(self) -> None:
        if not self.use_direct_cam:
            return
        ret, frame = self.cap.read() if self.cap.isOpened() else (False, None)
        self.process_mission_logic(frame)

    def main_control_loop_ros_img(self) -> None:
        if self.use_direct_cam:
            return
        self.process_mission_logic(self.latest_frame)

    # ------------------------------------------------------------------
    # FSM CORE
    # ------------------------------------------------------------------
    def process_mission_logic(self, frame: np.ndarray | None) -> None:
        # 1. Penanganan Koneksi & Arming awal
        if self.mission_state == MissionState.WAITING_FOR_FCU:
            if self.current_state.connected:
                self.get_logger().info("FCU Pixhawk terhubung! Set mode GUIDED...")
                self.mission_state = MissionState.SET_MODE_GUIDED
                self.state_timer = time.time()
            return

        elif self.mission_state == MissionState.SET_MODE_GUIDED:
            if self.current_state.mode != "GUIDED":
                if time.time() - self.state_timer > 2.0:
                    self.set_mode("GUIDED")
                    self.state_timer = time.time()
            else:
                self.get_logger().info("Mode GUIDED aktif! Meng-arm drone...")
                self.mission_state = MissionState.ARMING
                self.state_timer = time.time()
            return

        elif self.mission_state == MissionState.ARMING:
            if not self.current_state.armed:
                if time.time() - self.state_timer > 2.0:
                    self.arm_drone(True)
                    self.state_timer = time.time()
            else:
                self.get_logger().info("Drone Armed! Melakukan auto-takeoff...")
                self.mission_state = MissionState.TAKEOFF
                self.state_timer = time.time()
                self.send_takeoff_cmd()
            return

        elif self.mission_state == MissionState.TAKEOFF:
            curr_z = self.current_pose.pose.position.z
            if abs(curr_z - self.target_altitude) < 0.15:
                self.get_logger().info(f"Drone stabil di ketinggian {curr_z:.2f}m. Memulai Misi 2 (Double Gate)...")
                self.mission_state = MissionState.GOTO_GATE_1
                self.state_timer = time.time()
            else:
                self.publish_position_target(0.0, 0.0, self.target_altitude)
            return

        # ------------------------------------------------------------------
        # TILT TRANSITION COOLDOWN
        # ------------------------------------------------------------------
        # Jika servo kamera baru saja digerakkan (kurang dari 1.5 detik),
        # abaikan frame untuk mencegah getaran visual / distorsi mengacaukan kontrol.
        time_since_tilt = time.time() - self.last_camera_tilt_time
        if time_since_tilt < 1.5:
            # Hover stabil selama transisi kamera
            hover_vel = Twist()
            hover_vel.linear.x = 0.0
            hover_vel.linear.y = 0.0
            curr_z = self.current_pose.pose.position.z
            hover_vel.linear.z = self.pid_alt.update(self.target_altitude - curr_z)
            self.vel_pub.publish(hover_vel)
            
            # Tampilkan informasi di GUI jika frame valid
            if frame is not None:
                cv2.putText(
                    frame, "CAMERA TILTING...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                )
                cv2.imshow("Camera View", frame)
                cv2.waitKey(1)
            return

        # ------------------------------------------------------------------
        # VELOCITY COMMAND & DETEKSI OBJECT
        # ------------------------------------------------------------------
        cmd_vel = Twist()
        curr_z = self.current_pose.pose.position.z
        cmd_vel.linear.z = self.pid_alt.update(self.target_altitude - curr_z)

        # Proses deteksi YOLO (jika bukan state Line Following)
        detections = []
        stable_det = None
        
        if self.mission_state != MissionState.LINE_FOLLOWING and frame is not None:
            stable_det = self.detector.detect_stable(frame)
            if stable_det is not None:
                detections = [stable_det]
                annotated = self.detector.annotate_frame(frame, [stable_det])
                cv2.imshow("Camera View", annotated)
                cv2.waitKey(1)
            else:
                cv2.imshow("Camera View", frame)
                cv2.waitKey(1)

        # Cari target spesifik sesuai state aktif
        target_class = "gate"
        if self.mission_state in [MissionState.CENTERING_RED_BOX, MissionState.GOTO_DROP_ZONE]:
            target_class = "red_box"
        elif self.mission_state in [MissionState.SEARCHING_LANDING_PAD, MissionState.CENTERING_LANDING_PAD]:
            target_class = "landing_pad"

        target_detection = None
        for det in detections:
            if det["class"] == target_class:
                target_detection = det
                break

        # ------------------------------------------------------------------
        # BRANCH STATE MACHINE UTAMA
        # ------------------------------------------------------------------

        # --- MISI 2: DOUBLE GATE ---
        if self.mission_state == MissionState.GOTO_GATE_1:
            self.set_camera_servo(0.0)  # Kamera menghadap depan
            cmd_vel.linear.x = 0.22     # Maju mencari gate 1
            if target_detection is not None:
                self.get_logger().info("Gate 1 terdeteksi! Memulai centering...")
                self.mission_state = MissionState.CENTERING_GATE_1
                self.pid_roll.reset()
                self.pid_yaw.reset()

        elif self.mission_state == MissionState.CENTERING_GATE_1:
            if target_detection is not None:
                h, w = frame.shape[:2]
                err_x, err_y = self.detector.get_centering_error(target_detection, w, h, self.current_camera_angle)
                
                # Gunakan roll untuk geser lateral, yaw untuk penyesuaian sudut hadap
                cmd_vel.linear.y = -self.pid_roll.update(err_x)
                cmd_vel.angular.z = -self.pid_yaw.update(err_x)
                cmd_vel.linear.x = 0.12  # Mendekat perlahan
                
                # Siap tembus jika error kecil dan gate sudah sangat dekat (width gate > 45% frame)
                if abs(err_x) < 20.0 and target_detection["width"] > (w * 0.45):
                    self.get_logger().info("Gate 1 terpusat! Melakukan penetrasi...")
                    self.mission_state = MissionState.PASSING_GATE_1
                    self.state_timer = time.time()
            else:
                # Target hilang sesaat, kurangi kecepatan/hover
                cmd_vel.linear.x = 0.05
                cmd_vel.linear.y = 0.0

        elif self.mission_state == MissionState.PASSING_GATE_1:
            # Lewati gate dengan kecepatan buta lurus ke depan
            cmd_vel.linear.x = 0.38
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.0
            if time.time() - self.state_timer > 3.5:
                self.get_logger().info("Gate 1 terlewati! Mencari Gate 2...")
                self.mission_state = MissionState.GOTO_GATE_2
                self.state_timer = time.time()

        elif self.mission_state == MissionState.GOTO_GATE_2:
            cmd_vel.linear.x = 0.22
            if target_detection is not None:
                self.get_logger().info("Gate 2 terdeteksi! Memulai centering...")
                self.mission_state = MissionState.CENTERING_GATE_2
                self.pid_roll.reset()
                self.pid_yaw.reset()

        elif self.mission_state == MissionState.CENTERING_GATE_2:
            if target_detection is not None:
                h, w = frame.shape[:2]
                err_x, err_y = self.detector.get_centering_error(target_detection, w, h, self.current_camera_angle)
                cmd_vel.linear.y = -self.pid_roll.update(err_x)
                cmd_vel.angular.z = -self.pid_yaw.update(err_x)
                cmd_vel.linear.x = 0.12
                
                if abs(err_x) < 20.0 and target_detection["width"] > (w * 0.45):
                    self.get_logger().info("Gate 2 terpusat! Melakukan penetrasi...")
                    self.mission_state = MissionState.PASSING_GATE_2
                    self.state_timer = time.time()
            else:
                cmd_vel.linear.x = 0.05
                cmd_vel.linear.y = 0.0

        elif self.mission_state == MissionState.PASSING_GATE_2:
            cmd_vel.linear.x = 0.38
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.0
            if time.time() - self.state_timer > 3.5:
                self.get_logger().info("Gate 2 terlewati! Mengarahkan kamera kebawah untuk mencari Drop Zone...")
                self.mission_state = MissionState.GOTO_DROP_ZONE
                self.set_camera_servo(90.0)  # Kamera diputar ke bawah
                self.state_timer = time.time()

        # --- MISI 2: MEDICAL DELIVERY (RED BOX) ---
        elif self.mission_state == MissionState.GOTO_DROP_ZONE:
            # Maju perlahan sambil memindai lantai (kamera bawah) mencari Red Box
            cmd_vel.linear.x = 0.18
            if target_detection is not None:
                self.get_logger().info("Red Box terdeteksi di bawah! Memulai centering...")
                self.mission_state = MissionState.CENTERING_RED_BOX
                self.pid_roll.reset()
                self.pid_pitch.reset()
                self.centering_start_time = None  # Reset timer stabilisasi

        elif self.mission_state == MissionState.CENTERING_RED_BOX:
            if target_detection is not None:
                h, w = frame.shape[:2]
                err_x, err_y = self.detector.get_centering_error(target_detection, w, h, self.current_camera_angle)
                
                # Roll mengoreksi lateral X (kiri/kanan), Pitch mengoreksi longitudinal Y (maju/mundur)
                cmd_vel.linear.y = -self.pid_roll.update(err_x)
                cmd_vel.linear.x = -self.pid_pitch.update(err_y)
                
                # Logika filter inersia: error harus kecil secara terus menerus selama 2.0 detik
                # untuk memastikan drone benar-benar diam (hover stabil) di atas box.
                if abs(err_x) < 15.0 and abs(err_y) < 15.0:
                    if self.centering_start_time is None:
                        self.centering_start_time = time.time()
                        self.get_logger().info("Target centered. Memulai kalkulasi stabilitas inersia...")
                    elif time.time() - self.centering_start_time > 2.0:
                        self.get_logger().info("Drone stabil! Melakukan descent untuk drop payload...")
                        self.mission_state = MissionState.DESCENDING_FOR_DROP
                        self.target_altitude = 0.5  # Turun ke ketinggian 50 cm
                        self.state_timer = time.time()
                else:
                    if self.centering_start_time is not None:
                        self.get_logger().warn("Drone tergeser oleh inersia/angin, mereset filter stabil.")
                        self.centering_start_time = None
            else:
                # Red box hilang dari pandangan, pertahankan posisi (hover)
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                self.centering_start_time = None

        elif self.mission_state == MissionState.DESCENDING_FOR_DROP:
            # Jaga kestabilan posisi X-Y di atas box saat proses turun
            if target_detection is not None:
                h, w = frame.shape[:2]
                err_x, err_y = self.detector.get_centering_error(target_detection, w, h, self.current_camera_angle)
                cmd_vel.linear.y = -self.pid_roll.update(err_x)
                cmd_vel.linear.x = -self.pid_pitch.update(err_y)
            
            # Jika sudah di target ketinggian drop
            if abs(curr_z - 0.5) < 0.12:
                self.get_logger().info("Ketinggian drop tercapai. Mengaktifkan drop kit servo...")
                self.mission_state = MissionState.RELEASING_PAYLOAD
                self.state_timer = time.time()

        elif self.mission_state == MissionState.RELEASING_PAYLOAD:
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.0
            
            if not self.servo_triggered:
                self.trigger_servo()
                self.servo_triggered = True
                
            # Tunggu aksi mekanis servo drop 2.0 detik
            if time.time() - self.state_timer > 2.0:
                self.get_logger().info("Payload dilepas. Naik kembali ke ketinggian cruise...")
                self.mission_state = MissionState.ASCENDING_AFTER_DROP
                self.target_altitude = 1.0
                self.state_timer = time.time()

        elif self.mission_state == MissionState.ASCENDING_AFTER_DROP:
            if abs(curr_z - 1.0) < 0.15:
                self.get_logger().info("Ketinggian cruise tercapai. Mengarahkan kamera ke depan...")
                self.mission_state = MissionState.TRIPLE_GATE_NAVIGATION
                self.set_camera_servo(0.0)  # Kamera diputar kembali menghadap depan
                self.state_timer = time.time()

        # --- MISI 3: TRIPLE GATE CORRIDOR ---
        elif self.mission_state == MissionState.TRIPLE_GATE_NAVIGATION:
            cmd_vel.linear.x = 0.18
            
            # Lock yaw ke sudut kompas awal (0.0 rad) agar drone melaju lurus di koridor sempit
            yaw_err = 0.0 - self.get_current_yaw()
            cmd_vel.angular.z = self.pid_yaw.update(yaw_err)
            
            # Jika terdeteksi gate, gunakan roll untuk centering horizontal minor
            if target_detection is not None:
                h, w = frame.shape[:2]
                err_x, err_y = self.detector.get_centering_error(target_detection, w, h, self.current_camera_angle)
                cmd_vel.linear.y = -self.pid_roll.update(err_x)
            
            # Navigasi koridor berlangsung ~20 detik
            if time.time() - self.state_timer > 20.0:
                self.get_logger().info("Triple Gate terlewati! Memutar kamera ke bawah untuk follow line...")
                self.mission_state = MissionState.LINE_FOLLOWING
                self.target_altitude = 1.15  # Ketinggian optimal agar line follower mendapat FOV ideal
                self.set_camera_servo(90.0)  # Kamera ke bawah
                self.state_timer = time.time()

        # --- MISI 4: LINE FOLLOWING ---
        elif self.mission_state == MissionState.LINE_FOLLOWING:
            if frame is not None:
                # Jalankan OpenCV Line follower (menggunakan frame tunggal yang menghadap ke bawah)
                cte, heading_err, mask, annotated_bottom = self.line_follower.process_frame(frame)
                
                cv2.imshow("Camera View (Line Following)", annotated_bottom)
                cv2.waitKey(1)
                
                if cte is not None and heading_err is not None:
                    # Koreksi lateral (roll) berdasarkan CTE
                    cmd_vel.linear.y = self.pid_roll.update(cte)
                    
                    # Koreksi yaw berdasarkan sudut miring lintasan garis
                    cmd_vel.angular.z = self.pid_yaw.update(heading_err)
                    
                    # Kecepatan maju adaptif (melambat jika belok tajam agar tidak out of track)
                    cmd_vel.linear.x = max(0.08, 0.26 * (1.0 - abs(heading_err) / 90.0))
                else:
                    # Kehilangan garis, melayang (hover) di tempat mencari lintasan kembali
                    cmd_vel.linear.x = 0.0
                    cmd_vel.linear.y = 0.0
                    cmd_vel.angular.z = 0.0
            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                
            # Transisi ke Final Gate jika durasi follow line > 25 detik
            # (Atau jika mendeteksi pintu gate terakhir, namun karena kamera menghadap bawah,
            # durasi waktu terbang / GPS adalah pemicu paling andal sebelum kamera diputar depan lagi)
            if time.time() - self.state_timer > 25.0:
                self.get_logger().info("Misi Line Following selesai! Memutar kamera ke depan mencari Gate Terakhir...")
                self.mission_state = MissionState.GOTO_FINAL_GATE
                self.target_altitude = 1.0
                self.set_camera_servo(0.0)  # Putar kamera ke depan
                self.state_timer = time.time()

        # --- MISI 5: FINAL GATE & PRECISION LANDING ---
        elif self.mission_state == MissionState.GOTO_FINAL_GATE:
            cmd_vel.linear.x = 0.22
            if target_detection is not None:
                self.mission_state = MissionState.CENTERING_FINAL_GATE
                self.pid_roll.reset()
                self.pid_yaw.reset()

        elif self.mission_state == MissionState.CENTERING_FINAL_GATE:
            if target_detection is not None:
                h, w = frame.shape[:2]
                err_x, err_y = self.detector.get_centering_error(target_detection, w, h, self.current_camera_angle)
                cmd_vel.linear.y = -self.pid_roll.update(err_x)
                cmd_vel.angular.z = -self.pid_yaw.update(err_x)
                cmd_vel.linear.x = 0.12
                
                if abs(err_x) < 20.0 and target_detection["width"] > (w * 0.45):
                    self.mission_state = MissionState.PASSING_FINAL_GATE
                    self.state_timer = time.time()
            else:
                cmd_vel.linear.x = 0.05
                cmd_vel.linear.y = 0.0

        elif self.mission_state == MissionState.PASSING_FINAL_GATE:
            cmd_vel.linear.x = 0.38
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.0
            if time.time() - self.state_timer > 3.5:
                self.get_logger().info("Gate terakhir dilewati! Memutar kamera ke bawah mencari Landing Pad...")
                self.mission_state = MissionState.SEARCHING_LANDING_PAD
                self.set_camera_servo(90.0)  # Kamera diputar ke bawah untuk mencari area pendaratan
                self.state_timer = time.time()

        elif self.mission_state == MissionState.SEARCHING_LANDING_PAD:
            # Drone berputar perlahan untuk memindai landing pad H (5x5m) di bawahnya
            cmd_vel.linear.x = 0.05  # Maju perlahan
            cmd_vel.linear.y = 0.0
            cmd_vel.angular.z = 0.12  # Yaw rate lambat
            
            if target_detection is not None and target_detection["class"] == "landing_pad":
                self.get_logger().info("Landing Pad terdeteksi! Memulai centering...")
                self.mission_state = MissionState.CENTERING_LANDING_PAD
                self.pid_roll.reset()
                self.pid_pitch.reset()
                self.centering_start_time = None

        elif self.mission_state == MissionState.CENTERING_LANDING_PAD:
            if target_detection is not None:
                h, w = frame.shape[:2]
                err_x, err_y = self.detector.get_centering_error(target_detection, w, h, self.current_camera_angle)
                
                # Visual servoing presisi
                cmd_vel.linear.y = -self.pid_roll.update(err_x)
                cmd_vel.linear.x = -self.pid_pitch.update(err_y)
                
                # Gunakan filter inersia 2 detik agar drone tidak mendarat dalam keadaan miring/melaju kencang
                if abs(err_x) < 12.0 and abs(err_y) < 12.0:
                    if self.centering_start_time is None:
                        self.centering_start_time = time.time()
                        self.get_logger().info("Target Landing Pad centered. Memulai kalkulasi stabilitas inersia...")
                    elif time.time() - self.centering_start_time > 2.0:
                        self.get_logger().info("Drone stabil di atas Landing Pad! Mengirim perintah LAND...")
                        self.mission_state = MissionState.DESCENDING_FOR_LAND
                        self.state_timer = time.time()
                else:
                    if self.centering_start_time is not None:
                        self.centering_start_time = None
            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                self.centering_start_time = None

        elif self.mission_state == MissionState.DESCENDING_FOR_LAND:
            # Panggil service LAND MAVROS
            self.set_mode("LAND")
            self.get_logger().info("Perintah LAND dikirim ke Pixhawk. Misi Selesai!")
            self.mission_state = MissionState.LANDED

        elif self.mission_state == MissionState.LANDED:
            # Kurangi daya motor (throttle minimum)
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.0
            cmd_vel.linear.z = -0.2
            cmd_vel.angular.z = 0.0

        # Kirim command kecepatan ke FCU
        self.vel_pub.publish(cmd_vel)

    # ------------------------------------------------------------------
    # MAVROS SERVICES HELPERS
    # ------------------------------------------------------------------
    def set_mode(self, mode_str: str) -> None:
        if not self.set_mode_client.service_is_ready():
            self.get_logger().warn("Service set_mode belum siap!")
            return
        
        req = SetMode.Request()
        req.custom_mode = mode_str
        
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(lambda f: self._service_response_cb(f, f"SetMode {mode_str}"))

    def arm_drone(self, arm_status: bool) -> None:
        if not self.arming_client.service_is_ready():
            self.get_logger().warn("Service arming belum siap!")
            return
            
        req = CommandBool.Request()
        req.value = arm_status
        
        future = self.arming_client.call_async(req)
        future.add_done_callback(lambda f: self._service_response_cb(f, f"Arming {arm_status}"))

    def send_takeoff_cmd(self) -> None:
        self.publish_position_target(0.0, 0.0, self.target_altitude)

    def publish_position_target(self, x: float, y: float, z: float) -> None:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        self.pos_pub.publish(pose)

    def trigger_servo(self) -> None:
        self.get_logger().info("=== ACTION: SERVO UNLOCKED! PAYLOAD MEDICAL KIT DROPPED ===")

    def _service_response_cb(self, future, cmd_name: str) -> None:
        try:
            res = future.result()
            if res.success or (hasattr(res, 'mode_sent') and res.mode_sent):
                self.get_logger().info(f"Command '{cmd_name}' BERHASIL.")
            else:
                self.get_logger().error(f"Command '{cmd_name}' GAGAL.")
        except Exception as e:
            self.get_logger().error(f"Error memanggil service '{cmd_name}': {e}")

    # ------------------------------------------------------------------
    # NAVIGATION HELPERS
    # ------------------------------------------------------------------
    def get_current_yaw(self) -> float:
        q = self.current_pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def log_mission_status(self) -> None:
        state_names = {v: k for k, v in MissionState.__dict__.items() if not k.startswith('_')}
        state_str = state_names.get(self.mission_state, str(self.mission_state))
        self.get_logger().info(
            f"[FSM STATUS] State: {state_str} | Camera Angle: {self.current_camera_angle:.1f} deg | "
            f"Mode: {self.current_state.mode} | Alt: {self.current_pose.pose.position.z:.2f}m"
        )

    def destroy_node(self):
        if self.use_direct_cam:
            if self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            self.get_logger().info("Kamera dan GUI ditutup.")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VTOLMissionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Menerima KeyboardInterrupt, menghentikan node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
