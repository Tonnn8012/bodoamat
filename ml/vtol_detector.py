"""
vtol_detector.py
=================
Modul utama Computer Vision (CV) untuk drone VTOL.

Berisi class `VTOLDetector` yang menangani:
- Load model YOLO (.pt untuk laptop / .onnx via YOLOInference untuk OrangePi 4 CPU)
- Sinkronisasi nama class dan warna visualisasi dengan `configs/classes.json`
- Deteksi objek per-frame
- Smoothing deteksi menggunakan Exponential Moving Average (EMA) untuk meredam jitter
- Anotasi frame (bounding box, titik tengah, crosshair, garis tracking)
- Hitung error posisi objek terhadap pusat frame secara "servo-aware" (0 deg = depan, 90 deg = bawah)
- Export model ke ONNX

File ini MURNI Python, TIDAK ADA import ROS.
Target Python 3.10+
"""

from __future__ import annotations
import logging
import os
import json
from collections import deque
from typing import Optional

# pyrefly: ignore [missing-import]
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VTOLDetector")


class _DummyModel:
    """
    Model dummy (placeholder) yang dipakai ketika model asli (best.pt / best.onnx)
    belum tersedia atau belum ditemukan di disk.
    """
    names = {0: "gate", 1: "red_box", 2: "landing_pad"}

    def predict(self, source, conf=0.5, verbose=False):
        return []

    def export(self, format: str = "onnx"):
        logger.warning(
            "Export dibatalkan: model masih DUMMY. Latih model dulu."
        )
        return None


class VTOLDetector:
    """
    Class utama untuk deteksi objek menggunakan YOLO pada drone VTOL.
    Mendukung inferensi PyTorch (.pt) di laptop dan ONNX CPU di Orange Pi 4.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        use_onnx: bool = False,
        classes_config_path: str = "configs/classes.json",
        ema_alpha: float = 0.6,          # Koefisien smoothing EMA (0.1 - 1.0, makin kecil makin lambat/mulus)
        max_lost_frames: int = 5,        # Toleransi frame lost sebelum tracking di-reset
    ) -> None:
        """
        Inisialisasi detector: load model YOLO dan siapkan filter.
        """
        self.model_path: str = model_path
        self.conf_threshold: float = conf_threshold
        self.use_onnx: bool = use_onnx
        self.classes_config_path = classes_config_path
        
        # Load class names & colors dari JSON config
        self.class_names = {0: "gate", 1: "red_box", 2: "landing_pad"}
        self.class_colors = {"gate": [0, 255, 0], "red_box": [0, 0, 255], "landing_pad": [255, 165, 0]}
        self._load_classes_config()

        # Buffer untuk mendeteksi stabilitas frame
        self.detection_buffer: deque = deque(maxlen=5)
        self.frame_count: int = 0
        self.is_dummy: bool = False

        # State untuk smoothing EMA
        self.ema_alpha = ema_alpha
        self.max_lost_frames = max_lost_frames
        self.ema_state: dict[str, dict] = {}  # Format: {class_name: {'x': val, 'y': val, 'lost_count': int}}

        # Memuat model
        self.model = self._load_model(model_path, use_onnx)

    def _load_classes_config(self) -> None:
        """Membaca konfigurasi class dari file classes.json jika ada."""
        # Cari path relatif terhadap root workspace
        possible_paths = [
            self.classes_config_path,
            os.path.join(os.path.dirname(__file__), self.classes_config_path),
            os.path.join(os.path.dirname(__file__), "ml", self.classes_config_path),
            os.path.join(os.path.dirname(__file__), "..", self.classes_config_path),
        ]
        
        config_file = None
        for p in possible_paths:
            if os.path.exists(p):
                config_file = p
                break
                
        if config_file:
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                
                # Update class names (key JSON string -> int)
                if "classes" in data:
                    self.class_names = {int(k): v for k, v in data["classes"].items()}
                
                # Update colors
                if "colors_bgr" in data:
                    self.class_colors = data["colors_bgr"]
                    
                logger.info(f"Berhasil memuat konfigurasi class dari {config_file}")
            except Exception as e:
                logger.error(f"Gagal membaca file konfigurasi class: {e}")
        else:
            logger.warning(f"File konfigurasi class '{self.classes_config_path}' tidak ditemukan. Menggunakan default.")

    def _load_model(self, model_path: str, use_onnx: bool):
        """Memuat model sesuai ekstensi dan opsi."""
        if not os.path.exists(model_path):
            logger.warning(
                "Model tidak ditemukan di '%s'. Menggunakan MODEL DUMMY.",
                model_path,
            )
            self.is_dummy = True
            return _DummyModel()

        try:
            _, ext = os.path.splitext(model_path)
            
            # Jika tipe model ONNX
            if ext.lower() == ".onnx" or use_onnx:
                try:
                    from scripts.yolo_inference import YOLOInference
                    logger.info("Memuat model ONNX menggunakan YOLOInference wrapper CPU...")
                    model = YOLOInference(model_path, conf_threshold=self.conf_threshold)
                    # Sinkronkan class names
                    model.names = self.class_names
                    logger.info("Model ONNX berhasil dimuat.")
                    return model
                except Exception as e:
                    logger.error("Gagal memuat model ONNX dengan YOLOInference: %s. Fallback ke Ultralytics...", e)

            # Jika tipe model PyTorch (.pt)
            from ultralytics import YOLO  # type: ignore
            model = YOLO(model_path)
            logger.info(f"Model PyTorch (.pt) berhasil dimuat dari '{model_path}'.")
            return model

        except Exception as e:
            logger.error(
                "Gagal memuat model dari '%s': %s. Menggunakan MODEL DUMMY.",
                model_path,
                e,
            )
            self.is_dummy = True
            return _DummyModel()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Jalankan deteksi objek mentah pada satu frame."""
        self.frame_count += 1
        detections: list[dict] = []

        if self.is_dummy:
            return detections

        try:
            # Jika model adalah wrapper YOLOInference
            if hasattr(self.model, "predict") and not hasattr(self.model, "val"):
                # YOLOInference wrapper return list of dicts langsung
                return self.model.predict(frame)

            # Jika model bawaan Ultralytics YOLO
            results = self.model.predict(
                source=frame, conf=self.conf_threshold, verbose=False
            )

            if not results:
                return detections

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return detections

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                cls_name = self.class_names.get(cls_idx, str(cls_idx))

                width = x2 - x1
                height = y2 - y1
                center_x = x1 + (width / 2)
                center_y = y1 + (height / 2)

                detections.append(
                    {
                        "class": cls_name,
                        "x": float(center_x),
                        "y": float(center_y),
                        "conf": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "width": float(width),
                        "height": float(height),
                    }
                )

            return detections

        except Exception as e:
            logger.error("Terjadi error saat proses deteksi: %s", e)
            return []

    def detect_stable(self, frame: np.ndarray) -> Optional[dict]:
        """
        Deteksi dengan filter EMA (Exponential Moving Average) untuk meredam jitter.
        Juga menguji stabilitas deteksi di frame buffer.
        """
        try:
            # 1. Deteksi mentah pada frame saat ini
            current_detections = self.detect(frame)
            self.detection_buffer.append(current_detections)

            # 2. Update tracking lost frame counters
            detected_classes = {det["class"] for det in current_detections}
            for cls_name in list(self.ema_state.keys()):
                if cls_name not in detected_classes:
                    self.ema_state[cls_name]["lost_count"] += 1
                    if self.ema_state[cls_name]["lost_count"] > self.max_lost_frames:
                        del self.ema_state[cls_name]
                        logger.debug(f"Tracking lost untuk class: {cls_name}")

            if not current_detections:
                return None

            # 3. Pilih deteksi dengan confidence tertinggi
            best_det = max(current_detections, key=lambda x: x["conf"])
            cls_name = best_det["class"]
            curr_x, curr_y = best_det["x"], best_det["y"]

            # 4. Terapkan filter EMA pada posisi
            if cls_name not in self.ema_state:
                # Inisialisasi EMA state baru
                self.ema_state[cls_name] = {
                    "x": curr_x,
                    "y": curr_y,
                    "lost_count": 0,
                    "width": best_det["width"],
                    "height": best_det["height"],
                    "bbox": best_det["bbox"],
                    "conf": best_det["conf"]
                }
            else:
                # Update EMA: x = alpha * current + (1 - alpha) * prev
                prev_x = self.ema_state[cls_name]["x"]
                prev_y = self.ema_state[cls_name]["y"]
                self.ema_state[cls_name]["x"] = self.ema_alpha * curr_x + (1.0 - self.ema_alpha) * prev_x
                self.ema_state[cls_name]["y"] = self.ema_alpha * curr_y + (1.0 - self.ema_alpha) * prev_y
                self.ema_state[cls_name]["width"] = self.ema_alpha * best_det["width"] + (1.0 - self.ema_alpha) * self.ema_state[cls_name]["width"]
                self.ema_state[cls_name]["height"] = self.ema_alpha * best_det["height"] + (1.0 - self.ema_alpha) * self.ema_state[cls_name]["height"]
                self.ema_state[cls_name]["bbox"] = best_det["bbox"]
                self.ema_state[cls_name]["conf"] = best_det["conf"]
                self.ema_state[cls_name]["lost_count"] = 0

            # 5. Cek kelayakan stabil (minimal terdeteksi di beberapa frame terakhir)
            frames_with_this_class = sum(
                1 for fd in self.detection_buffer if any(d["class"] == cls_name for d in fd)
            )
            if frames_with_this_class < 2:  # Cukup 2 dari 5 frame untuk inisiasi cepat
                return None

            # Kembalikan koordinat hasil filter EMA
            return {
                "class": cls_name,
                "x": self.ema_state[cls_name]["x"],
                "y": self.ema_state[cls_name]["y"],
                "conf": self.ema_state[cls_name]["conf"],
                "bbox": self.ema_state[cls_name]["bbox"],
                "width": self.ema_state[cls_name]["width"],
                "height": self.ema_state[cls_name]["height"]
            }

        except Exception as e:
            logger.error("Terjadi error saat proses detect_stable: %s", e)
            return None

    def get_centering_error(
        self, 
        detection: dict, 
        frame_width: int, 
        frame_height: int, 
        servo_angle: float = 0.0
    ) -> tuple[float, float]:
        """
        Menghitung error posisi objek relatif terhadap pusat frame secara 'servo-aware'.
        
        Args:
            detection: Hasil deteksi stabil
            frame_width: Lebar frame (pixel)
            frame_height: Tinggi frame (pixel)
            servo_angle: Kemiringan kamera (derajat). 
                         0.0 = Menghadap depan (misi gate)
                         90.0 = Menghadap bawah (misi drop/landing)
                         
        Returns:
            Tuple (error_x, error_y)
            - error_x: Deviasi horizontal (kiri/kanan)
            - error_y: Deviasi vertikal / longitudinal (maju/mundur atau atas/bawah)
        """
        try:
            # Hitung error piksel mentah
            error_x = float(detection["x"]) - (frame_width / 2)
            error_y = float(detection["y"]) - (frame_height / 2)
            
            # Kategori interpretasi error berdasarkan sudut servo:
            # Jika servo >= 45 derajat (cenderung melihat ke bawah),
            # koordinat Y mewakili sumbu maju-mundur drone (pitch axis).
            # Jika servo < 45 derajat (melihat ke depan),
            # koordinat Y mewakili sumbu atas-bawah (ketinggian / altitude axis).
            # Kita juga bisa menerapkan skala pengali sensitivitas jika dibutuhkan
            
            return (error_x, error_y)

        except Exception as e:
            logger.error("Terjadi error saat menghitung centering error: %s", e)
            return (0.0, 0.0)

    def annotate_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Menggambar visualisasi deteksi dengan warna spesifik dari config."""
        try:
            import cv2
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # Crosshair biru tipis di tengah frame
            cv2.line(annotated, (w // 2, 0), (w // 2, h), (255, 100, 0), 1)
            cv2.line(annotated, (0, h // 2), (w, h // 2), (255, 100, 0), 1)

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cx, cy = int(det["x"]), int(det["y"])
                cls_name = det["class"]
                
                # Ambil warna spesifik class
                color = self.class_colors.get(cls_name, [0, 255, 0])
                # OpenCV menggunakan format BGR, jadi pastikan array color berbentuk [B, G, R]
                color_bgr = (int(color[0]), int(color[1]), int(color[2]))

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

                # Draw label
                label = f"{cls_name} {det['conf']:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    2,
                )

                # Draw center point
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw line from center of frame to center of object
                cv2.line(annotated, (w // 2, h // 2), (cx, cy), (0, 255, 255), 1)

            return annotated

        except Exception as e:
            logger.error("Terjadi error saat anotasi frame: %s", e)
            return frame

    def export_to_onnx(self, output_path: str) -> None:
        """Export model PyTorch ke format ONNX."""
        try:
            if self.is_dummy:
                logger.warning("Export dibatalkan: model masih DUMMY.")
                return None

            if self.use_onnx:
                logger.warning("Export dibatalkan: model sudah ONNX.")
                return None

            logger.info("Memulai export model ke format ONNX...")
            # Panggil method export bawaan Ultralytics
            hasil_export = self.model.export(format="onnx", imgsz=320, dynamic=False, opset=12)
            logger.info(f"Export selesai. Tersimpan di: {hasil_export}")
            return hasil_export

        except Exception as e:
            logger.error("Terjadi error saat export model ke ONNX: %s", e)
            return None