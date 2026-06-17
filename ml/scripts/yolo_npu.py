"""
scripts/yolo_npu.py
====================
Inference Engine untuk menjalankan YOLOv8/v11 di NPU Orange Pi 5 (RK3588).

Modul ini menggunakan library `rknnlite` untuk melakukan akselerasi hardware.
Jika dijalankan di luar Orange Pi 5 (misal di laptop Windows/Mac), modul ini akan
secara otomatis melakukan fallback ke CPU menggunakan ONNX Runtime (`onnxruntime`)
sehingga script pengujian tetap dapat dijalankan tanpa error import.
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("YOLONPU")

# Deteksi ketersediaan library RKNN-Lite (hanya ada di Orange Pi/Rockchip Linux)
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    logger.warning("RKNNLite tidak terdeteksi. Mode NPU dinonaktifkan. Menggunakan fallback ONNX Runtime CPU.")

# Deteksi ketersediaan ONNX Runtime untuk mode fallback laptop
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    if not HAS_RKNN:
        logger.error("ONNX Runtime tidak terdeteksi. Silakan jalankan: pip install onnxruntime")


class YOLONPU:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.45) -> None:
        """
        Inisialisasi Model YOLO untuk NPU Orange Pi 5 (RKNN) atau CPU (ONNX).
        
        Args:
            model_path: Path ke file model (.rknn untuk NPU, .onnx untuk CPU/Laptop)
            conf_threshold: Confidence threshold minimum
            nms_threshold: Non-Maximum Suppression overlap threshold
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Nama kelas target default
        self.names = {0: "gate", 1: "red_box", 2: "landing_pad"}
        
        # Cek tipe model dari ekstensi
        _, ext = os.path.splitext(model_path)
        self.is_rknn = ext.lower() == ".rknn"

        # Inisialisasi runtime
        if self.is_rknn:
            if not HAS_RKNN:
                raise ImportError("Model bertipe RKNN, tetapi library RKNNLite tidak terinstall di sistem ini!")
            self._init_rknn()
        else:
            if not HAS_ORT:
                raise ImportError("Model bertipe ONNX, tetapi library onnxruntime tidak terinstall di sistem ini!")
            self._init_onnx()

    def _init_rknn(self) -> None:
        """Inisialisasi runtime RKNN-Lite di NPU Orange Pi 5."""
        logger.info(f"Menginisialisasi NPU Orange Pi 5 menggunakan model RKNN: {self.model_path}")
        self.rknn = RKNNLite()
        
        # Load model rknn
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            logger.error(f"Gagal memuat file RKNN: {self.model_path}")
            return
            
        # Inisialisasi runtime dengan auto NPU core allocation (Core 0, 1, atau 2)
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            logger.error("Gagal menginisialisasi runtime NPU RKNN!")
            return
        logger.info("Runtime NPU RKNN berhasil diinisialisasi.")

    def _init_onnx(self) -> None:
        """Inisialisasi runtime ONNX di CPU (Laptop/Dev mode)."""
        logger.info(f"Menginisialisasi CPU ONNX Runtime menggunakan model ONNX: {self.model_path}")
        # Gunakan CPU Execution Provider
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Runtime ONNX berhasil diinisialisasi.")

    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        Melakukan inferensi dan mengembalikan hasil deteksi bounding box.
        
        Args:
            frame: Gambar input OpenCV format BGR (H, W, C)
            
        Returns:
            List dictionary berisi hasil deteksi:
            [
                {
                    'class': str,
                    'x': float,   # Center X
                    'y': float,   # Center Y
                    'conf': float,
                    'bbox': [x1, y1, x2, y2],
                    'width': float,
                    'height': float
                }, ...
            ]
        """
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Pre-processing: Resize ke 640x640 (standard YOLOv8)
        # RKNN biasanya dikonfigurasi dengan RGB input, sedangkan OpenCV menggunakan BGR
        input_size = 640
        img_resized = cv2.resize(frame, (input_size, input_size))
        
        if self.is_rknn:
            # Model RKNN biasanya ditraining dengan format RGB
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Input RKNN membutuhkan dimensi [1, H, W, C] dan tipe uint8 (tanpa normalisasi /255.0,
            # karena normalisasi biasanya sudah disematkan di dalam file RKNN saat konversi)
            img_input = np.expand_dims(img_input, axis=0)
            
            # Jalankan inferensi di NPU
            outputs = self.rknn.inference(inputs=[img_input])
        else:
            # ONNX Runtime memerlukan normalisasi float32 [1, C, H, W]
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype(np.float32) / 255.0
            img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
            img_input = np.expand_dims(img_input, axis=0)   # CHW -> BCHW
            
            # Jalankan inferensi di CPU
            outputs = self.session.run(None, {self.input_name: img_input})

        # 2. Post-processing: Parsing output YOLOv8 [1, 4 + num_classes, 8400]
        # YOLOv8 mengembalikan satu tensor berisi bounding box + class scores untuk 8400 anchor points
        raw_output = outputs[0]
        if len(raw_output.shape) == 3:
            raw_output = raw_output[0] # Ambil batch 0 -> [84, 8400]
            
        # Transpose ke bentuk [8400, 84] agar gampang di-slice per anchor point
        predictions = raw_output.T
        
        # Pecah kolom box (0,1,2,3) dan class scores (4 s/d end)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Cari confidence score tertinggi dan index class untuk tiap anchor point
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter berdasarkan confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Convert box format dari [center_x, center_y, w, h] ke [x1, y1, w, h] untuk OpenCV NMS
        # Serta skalakan kembali koordinat dari 640 ke resolusi frame asli
        scale_x = orig_w / float(input_size)
        scale_y = orig_h / float(input_size)
        
        cv_boxes = []
        cv_confidences = []
        cv_class_ids = []
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            cx, cy, w, h = box
            
            # x1, y1 adalah pojok kiri atas
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            box_w = int(w * scale_x)
            box_h = int(h * scale_y)
            
            cv_boxes.append([x1, y1, box_w, box_h])
            cv_confidences.append(float(conf))
            cv_class_ids.append(int(cls_id))
            
        # 3. Non-Maximum Suppression (NMS) menggunakan OpenCV C++ API yang sangat cepat
        indices = cv2.dnn.NMSBoxes(
            bboxes=cv_boxes,
            scores=cv_confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )
        
        detections = []
        if len(indices) > 0:
            # Ratakan indices jika OpenCV mengembalikannya dalam bentuk list 2D
            for i in indices.flatten():
                x1, y1, box_w, box_h = cv_boxes[i]
                conf = cv_confidences[i]
                cls_id = cv_class_ids[i]
                cls_name = self.names.get(cls_id, str(cls_id))
                
                # Titik tengah box
                cx = x1 + (box_w / 2.0)
                cy = y1 + (box_h / 2.0)
                
                detections.append({
                    "class": cls_name,
                    "x": float(cx),
                    "y": float(cy),
                    "conf": conf,
                    "bbox": [float(x1), float(y1), float(x1 + box_w), float(y1 + box_h)],
                    "width": float(box_w),
                    "height": float(box_h)
                })
                
        return detections

    def release(self) -> None:
        """Melepas context runtime NPU."""
        if self.is_rknn and hasattr(self, 'rknn'):
            self.rknn.release()
            logger.info("Context NPU RKNN berhasil dilepas.")
