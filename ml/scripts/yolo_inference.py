"""
scripts/yolo_inference.py
==========================
Engine Inferensi CPU-only menggunakan ONNX Runtime untuk Orange Pi 4 (RK3399).
Menggantikan `yolo_npu.py` karena RK3399 tidak memiliki hardware NPU.

Optimasi Performa CPU ARM:
1. Mengatur `intra_op_num_threads` dan `inter_op_num_threads` agar tidak overhead.
2. Membaca dimensi input model secara dinamis (misal 320x320) langsung dari metadata ONNX.
3. Menggunakan OpenCV NMS (Non-Maximum Suppression) berbasis C++ yang sangat cepat.
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("YOLOInference")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    logger.error("ONNX Runtime tidak terdeteksi. Silakan jalankan: pip install onnxruntime")


class YOLOInference:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> None:
        """
        Inisialisasi Model YOLO menggunakan ONNX Runtime CPU.
        
        Args:
            model_path: Path ke file model (.onnx)
            conf_threshold: Ambang batas minimum confidence
            nms_threshold: Ambang batas overlap NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Kelas default kompetisi VTOL KRTI 2026
        self.names = {0: "gate", 1: "red_box", 2: "landing_pad"}
        
        if not HAS_ORT:
            raise ImportError(
                "Library onnxruntime tidak terpasang di sistem! "
                "Jalankan: pip install onnxruntime"
            )
            
        self._init_session()

    def _init_session(self) -> None:
        """Menginisialisasi ONNX Runtime Session dengan konfigurasi optimal untuk Orange Pi 4 CPU."""
        logger.info(f"Menginisialisasi ONNX Runtime CPU. Model: {self.model_path}")
        
        # Opsi optimasi CPU ARM
        opts = ort.SessionOptions()
        # Batasi penggunaan core agar tidak melebihi 2 core besar A72 pada Orange Pi 4 (mencegah throttling panas)
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        # Aktifkan semua optimasi dasar grafik ONNX
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            self.model_path, 
            sess_options=opts, 
            providers=["CPUExecutionProvider"]
        )
        
        # Dapatkan detail input tensor
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = model_input.shape  # Format biasanya [1, 3, H, W]
        
        # Deteksi resolusi input model secara dinamis
        self.input_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 320
        self.input_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 320
        
        logger.info(
            f"ONNX Session siap. Resolusi Input Model: {self.input_w}x{self.input_h}. "
            f"Threads: {opts.intra_op_num_threads}"
        )

    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        Melakukan inferensi pada frame gambar dan mengembalikan koordinat bounding box.
        
        Args:
            frame: Gambar input BGR (OpenCV)
            
        Returns:
            List dictionary berisi detail objek terdeteksi.
        """
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Pre-processing
        # Resize ke resolusi input model yang terdeteksi secara dinamis (misal 320x320)
        img_resized = cv2.resize(frame, (self.input_w, self.input_h))
        
        # Ubah BGR ke RGB (karena YOLO ditraining dengan RGB)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalisasi ke 0.0 - 1.0 & Transpose HWC -> CHW
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)  # BCHW
        
        # 2. Run ONNX Session
        outputs = self.session.run(None, {self.input_name: img_input})
        
        # 3. Post-processing
        # Output YOLOv8 berformat [1, 4 + num_classes, num_anchors]
        raw_output = outputs[0]
        if len(raw_output.shape) == 3:
            raw_output = raw_output[0]  # [7, 2100] untuk 320x320, 3 kelas
            
        predictions = raw_output.T  # Transpose menjadi [num_anchors, 4 + num_classes]
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Cari class dengan probabilitas tertinggi untuk setiap anchor
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter berdasarkan confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Skala konversi koordinat kembali ke resolusi gambar asli
        scale_x = orig_w / float(self.input_w)
        scale_y = orig_h / float(self.input_h)
        
        cv_boxes = []
        cv_confidences = []
        cv_class_ids = []
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            cx, cy, w, h = box
            
            # Konversi center X-Y ke pojok kiri atas (x1, y1) untuk OpenCV NMS
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            box_w = int(w * scale_x)
            box_h = int(h * scale_y)
            
            cv_boxes.append([x1, y1, box_w, box_h])
            cv_confidences.append(float(conf))
            cv_class_ids.append(int(cls_id))
            
        # Non-Maximum Suppression (NMS) OpenCV
        indices = cv2.dnn.NMSBoxes(
            bboxes=cv_boxes,
            scores=cv_confidences,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )
        
        detections = []
        if len(indices) > 0:
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
