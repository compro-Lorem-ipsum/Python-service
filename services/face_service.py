import cv2
import numpy as np

from insightface.app import FaceAnalysis

class FaceRecognitionService:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _process_image(self, img):
        faces = self.app.get(img)

        if not faces:
            return {"success": False, "error": "No face detected"}
        if len(faces) > 1:
            return {"success": False, "error": "Multiple faces detected"}

        face = faces[0]

        return {
            "success": True,
            "embedding": face.normed_embedding,
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score)
        }

    def extract_embedding_from_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Failed to decode image bytes"}

        return self._process_image(img)
