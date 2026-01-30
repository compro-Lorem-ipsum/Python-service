import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognitionService:
    def __init__(self, model_name: str, providers: list[str], det_size):
        self._det_size = det_size
        self._warmed_up = False

        self.app = self._init_with_fallback(model_name, providers, det_size)

    def _init_with_fallback(self, model_name: str, providers: list[str], det_size):
        provider_options = [providers]
        if providers != ["CPUExecutionProvider"]:
            provider_options.append(["CPUExecutionProvider"])

        for prov in provider_options:
            try:
                print("[InsightFace] Trying providers:", prov)

                app = FaceAnalysis(name=model_name, providers=prov)
                app.prepare(ctx_id=0, det_size=det_size)

                for name, model in app.models.items():
                    print(f"[InsightFace] {name} providers:", model.session.get_providers())

                return app

            except Exception as exc:
                print("[InsightFace] Failed providers:", prov, exc)

        raise RuntimeError("Failed to initialize FaceAnalysis")

    def warmup(self):
        if self._warmed_up:
            return
        dummy = np.zeros((self._det_size[1], self._det_size[0], 3), dtype=np.uint8)
        self.app.get(dummy)
        self._warmed_up = True

    def extract_embedding_from_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._process_image(img)

    def _process_image(self, img):
        if img is None:
            return {"success": False, "error": "Invalid image"}

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
            "det_score": float(face.det_score),
        }
    
    
