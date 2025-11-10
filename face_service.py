import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

class FaceRecognitionService:
    def __init__(self):
        """Initialize ArcFace model"""
        self.app = FaceAnalysis(
            name='buffalo_l',  # ArcFace model
            providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("ArcFace model loaded successfully")
    
    def extract_embedding(self, image_path):
        """
        Extract face embedding from image
        
        Args:
            image_path: Path to image file or numpy array
            
        Returns:
            dict with success status and embedding or error
        """
        try:
            # Read image
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    return {
                        "success": False,
                        "error": "Failed to read image"
                    }
            elif isinstance(image_path, np.ndarray):
                img = image_path
            else:
                return {
                    "success": False,
                    "error": "Invalid image input"
                }
            
            # Detect faces
            faces = self.app.get(img)
            
            if len(faces) == 0:
                return {
                    "success": False,
                    "error": "No face detected in image"
                }
            
            if len(faces) > 1:
                return {
                    "success": False,
                    "error": f"Multiple faces detected ({len(faces)}). Please use image with single face"
                }
            
            # Get embedding from the first (and only) face
            face = faces[0]
            embedding = face.normed_embedding  # Already normalized
            
            return {
                "success": True,
                "embedding": embedding,
                "bbox": face.bbox.tolist(),  # Bounding box
                "det_score": float(face.det_score)  # Detection confidence
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error extracting embedding: {str(e)}"
            }
    
    def extract_embedding_from_bytes(self, image_bytes):
        """
        Extract face embedding from image bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            dict with success status and embedding or error
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    "success": False,
                    "error": "Failed to decode image"
                }
            
            return self.extract_embedding(img)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing image bytes: {str(e)}"
            }
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two embeddings using cosine similarity
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Compute cosine similarity (dot product of normalized vectors)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            print(f"Error comparing embeddings: {e}")
            return 0.0