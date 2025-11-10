from flask import Flask, request, jsonify
from face_service import FaceRecognitionService
from milvus_db import MilvusDB
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize services
face_service = FaceRecognitionService()
milvus_db = MilvusDB()

# Connect to Milvus
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')

if not milvus_db.connect(host=MILVUS_HOST, port=MILVUS_PORT):
    print("Warning: Failed to connect to Milvus. Please ensure Milvus is running.")
else:
    # Create collection if not exists
    milvus_db.create_collection()

# Configuration
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "face-recognition-api"
    }), 200

@app.route('/enroll', methods=['POST'])
def enroll_employee():
    """
    Enroll new employee face
    Expected: employee_id and image file
    """
    try:
        # Check if employee_id is provided
        if 'employee_id' not in request.form:
            return jsonify({
                "success": False,
                "error": "employee_id is required"
            }), 400
        
        employee_id = request.form['employee_id']
        
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "image file is required"
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image file selected"
            }), 400
        
        # Read image bytes
        image_bytes = image_file.read()
        
        # Extract embedding
        result = face_service.extract_embedding_from_bytes(image_bytes)
        
        if not result['success']:
            return jsonify(result), 400
        
        embedding = result['embedding']
        
        # Insert to Milvus
        insert_result = milvus_db.insert_embedding(employee_id, embedding)
        
        if not insert_result['success']:
            return jsonify(insert_result), 500
        
        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "message": "Employee enrolled successfully",
            "detection_score": result['det_score']
        }), 201
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/verify', methods=['POST'])
def verify_face():
    """
    Verify face for attendance
    Expected: image file
    Returns: employee_id if match found
    """
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "image file is required"
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image file selected"
            }), 400
        
        # Read image bytes
        image_bytes = image_file.read()
        
        # Extract embedding
        extract_result = face_service.extract_embedding_from_bytes(image_bytes)
        
        if not extract_result['success']:
            return jsonify(extract_result), 400
        
        embedding = extract_result['embedding']
        
        # Get threshold from request or use default
        threshold = float(request.form.get('threshold', SIMILARITY_THRESHOLD))
        
        # Search in Milvus
        search_result = milvus_db.search_similar(embedding, threshold=threshold)
        
        if not search_result['success']:
            return jsonify(search_result), 500
        
        if not search_result['matched']:
            return jsonify({
                "success": True,
                "matched": False,
                "message": search_result.get('message', 'No match found'),
                "similarity": search_result.get('similarity', 0.0),
                "threshold": threshold
            }), 200
        
        return jsonify({
            "success": True,
            "matched": True,
            "employee_id": search_result['employee_id'],
            "similarity": search_result['similarity'],
            "threshold": threshold,
            "detection_score": extract_result['det_score']
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/delete/<employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    """Delete employee face data"""
    try:
        result = milvus_db.delete_by_employee_id(employee_id)
        
        if not result['success']:
            return jsonify(result), 500
        
        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "message": "Employee data deleted successfully"
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/extract-embedding', methods=['POST'])
def extract_embedding_only():
    """
    Extract embedding without storing
    Useful for testing
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "image file is required"
            }), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        result = face_service.extract_embedding_from_bytes(image_bytes)
        
        if not result['success']:
            return jsonify(result), 400
        
        # Convert embedding to list for JSON serialization
        result['embedding'] = result['embedding'].tolist()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )