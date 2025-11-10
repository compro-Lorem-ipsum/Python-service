from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os

class MilvusDB:
    def __init__(self):
        self.collection_name = "face_embeddings"
        self.dim = 512  # ArcFace embedding dimension
        
    def connect(self, host="localhost", port="19530"):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port
            )
            print(f"Connected to Milvus at {host}:{port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            return False
    
    def create_collection(self):
        """Create collection if not exists"""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                print(f"Collection {self.collection_name} already exists")
                return Collection(self.collection_name)
            
            # Define fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            
            # Create schema
            schema = CollectionSchema(
                fields=fields,
                description="Face embeddings for employee attendance"
            )
            
            # Create collection
            collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # Create IVF_FLAT index for vector search
            index_params = {
                "metric_type": "IP",  # Inner Product (for cosine similarity)
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            print(f"Collection {self.collection_name} created successfully")
            return collection
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            return None
    
    def get_collection(self):
        """Get existing collection"""
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.load()
                return collection
            else:
                print(f"Collection {self.collection_name} does not exist")
                return None
        except Exception as e:
            print(f"Error getting collection: {e}")
            return None
    
    def insert_embedding(self, employee_id, embedding):
        """Insert employee face embedding"""
        try:
            collection = self.get_collection()
            if collection is None:
                collection = self.create_collection()
            
            # Prepare data
            data = [
                [employee_id],  # employee_id
                [embedding.tolist()]  # embedding
            ]
            
            # Insert
            result = collection.insert(data)
            collection.flush()
            
            print(f"Inserted embedding for employee_id: {employee_id}")
            return {
                "success": True,
                "employee_id": employee_id,
                "insert_count": result.insert_count
            }
            
        except Exception as e:
            print(f"Error inserting embedding: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_similar(self, query_embedding, threshold=0.6, limit=1):
        """Search for similar face embeddings"""
        try:
            collection = self.get_collection()
            if collection is None:
                return {
                    "success": False,
                    "error": "Collection not found"
                }
            
            # Search parameters
            search_params = {
                "metric_type": "IP",  # Inner Product
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["employee_id"]
            )
            
            # Process results
            if len(results) > 0 and len(results[0]) > 0:
                top_match = results[0][0]
                similarity = top_match.distance  # IP similarity score
                
                # Check threshold
                if similarity >= threshold:
                    return {
                        "success": True,
                        "employee_id": top_match.entity.get("employee_id"),
                        "similarity": float(similarity),
                        "matched": True
                    }
                else:
                    return {
                        "success": True,
                        "matched": False,
                        "similarity": float(similarity),
                        "message": f"Similarity {similarity:.3f} below threshold {threshold}"
                    }
            else:
                return {
                    "success": True,
                    "matched": False,
                    "message": "No faces found in database"
                }
                
        except Exception as e:
            print(f"Error searching: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_by_employee_id(self, employee_id):
        """Delete embeddings by employee_id"""
        try:
            collection = self.get_collection()
            if collection is None:
                return {
                    "success": False,
                    "error": "Collection not found"
                }
            
            # Delete
            expr = f'employee_id == "{employee_id}"'
            collection.delete(expr)
            collection.flush()
            
            print(f"Deleted embeddings for employee_id: {employee_id}")
            return {
                "success": True,
                "employee_id": employee_id
            }
            
        except Exception as e:
            print(f"Error deleting: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        print("Disconnected from Milvus")