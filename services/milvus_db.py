import time

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

class MilvusDB:
    def __init__(self):
        self.collection_name = "face_embeddings"
        self.dim = 512
        self.host = None
        self.port = None
        self.connected = False
        self._collection = None
        self._collection_loaded = False

    def connect(self, host, port, retries: int = 3, delay: float = 2.0):
        self.host = host
        self.port = port
        for attempt in range(1, retries + 1):
            try:
                connections.connect(alias="default", host=host, port=port)
                self.connected = True
                print("Connected to Milvus")
                return True
            except Exception as e:
                self.connected = False
                print(f"Failed to connect to Milvus (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(delay)
        return False

    def create_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                if not self._collection_loaded:
                    self._collection.load()
                    self._collection_loaded = True
                return self._collection

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]

            schema = CollectionSchema(fields, "Face embeddings")
            collection = Collection(self.collection_name, schema=schema)

            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }

            collection.create_index("embedding", index_params=index_params)
            collection.load()
            self._collection = collection
            self._collection_loaded = True
            return collection
        except Exception as e:
            # If the call fails because there's no connection, mark as not connected
            print(f"create_collection error: {e}")
            self.connected = False
            return None

    def get_collection(self):
        # Ensure we have a recorded connection; if not, attempt to reconnect once
        if not self.connected and self.host and self.port:
            try:
                connections.connect(alias="default", host=self.host, port=self.port)
                self.connected = True
            except Exception as e:
                print(f"Reconnection failed: {e}")
                self.connected = False
                return None

        try:
            if self._collection:
                if not self._collection_loaded:
                    self._collection.load()
                    self._collection_loaded = True
                return self._collection

            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                self._collection.load()
                self._collection_loaded = True
                return self._collection
            return None
        except Exception as e:
            print(f"get_collection error: {e}")
            self.connected = False
            return None

    def insert_embedding(self, employee_id, embedding):
        if not self.connected:
            return {"success": False, "error": "Milvus not connected"}

        col = self.get_collection()
        if col is None:
            return {"success": False, "error": "Collection not found"}

        try:
            data = [[employee_id], [embedding.tolist()]]

            result = col.insert(data)
            col.flush()

            return {"success": True, "insert_count": result.insert_count}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_similar(self, embedding, threshold, limit=1):
        if not self.connected:
            return {"success": False, "error": "Milvus not connected"}

        col = self.get_collection()
        if col is None:
            return {"success": False, "error": "Collection not found"}

        try:
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

            results = col.search(
                data=[embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["employee_id"],
            )

            if results and results[0]:
                top = results[0][0]
                sim = top.distance

                if sim >= threshold:
                    return {
                        "success": True,
                        "matched": True,
                        "employee_id": top.entity.get("employee_id"),
                        "similarity": float(sim),
                    }

                return {"success": True, "matched": False, "similarity": float(sim)}

            return {"success": True, "matched": False, "message": "No match found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_by_employee_id(self, employee_id: str) -> dict:
        if not self.connected:
            return {"success": False, "error": "Milvus not connected"}

        col = self.get_collection()
        if col is None:
            return {"success": False, "error": "Collection not found"}

        try:
            res = col.query(expr=f"employee_id == '{employee_id}'", output_fields=["employee_id"])
            if not res:
                return {"success": False, "error": f"Employee {employee_id} not found"}

            col.delete(expr=f"employee_id == '{employee_id}'")
            col.flush()
            return {"success": True, "employee_id": employee_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_employee_ids(self):
        if not self.connected:
            return {"success": False, "error": "Milvus not connected"}

        col = self.get_collection()
        if col is None:
            return {"success": False, "error": "Collection not found"}
        try:
            res = col.query(expr="employee_id != ''", output_fields=["employee_id"])
            employee_ids = [item["employee_id"] for item in res]
            return {"success": True, "employee_ids": employee_ids}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def health(self) -> dict:
        if not self.connected:
            return {"success": False, "error": "Milvus not connected"}

        try:
            if not utility.has_collection(self.collection_name):
                return {"success": False, "error": "Collection not found"}
            # Touch the collection metadata to ensure liveness
            col = self.get_collection()
            if col is None:
                return {"success": False, "error": "Collection not found"}
            return {"success": True, "collection": self.collection_name}
        except Exception as e:
            self.connected = False
            return {"success": False, "error": str(e)}
