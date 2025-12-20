# Python Service


## syarat

- Python 3.8 atau lebih tinggi
- Microsoft C++ Build Tools (untuk Windows)

### Install Microsoft C++ Build Tools (Windows)

1. Buka halaman [Download Visual Studio](https://visualstudio.microsoft.com/downloads/): klik "Download Build Tools".
2. Jalankan installer yang diunduh.
3. Di jendela installer, centang opsi:

   ✅ **"Desktop development with C++"**

4. Klik "Install" dan tunggu proses selesai. Setelah selesai, restart komputer jika diminta.


---


### Buat Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

```bash
cp .env.example .env

```

## Cara menjalankan

Service ini adalah **FastAPI** dan entrypoint-nya `main:app`.

### Run lokal (Windows)

1) Pastikan `.env` sudah terisi (minimal Milvus):

- `MILVUS_HOST` (contoh: `localhost`)
- `MILVUS_PORT` (default: `19530`)
- `SIMILARITY_THRESHOLD` opsional (default: `0.6`)

2) Jalankan API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run pakai Docker

File Docker di repo ini bernama `dockerfile` (huruf kecil), jadi pakai flag `-f`.

Build image:

```bash
docker build -f dockerfile -t python-service .
```

Run container:

```bash
docker run --rm -p 8000:8000 --env-file .env python-service
```



Response (200):
```json
{ "status": "healthy" }
```

### Enroll (daftarkan wajah karyawan)
`POST /enroll`

**Input dari backend Express**:
- `employee_id` (string, wajib): ID karyawan.
- `image` (file, wajib): file gambar (JPG/PNG).

Response sukses (200):
```json
{
   "success": true,
   "employee_id": "EMP001",
   "message": "Employee enrolled successfully",
   "detection_score": 0.98,
   "insert_result": {
      "success": true,
      "insert_count": 1
   }
}
```

Response gagal :
```json
{ "success": false, "error": "No face detected" }
```
atau
```json
{ "success": false, "error": "Multiple faces detected" }
```
atau
```json
{ "success": false, "error": "Failed to decode image bytes" }
```

Response gagal :
```json
{ "success": false, "error": "Collection not found" }
```

Mapping status code (untuk response `success: false`):
- 422: `No face detected`, `Multiple faces detected`
- 400: `Failed to decode image bytes` dan error validasi lain
- 503: `Collection not found` (Milvus/collection belum siap)

### Verify (cek wajah vs database)
`POST /verify`

**Input dari backend Express** (multipart/form-data):
- `image` (file, wajib): file gambar.
- `threshold` (float, opsional):

Response sukses - match ditemukan (200):
```json
{
   "success": true,
   "matched": true,
   "employee_id": "EMP001",
   "similarity": 0.83,
   "threshold": 0.65,
   "detection_score": 0.97
}
```

Response sukses - tidak ada match (200):
```json
{
   "success": true,
   "matched": false,
   "similarity": 0.42,
   "threshold": 0.65,
   "message": "No match found"
}
```

Response gagal:
```json
{ "success": false, "error": "No face detected" }
```
atau
```json
{ "success": false, "error": "Collection not found" }
```

### Delete (hapus data karyawan)
`DELETE /delete/{employee_id}`

**Input dari backend Express**:
- Path param `employee_id` (string, wajib)

Response sukses (200):
```json
{
   "success": true,
   "employee_id": "EMP001",
   "message": "Employee data deleted successfully"
}
```
Response gagal - tidak ditemukan (404):
```json
{ "success": false, "error": "Employee EMP001 not found" }
```

Response gagal - error internal (500):
```json
{ "detail": "..." }
```

### List Employees (ambil daftar employee_id yang tersimpan)
`GET /employees`

**Input dari backend Express**: tidak ada.

Response sukses (200):
```json
{ "success": true, "employee_ids": ["EMP001", "EMP002"] }
```

Response gagal (contoh, HTTP non-200):
```json
{ "success": false, "error": "Collection not found" }
```

### Extract Embedding
`POST /extract/embedding`

Response sukses (200) mengembalikan embedding.

Response gagal (contoh, HTTP non-200):
```json
{ "success": false, "error": "No face detected" }
```
atau
```json
{ "success": false, "error": "Multiple faces detected" }
```
atau
```json
{ "success": false, "error": "Failed to decode image bytes" }
```

### Ringkas daftar endpoint
```
GET    /health
POST   /enroll
POST   /verify
DELETE /delete/{employee_id}
POST   /extract/embedding
GET    /employees
```