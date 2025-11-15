# Python Service


## syarat

- Python 3.8 atau lebih tinggi
- Microsoft C++ Build Tools (untuk Windows)

### Instal Microsoft C++ Build Tools (Windows)

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
## API Endpoints

### Health Check
```
GET /health
```

### Face Service Endpoints
```
POST /enroll
POST /verify
DELETE /delete/{employee_id}
GET /api/db/collections
GET /employees
```

Detail lengkap API liat di `/docs`.

