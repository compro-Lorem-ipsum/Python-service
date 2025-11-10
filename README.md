## Face Recognition API

> API Flask untuk pendaftaran dan verifikasi wajah menggunakan ArcFace (insightface) dan penyimpanan vektor di Milvus.

---

## Ringkasan

Proyek ini menjalankan layanan REST sederhana untuk:
- Enroll (mendaftarkan) wajah karyawan
- Verifikasi wajah untuk absensi
- Ekstraksi embedding wajah untuk testing

Kode utama:
- `app.py` — Flask API
- `face_service.py` — pemanggilan model ArcFace (insightface) untuk ekstraksi embedding
- `milvus_db.py` — wrapper sederhana untuk menyimpan dan mencari embedding di Milvus

---

## Prasyarat

1. Python 3.10+ terpasang.
2. Microsoft C++ Build Tools — diperlukan untuk membangun beberapa dependensi (mis. paket yang memerlukan kompilasi native seperti beberapa bagian dari PyTorch / paket wheel terkait). Ikuti panduan instalasi di bawah.
3. Milvus berjalan (opsional di lingkungan pengembangan lokal) pada `MILVUS_HOST` dan `MILVUS_PORT` yang dikonfigurasi di `.env`.

### Instal Microsoft C++ Build Tools (Windows)

1. Buka halaman Download Visual Studio: klik "Download Build Tools".
2. Jalankan installer yang diunduh.
3. Di jendela installer, centang opsi:

   ✅ "Desktop development with C++"

4. Klik "Install" dan tunggu proses selesai. Setelah selesai, restart komputer jika diminta.

Catatan: opsi ini menginstal komponen compiler C++ (MSVC), toolchain, dan library header yang sering dibutuhkan saat pip mencoba membangun paket dari source.

---

## Setup lingkungan (direkomendasikan)

Jalankan perintah berikut di PowerShell dari root proyek :

```powershell
# Buat virtual environment bernama .venv (hanya jika belum dibuat)
python -m venv .venv

# Aktifkan venv di PowerShell
. .\.venv\Scripts\Activate.ps1
# atau: & .\.venv\Scripts\Activate.ps1

# Upgrade pip dan install requirements
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Jika PowerShell menolak menjalankan skrip aktivasi karena ExecutionPolicy, jalankan sekali saja (PowerShell akan mengeksekusi installer aktivasi dalam session baru):

```powershell
powershell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1
# atau ubah policy untuk user saat ini (gunakan dengan hati-hati):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Atau, tanpa mengaktifkan venv, panggil interpreter langsung:

```powershell
.venv\Scripts\python .\app.py
.venv\Scripts\python -m pip install <paket>
```

---

## Menjalankan server

Setelah venv aktif dan dependensi terpasang:

```powershell
python app.py
```

Server akan berjalan di `http://0.0.0.0:5000` (atau port di `.env` jika Anda ubah).

---

## File konfigurasi lingkungan

File `.env` di root berisi konfigurasi (contoh isi di repository Anda):

```
# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Face Recognition Configuration
SIMILARITY_THRESHOLD=0.6

# Flask Configuration
PORT=5000
DEBUG=False
```

JANGAN mengunggah file `.env` yang sebenarnya ke repository publik karena berpotensi berisi kredensial.

---


