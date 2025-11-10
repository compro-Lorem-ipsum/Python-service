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

Jalankan perintah berikut di PowerShell dari root proyek (`c:\Semester 7\Compro\python-service`):

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

## Mengunggah proyek ke GitHub dengan aman (menangani `.env`)

Langkah-langkah berikut menjelaskan cara membuat repo Git lokal, menambahkan `.gitignore` untuk mengecualikan `.env`, mengganti berkas konfigurasi nyata dengan `.env.example`, dan mendorong ke GitHub.

1. Inisialisasi git (jika belum):

```powershell
git init
```

2. Buat file `.gitignore` (pastikan `.env` ada di dalamnya):

```text
# .gitignore contoh
.venv/
__pycache__/
.env
*.pyc
dist/
build/
```

3. Buat file contoh konfigurasi publik ` .env.example` (isi hanya placeholder, tanpa rahasia):

```text
MILVUS_HOST=localhost
MILVUS_PORT=19530
SIMILARITY_THRESHOLD=0.6
PORT=5000
DEBUG=False
```

4. Pastikan `.env` sudah ada di `.gitignore` dan tidak ter-track. Jika Anda pernah tidak sengaja commit `.env`, hapus dari index git sebelum commit:

```powershell
# Hapus dari index (tetap simpan file lokal)
git rm --cached .env
```

5. Tambahkan file, commit, dan hubungkan ke remote GitHub:

```powershell
git add .
git commit -m "Initial commit: add project files (exclude .env)"

# Buat repo di GitHub (web) lalu hubungkan remote, contoh:
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

6. Jika `.env` pernah ter-commit ke remote sebelumnya dan sudah dipush, Anda harus menghapusnya dari riwayat git (operasi sensitif):

Singkat: hapus file dari index, commit, lalu gunakan alat untuk membersihkan riwayat (contoh singkat, hati-hati):

```powershell
git rm --cached .env
git commit -m "Remove .env from repo"
# Gunakan tool seperti 'git filter-repo' atau BFG untuk menghapus file dari riwayat seluruhnya.
# Setelah membersihkan riwayat, Anda perlu force push (hati-hati — ini akan menimpa riwayat remote):
git push --force origin main
```

Jika ini terdengar rumit atau repo sudah berisi kontributor lain, saya bisa bantu langkah demi langkah berdasarkan situasi Anda.

### Menyimpan rahasia untuk CI / deployment

- Untuk menyimpan variabel environment di GitHub Actions / GitHub repository, gunakan **Settings → Secrets and variables → Actions → New repository secret**.
- Untuk deployment ke server, simpan `.env` langsung di server (jangan di-repo publik).

---

## Troubleshooting singkat

- Jika instalasi paket gagal karena kurangnya build tools, pastikan Microsoft C++ Build Tools terpasang (lihat bagian atas README).
- Jika insightface atau PyTorch butuh GPU, pastikan Anda menginstal versi `torch` dan `onnxruntime` yang cocok dengan CUDA Anda.
- Jika activation script diblokir di PowerShell, gunakan `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` atau jalankan script aktivasi dengan `-ExecutionPolicy Bypass`.

---

Jika mau, saya bisa:
- Membuat file `.env.example` dan `.gitignore` untuk Anda sekarang.
- Menjalankan perintah git lokal untuk menyiapkan repo dan commit awal (beri tahu jika Anda ingin saya jalankan di terminal).

Terakhir: jangan lupa cek kembali `requirements.txt` bila ada paket yang perlu versinya disesuaikan untuk platform Anda.

---

Dokumentasi ini ditulis singkat dan praktis agar Anda cepat menjalankan proyek. Jika ingin versi bahasa Inggris atau format README yang lebih panjang (bagian API endpoints, contoh curl, dll.), bilang saja.
