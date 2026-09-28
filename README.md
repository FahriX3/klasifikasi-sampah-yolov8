# Smart Trash Alert ML Prototype

Prototipe web aplikasi untuk klasifikasi sampah menggunakan model ML YOLOv8. Model ini akan secara otomatis diunduh dari Hugging Face pada saat dijalankan.

## Fitur
- Deteksi gambar sampah menjadi 12 kelas: Plastik, Kertas, Kardus, Logam, Kaca (Bening, Hijau, Cokelat), Organik/Biologis, Pakaian, Sepatu, Baterai, dan Sampah Campuran.
- UI Responsif dan modern.
- Ditenagai oleh **FastAPI** di backend dan **YOLOv8** untuk Machine Learning.

## Cara Instalasi & Menjalankan

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/FahriX3/klasifikasi-sampah-yolov8.git
   cd klasifikasi-sampah-yolov8
   ```

2. **Buat dan jalankan Virtual Environment (Sangat Disarankan):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # (Untuk Mac/Linux)
   # venv\Scripts\activate    # (Untuk Windows)
   ```

3. **Install dependensi library:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi:**
   ```bash
   python app.py
   ```

5. **Akses di Browser:**
   Buka `http://localhost:8000` atau alamat IP lokal Anda di port 8000 (misal: `http://192.168.x.x:8000`) untuk membuka web melalui HP atau perangkat lain di jaringan yang sama.

## Catatan Model ML
Model akan otomatis di-download dari `kendrickfff/YOLOv8GarbageClassification` (Hugging Face) saat pertama kali dijalankan, dengan ukuran sekitar ~87MB.
