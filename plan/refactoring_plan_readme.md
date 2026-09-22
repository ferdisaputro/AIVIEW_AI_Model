# Face Embedding Extraction Optimization - Uniform Sampling Strategy

## Deskripsi
Pembaruan (*refactoring*) ini bertujuan untuk mengoptimalkan proses ekstraksi *face embedding* (VGG-Face via DeepFace) pada video. Alih-alih melakukan deteksi wajah pada setiap frame secara keseluruhan (*brute-force*), sistem kini menggunakan metode **Uniform Sampling** dengan menargetkan pengambilan tepat 30 frame representatif. 

Pendekatan ini secara drastis akan mengurangi beban komputasi dan mempercepat waktu pemrosesan (*inference time*), sambil tetap mempertahankan representasi visual yang akurat dari keseluruhan video.

## Logika Utama (Uniform Sampling)
Sistem akan membagi total frame video ke dalam 30 titik sampel dengan interval yang sama.
*   **Target Output:** 30 Frame
*   **Rumus Interval ($k$):** `floor(Total Frame / 30)`
*   **Titik Target Ideal:** Frame ke-`0`, `k`, `2k`, `3k`, ... , `29k`.
*   **Video < 30 frame:** `k = 0` → video ditandai INVALID (`too_short`), tidak ada `.npy` yang dihasilkan.
*   **Deteksi hanya berjalan** pada frame target (plus jendela forward search target 1, lihat bawah). Frame di antara target hanya di-*decode*, tanpa inferensi.

## Skenario Fallback (Penanganan Gagal Deteksi)
Karena model AI mungkin gagal mendeteksi objek pada frame target tertentu (misal karena *motion blur* atau oklusi), sistem dilengkapi dengan dua logika *fallback*:

### 1. Fallback Frame Pertama (Target 1 / Index 0)
Jika model gagal mendeteksi objek pada frame target pertama (Frame 0), sistem akan melakukan **Forward Search** (Pencarian ke depan).
*   **Aksi:** Cek frame secara berurutan: 1, 2, 3, dst.
*   **Batas Pencarian:** Maksimal pencarian adalah frame ke-`(k-1)` (tepat sebelum target sampel ke-2).
*   **Hasil:** Frame pertama yang berhasil terdeteksi dalam rentang tersebut akan diubah statusnya menjadi representasi frame target pertama.
*   **Jika seluruh jendela habis tanpa deteksi:** frame target pertama tetap berstatus *pending* dan akan diisi (*forward fill*) oleh keberhasilan deteksi target pertama berikutnya.

### 2. Fallback Frame Selanjutnya (Target ke-2 s/d 30)
Jika model gagal mendeteksi objek pada frame target selanjutnya, sistem menggunakan logika **Last Known Good / Backward Fill**.
*   **Aksi:** Sistem **TIDAK** melakukan pencarian ke depan maupun memproses frame tambahan.
*   **Hasil:** Sistem langsung menduplikasi hasil deteksi (beserta frame representasinya) dari frame target sebelumnya yang berhasil.

---

## Contoh Simulasi Kasus
*   **Total Video Frame:** 300 Frame
*   **Interval ($k$):** 300 / 30 = 10 Frame
*   **Target Sampel:** Frame 0, Frame 10, Frame 20, ..., Frame 290.

| Target | Frame Ideal | Status Deteksi | Eksekusi Fallback | Frame Representasi Akhir |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Frame 0 | ❌ Gagal | Cari maju: Frame 1 (Gagal) -> Frame 2 (Berhasil) | **Frame 2** |
| **2** | Frame 10 | ❌ Gagal | *Backward fill*: Ambil dari Target 1 | **Frame 2** |
| **3** | Frame 20 | ✅ Berhasil | Tidak perlu *fallback* | **Frame 20** |

*Catatan: Pada kasus di atas, output akhir dari pipeline tetap berupa array berisi 30 data deteksi, sehingga konsistensi data tetap terjaga untuk proses selanjutnya.*

## Catatan Implementasi
*   Pastikan variabel penampung (*array* atau *list*) selalu memiliki panjang (*length*) 30 setelah seluruh proses ekstraksi selesai.
*   Logika *fallback* ini bergantung pada hasil *face detection*/*embedding* (VGG-Face via DeepFace, backend detector `opencv`). Pastikan *threshold* deteksi wajah sudah diatur dengan optimal.
*   Embedding yang dihasilkan sebelum *refactoring* ini memakai sampling berbeda (`linspace`); hapus `output/log.csv` (atau set `force=True` sekali) agar seluruh video diproses ulang dengan sampling baru.