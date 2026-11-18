# Prediksi Suhu Kritis Superkonduktor (Tc) dengan Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/NAMA-REPO/blob/main/prediksi_tc_superkonduktor_ml.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Task](https://img.shields.io/badge/Task-Regression-green)

Proyek ini memprediksi **suhu kritis (Tc)** material superkonduktor dari **81 deskriptor numerik** menggunakan beberapa model regresi *machine learning*. Lima model dibandingkan dengan metrik MAE, RMSE, dan R², lalu fitur yang paling informatif dianalisis.

> Proyek ini dikerjakan mengacu pada materi tutorial *"Challenges in Using AI with Large Scale Scientific Computing"* (Mohamed Wahib, 2026 ACM Europe Summer School), pada topik **Computational Physics: Predict Superconducting Critical Temperature**.

---

## Daftar Isi

1. [Latar Belakang](#latar-belakang)
2. [Dataset](#dataset)
3. [Alur Kerja](#alur-kerja)
4. [Model dan Metrik](#model-dan-metrik)
5. [Hasil](#hasil)
6. [Cara Menjalankan](#cara-menjalankan)
7. [Struktur Repositori](#struktur-repositori)
8. [Keterbatasan](#keterbatasan)
9. [Pengembangan Selanjutnya](#pengembangan-selanjutnya)
10. [Anggota Kelompok](#anggota-kelompok)
11. [Sumber dan Kredit](#sumber-dan-kredit)

---

## Latar Belakang

Superkonduktor adalah material yang menghantarkan listrik **tanpa hambatan** di bawah suhu tertentu, yaitu **suhu kritis (Tc)**. Kebanyakan superkonduktor baru bekerja pada suhu yang sangat rendah sehingga belum praktis. Material dengan Tc tinggi lebih mudah dimanfaatkan; misalnya Tc di atas 77 K (titik didih nitrogen cair) bisa didinginkan dengan nitrogen cair yang jauh lebih murah dibandingkan helium cair.

Proyek ini memakai *machine learning* untuk memprediksi Tc dari deskriptor material, sebagai langkah awal membangun *pipeline* penyaringan (*screening*) kandidat superkonduktor.

**Jenis masalah:** regresi (output berupa angka kontinu dalam Kelvin), bukan klasifikasi.

## Dataset

| Item | Keterangan |
|---|---|
| Sumber | [Superconductor Dataset (Kaggle, munumbutt)](https://www.kaggle.com/datasets/munumbutt/superconductor-dataset), yang bersumber dari data superkonduktivitas UCI |
| Jumlah data | 21.263 material |
| Fitur (input) | 81 fitur numerik |
| Target (output) | `critical_temp` (Kelvin) |

Fitur berasal dari delapan sifat atom (massa atom, energi ionisasi pertama, jari-jari atom, densitas, afinitas elektron, panas peleburan, konduktivitas termal, valensi) yang masing-masing diringkas dengan 10 statistik (rata-rata, rentang, simpangan baku, entropi, dan lainnya), ditambah jumlah unsur (8 × 10 + 1 = 81). Fitur ini adalah deskriptor hasil rekayasa fitur, bukan pengukuran eksperimen langsung.

## Alur Kerja

```
Muat data  ->  Bersihkan  ->  Pisahkan X dan y  ->  Bagi 80/20
     ->  Latih 5 model  ->  Evaluasi (MAE, RMSE, R², CV)
     ->  Analisis (feature importance, korelasi, PCA)  ->  Demo input/output
```

1. **Muat data**: mencari `train.csv` secara otomatis (Kaggle, Colab, atau folder lokal) dan mengunduh dari UCI bila tidak ditemukan.
2. **Pembersihan**: semua kolom dipaksa menjadi numerik, baris tidak valid dibuang.
3. **Pembagian data**: 80% data latih dan 20% data uji (`random_state=42`).
4. **Pelatihan dan evaluasi**: lima model dilatih, diuji pada data uji, dan dicek dengan *cross-validation* 3-lipat pada data latih.
5. **Analisis**: pentingnya fitur (Random Forest), korelasi Pearson dengan Tc, distribusi target, dan visualisasi PCA.
6. **Demo input/output**: prediksi pada sampel uji, input manual, analisis "bagaimana jika", peringkat kandidat, serta simpan dan muat model.

## Model dan Metrik

**Model yang dibandingkan**

| Model | Keterangan |
|---|---|
| Linear Regression | Pembanding dasar (*baseline*) |
| Decision Tree | Satu pohon keputusan |
| Random Forest | Gabungan banyak pohon (100 pohon) |
| Gradient Boosting (Hist) | *Boosting* berbasis histogram |
| XGBoost | *Gradient boosting* yang efisien (dilewati otomatis bila tidak terpasang) |

**Metrik evaluasi**

- **MAE**: rata-rata selisih mutlak prediksi dan nilai asli (Kelvin). Makin kecil makin baik.
- **RMSE**: akar rata-rata kuadrat selisih; lebih menghukum kesalahan besar.
- **R²**: proporsi variasi Tc yang dijelaskan model (maksimum 1).
- **CV_RMSE / CV_STD**: rata-rata dan sebaran RMSE dari *cross-validation* 3-lipat.

## Hasil

> **TODO:** isi tabel berikut dengan angka dari hasil eksekusi notebook Anda sendiri. Jangan memakai angka dari sumber lain.

| Model | MAE (K) | RMSE (K) | R² | CV_RMSE |
|---|---|---|---|---|
| Linear Regression | 13.21 | 17.38 | 0.738  | 17.70 |
| Decision Tree | 6.06 | 11.80 | 0.879 | 13.38 |
| Random Forest | 5.09 | 8.98 | 0.930 | 9.91 |
| Gradient Boosting (Hist) | 5.53 | 9.21 | 0.926 | 10.29 |
| XGBoost | 5.83 | 9.31 | 0.925 | 10.08 |

**Model terbaik:** _Random Forest_  
**Fitur terpenting menurut Random Forest:** _range_ThermalConductivity_

## Cara Menjalankan

### Opsi 1: Google Colab (disarankan)

1. Klik tombol **Open In Colab** di bagian atas README ini, atau unggah `prediksi_tc_superkonduktor_ml.ipynb` ke [Google Colab](https://colab.research.google.com).
2. Klik **Connect**, lalu pilih **Runtime > Run all** (atau jalankan sel satu per satu dengan `Shift+Enter` dari atas ke bawah).
3. Dataset dicari atau diunduh otomatis. Jika gagal, unduh `train.csv` dari Kaggle atau UCI dan unggah ke Colab (`/content/train.csv`).

Perkiraan waktu jalan: sekitar 2 sampai 5 menit untuk data penuh. Untuk lebih cepat, ubah `JALANKAN_CV = False` di sel pengaturan.

### Opsi 2: Lokal

```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO
pip install numpy pandas matplotlib scikit-learn xgboost joblib jupyter
jupyter notebook prediksi_tc_superkonduktor_ml.ipynb
```

> **Penting:** pastikan `GUNAKAN_DATA_SINTETIS = False` di sel pengaturan. Jika `True`, notebook memakai data tiruan hanya untuk menguji kode dan hasilnya **bukan** hasil sains.

### Contoh input dan output

```python
INPUT_SAYA = {
    "number_of_elements": 4,
    "range_ThermalConductivity": 350.0,
    "wtd_std_ThermalConductivity": 100.0,
    "mean_Valence": 3.5,
}
tc = prediksi_tc(INPUT_SAYA)   # output: prediksi Tc dalam Kelvin
```

Fitur yang tidak diisi otomatis memakai median data latih.

## Struktur Repositori

```
.
├── prediksi_tc_superkonduktor_ml.ipynb   # notebook utama
├── README.md                             # dokumen ini
└── (opsional) train.csv                  # dataset, jika disertakan
```

## Keterbatasan

- Fitur adalah deskriptor hasil rekayasa, bukan pengukuran fisika langsung.
- Evaluasi memakai satu pembagian latih-uji acak. Banyak material di dataset saling mirip, sehingga ada risiko kebocoran data antar-sampel dan skor bisa tampak lebih baik daripada kinerja pada material yang benar-benar baru.
- Pentingnya fitur dan korelasi tidak membuktikan hubungan sebab-akibat fisika.
- PCA hanya alat visualisasi.
- Demo input manual memakai median untuk fitur yang tidak diisi, sehingga kombinasi fitur belum tentu realistis.
- Model tidak memberi estimasi ketidakpastian pada tiap prediksi.

Hasil ini sebaiknya dianggap sebagai **bukti konsep (*proof of concept*)**, bukan kesimpulan ilmiah akhir.

## Pengembangan Selanjutnya

- Pembagian data berbasis klaster atau kelompok material (*group split*) agar evaluasi lebih jujur.
- Penyetelan *hyperparameter* secara sistematis.
- Model MLP dengan PyTorch sebagai pembanding.
- Estimasi ketidakpastian pada prediksi.

## Anggota Kelompok

| No | Nama | NIM |
|---|---|---|
| 1 | Athaya Ashalina M. | G1A025008 |
| 2 | Kaysa Naura | G1A025016 |
| 3 | Keysa Angel Anandita | G1A025024 |
| 4 | Dini Kayla Putri R | G1A025030 |
| 5 | Rafi Andika | G1A025048 |
| 6 | M Farrel Audrico | G1A025054 |
| 7 | Khairunnisa Lizal N | G1A025060 |
| 8 | Alysa Kirana | G1A025066 |
| 9 | Aurel Clara | G1A025070 |
| 10 | Adellia Fakhira | G1A025078 |
| 11 | M Rafif Al-falah | G1A025116 |

## Sumber dan Kredit

- **Dataset:** [Superconductor Dataset, Kaggle (munumbutt)](https://www.kaggle.com/datasets/munumbutt/superconductor-dataset), bersumber dari data superkonduktivitas UCI.
- **Notebook acuan:** [Superconductor Tc Prediction with ML (israfil123, Kaggle)](https://www.kaggle.com/code/israfil123/superconductor-tc-prediction-with-ml). Notebook dalam repositori ini merupakan adaptasi yang kami modifikasi (perbaikan path data agar berjalan di Colab, percepatan model, penyatuan fungsi pelatihan, dan penambahan demo input/output).
- **Materi tugas:** Mohamed Wahib, *Challenges in Using AI with Large Scale Scientific Computing (Tutorial)*, 2026 ACM Europe Summer School.
- **Penggunaan LLM:** _isi dengan jujur bagian mana yang dibantu LLM dan bagian mana yang diputuskan sendiri oleh tim._
- **Library:** NumPy, pandas, Matplotlib, scikit-learn, XGBoost, joblib.
