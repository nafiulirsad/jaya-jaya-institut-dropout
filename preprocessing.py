"""Modul bersama untuk proyek prediksi dropout Jaya Jaya Institut.

Berisi tiga hal yang dipakai ulang oleh `notebook.ipynb` maupun `app.py`:

1. **Kamus label** (`COURSE_LABELS`, `APPLICATION_MODE_LABELS`, ...) untuk menerjemahkan
   kode numerik pada dataset mentah menjadi teks yang terbaca manusia.
2. **Feature engineering** (`engineer_features`) — turunan rasio kelulusan mata kuliah,
   rata-rata nilai, tren antarsemester, dan indikator risiko finansial.
3. **Definisi fitur produksi** (`CATEGORICAL_FEATURES`, `NUMERIC_FEATURES`) yang menjadi
   kontrak input model sehingga notebook dan aplikasi Streamlit tidak pernah berbeda.

Menyimpan logika ini di satu berkas mencegah *training-serving skew*: transformasi yang
dipakai saat melatih model sama persis dengan yang dipakai saat aplikasi memprediksi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ======================================================================================
# 1. Kamus label
# ======================================================================================

TARGET_COLUMN = "Status"
POSITIVE_CLASS = "Dropout"

COURSE_LABELS = {
    33: "Teknologi Produksi Biofuel",
    171: "Desain Animasi & Multimedia",
    8014: "Pekerjaan Sosial (kelas malam)",
    9003: "Agronomi",
    9070: "Desain Komunikasi",
    9085: "Keperawatan Hewan",
    9119: "Teknik Informatika",
    9130: "Pengelolaan Kuda (Equinculture)",
    9147: "Manajemen",
    9238: "Pekerjaan Sosial",
    9254: "Pariwisata",
    9500: "Keperawatan",
    9556: "Kesehatan Gigi",
    9670: "Manajemen Periklanan & Pemasaran",
    9773: "Jurnalistik & Komunikasi",
    9853: "Pendidikan Dasar",
    9991: "Manajemen (kelas malam)",
}

APPLICATION_MODE_LABELS = {
    1: "Seleksi umum - fase 1",
    2: "Ordinance No. 612/93",
    5: "Kontingen khusus - Kep. Azores",
    7: "Pemegang ijazah pendidikan tinggi lain",
    10: "Ordinance No. 854-B/99",
    15: "Mahasiswa internasional (sarjana)",
    16: "Kontingen khusus - Kep. Madeira",
    17: "Seleksi umum - fase 2",
    18: "Seleksi umum - fase 3",
    26: "Ordinance No. 533-A/99 b2 (beda kurikulum)",
    27: "Ordinance No. 533-A/99 b3 (institusi lain)",
    39: "Jalur usia di atas 23 tahun",
    42: "Transfer",
    43: "Pindah program studi",
    44: "Pemegang diploma spesialisasi teknologi",
    51: "Pindah institusi/program studi",
    53: "Pemegang diploma siklus pendek",
    57: "Pindah institusi/program studi (internasional)",
}

# Pengelompokan jalur masuk agar dashboard tidak memuat 18 kategori tipis.
ADMISSION_PATH_GROUPS = {
    "Seleksi Reguler": [1, 17, 18],
    "Kontingen Khusus (Kepulauan)": [5, 16],
    "Mahasiswa Internasional": [15, 57],
    "Jalur Usia >23 Tahun": [39],
    "Pemegang Ijazah/Diploma Lain": [7, 44, 53],
    "Pindahan / Ganti Program": [42, 43, 51],
    "Jalur Ordinance & Lainnya": [2, 10, 26, 27],
}

MARITAL_STATUS_LABELS = {
    1: "Lajang",
    2: "Menikah",
    3: "Duda/Janda",
    4: "Bercerai",
    5: "Kumpul kebo (facto union)",
    6: "Pisah secara hukum",
}

GENDER_LABELS = {0: "Perempuan", 1: "Laki-laki"}
ATTENDANCE_LABELS = {0: "Kelas Malam", 1: "Kelas Siang"}
YES_NO_LABELS = {0: "Tidak", 1: "Ya"}

# Kolom biner beserta label yang dipakai di dashboard.
BINARY_LABEL_COLUMNS = {
    "Displaced": "Perantau (Displaced)",
    "Educational_special_needs": "Kebutuhan Khusus",
    "Debtor": "Menunggak (Debtor)",
    "Tuition_fees_up_to_date": "UKT Lunas",
    "Scholarship_holder": "Penerima Beasiswa",
    "International": "Mahasiswa Internasional",
}


def _group_admission_path(code: int) -> str:
    for label, codes in ADMISSION_PATH_GROUPS.items():
        if code in codes:
            return label
    return "Jalur Ordinance & Lainnya"


# ======================================================================================
# 2. Feature engineering
# ======================================================================================

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan akademik dan finansial.

    Semua fitur hanya memakai informasi yang sudah tersedia pada akhir tahun pertama,
    sehingga tidak terjadi kebocoran data (data leakage) terhadap status akhir mahasiswa.
    """
    df = frame.copy()

    for sem in ("1st", "2nd"):
        enrolled = df[f"Curricular_units_{sem}_sem_enrolled"]
        approved = df[f"Curricular_units_{sem}_sem_approved"]
        df[f"approval_rate_{sem}"] = np.where(enrolled > 0, approved / enrolled, 0.0)

    total_enrolled = (df["Curricular_units_1st_sem_enrolled"]
                      + df["Curricular_units_2nd_sem_enrolled"])
    total_approved = (df["Curricular_units_1st_sem_approved"]
                      + df["Curricular_units_2nd_sem_approved"])

    df["approval_rate_total"] = np.where(total_enrolled > 0, total_approved / total_enrolled, 0.0)
    df["total_approved"] = total_approved
    df["avg_grade"] = df[["Curricular_units_1st_sem_grade",
                          "Curricular_units_2nd_sem_grade"]].mean(axis=1)
    df["grade_trend"] = (df["Curricular_units_2nd_sem_grade"]
                         - df["Curricular_units_1st_sem_grade"])
    df["approval_trend"] = df["approval_rate_2nd"] - df["approval_rate_1st"]
    df["financial_risk"] = df["Debtor"] + (1 - df["Tuition_fees_up_to_date"])
    df["no_pass_flag"] = (total_approved == 0).astype(int)
    return df


def add_readable_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom berlabel manusia + pengelompokan untuk kebutuhan dashboard."""
    df = frame.copy()
    df["program_studi"] = df["Course"].map(COURSE_LABELS).fillna("Lainnya")
    df["jalur_masuk_detail"] = df["Application_mode"].map(APPLICATION_MODE_LABELS).fillna("Lainnya")
    df["jalur_masuk"] = df["Application_mode"].map(_group_admission_path)
    df["status_pernikahan"] = df["Marital_status"].map(MARITAL_STATUS_LABELS).fillna("Lainnya")
    df["jenis_kelamin"] = df["Gender"].map(GENDER_LABELS)
    df["waktu_kuliah"] = df["Daytime_evening_attendance"].map(ATTENDANCE_LABELS)

    for col in BINARY_LABEL_COLUMNS:
        df[f"{col.lower()}_label"] = df[col].map(YES_NO_LABELS)

    df["kelompok_usia"] = pd.cut(
        df["Age_at_enrollment"], [16, 19, 22, 25, 30, 40, 120],
        labels=["17-19", "20-22", "23-25", "26-30", "31-40", "41+"]).astype(str)
    df["band_nilai_masuk"] = pd.cut(
        df["Admission_grade"], [0, 120, 130, 140, 150, 300],
        labels=["<120", "120-130", "130-140", "140-150", ">150"]).astype(str)
    df["band_kelulusan_sks"] = pd.cut(
        df["approval_rate_total"], [-0.01, 0.001, 0.5, 0.8, 1.01],
        labels=["0% (tidak lulus satu pun)", "1-50%", "51-80%", "81-100%"]).astype(str)
    df["band_nilai_semester"] = pd.cut(
        df["avg_grade"], [-0.01, 0.001, 10, 12, 14, 21],
        labels=["Tanpa nilai", "0-10", "10-12", "12-14", ">14"]).astype(str)
    df["status_keuangan"] = np.select(
        [df["financial_risk"] == 0, df["financial_risk"] == 1],
        ["Aman (lunas & tidak menunggak)", "Perlu perhatian (1 masalah)"],
        default="Bermasalah (menunggak & UKT tertunggak)")
    return df


# ======================================================================================
# 3. Kontrak fitur model produksi
# ======================================================================================

CATEGORICAL_FEATURES = ["Marital_status", "Application_mode", "Course"]

NUMERIC_FEATURES = [
    "Application_order", "Daytime_evening_attendance", "Previous_qualification_grade",
    "Admission_grade", "Displaced", "Debtor", "Tuition_fees_up_to_date", "Gender",
    "Scholarship_holder", "Age_at_enrollment", "International",
    "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade",
    "approval_rate_1st", "approval_rate_2nd", "approval_rate_total", "total_approved",
    "avg_grade", "grade_trend", "approval_trend", "financial_risk", "no_pass_flag",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Kolom mentah yang wajib diisi pengguna aplikasi sebelum feature engineering dijalankan.
RAW_INPUT_COLUMNS = [
    "Marital_status", "Application_mode", "Application_order", "Course",
    "Daytime_evening_attendance", "Previous_qualification_grade", "Admission_grade",
    "Displaced", "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder",
    "Age_at_enrollment", "International",
    "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade",
]

RISK_BANDS = [
    (0.00, 0.20, "Rendah"),
    (0.20, 0.35, "Sedang"),
    (0.35, 0.60, "Tinggi"),
    (0.60, 1.01, "Sangat Tinggi"),
]


def risk_band(probability: float) -> str:
    """Ubah probabilitas dropout menjadi band risiko untuk kebutuhan operasional."""
    for low, high, label in RISK_BANDS:
        if low <= probability < high:
            return label
    return "Sangat Tinggi"


def prepare_for_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Jalankan feature engineering lalu kembalikan kolom sesuai kontrak model."""
    engineered = engineer_features(frame)
    missing = [c for c in MODEL_FEATURES if c not in engineered.columns]
    if missing:
        raise ValueError(f"Kolom berikut belum tersedia untuk model: {missing}")
    return engineered[MODEL_FEATURES]
