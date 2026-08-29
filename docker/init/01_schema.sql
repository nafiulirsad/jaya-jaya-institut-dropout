-- Skema basis data analitik Jaya Jaya Institut.
-- Dijalankan otomatis oleh PostgreSQL saat container pertama kali dibuat.
-- Sumber data: berkas CSV hasil notebook.ipynb yang di-mount pada /seed.

-- =====================================================================================
-- 1. Tabel utama: seluruh mahasiswa beserta skor risiko hasil model
-- =====================================================================================
CREATE TABLE students (
    student_id              TEXT PRIMARY KEY,
    status                  TEXT,           -- Dropout / Enrolled / Graduate
    is_dropout              SMALLINT,       -- 1 = Dropout, 0 = Graduate, NULL untuk Enrolled
    risk_score              NUMERIC(6, 2),  -- probabilitas dropout x 100
    risk_band               TEXT,           -- Rendah / Sedang / Tinggi / Sangat Tinggi
    is_flagged              SMALLINT,       -- 1 bila risiko >= ambang operasional 0,35
    sumber_skor             TEXT,           -- out-of-fold (berlabel) / prediksi (mahasiswa aktif)
    program_studi           TEXT,
    jalur_masuk             TEXT,
    jalur_masuk_detail      TEXT,
    waktu_kuliah            TEXT,
    jenis_kelamin           TEXT,
    status_pernikahan       TEXT,
    kelompok_usia           TEXT,
    usia_saat_daftar        INTEGER,
    band_nilai_masuk        TEXT,
    nilai_masuk             NUMERIC(6, 2),
    status_keuangan         TEXT,
    ukt_lunas               TEXT,
    menunggak               TEXT,
    penerima_beasiswa       TEXT,
    perantau                TEXT,
    mahasiswa_internasional TEXT,
    band_kelulusan_sks      TEXT,
    band_nilai_semester     TEXT,
    rasio_kelulusan         NUMERIC(6, 4),
    rata_rata_nilai         NUMERIC(6, 3),
    tren_kelulusan          NUMERIC(6, 4),
    total_mk_lulus          INTEGER,
    mk_lulus_sem1           INTEGER,
    mk_lulus_sem2           INTEGER,
    nilai_sem1              NUMERIC(6, 3),
    nilai_sem2              NUMERIC(6, 3)
);

COPY students FROM '/seed/students_scored.csv' WITH (FORMAT csv, HEADER true);

-- =====================================================================================
-- 2. Faktor penentu model (permutation importance)
-- =====================================================================================
CREATE TABLE feature_importance (
    fitur       TEXT PRIMARY KEY,
    importance  NUMERIC(10, 6),
    std_dev     NUMERIC(10, 6)
);

COPY feature_importance FROM '/seed/feature_importance.csv' WITH (FORMAT csv, HEADER true);

-- =====================================================================================
-- 3. View bantu
-- =====================================================================================

-- Mahasiswa yang status akhirnya sudah pasti (Dropout / Graduate).
-- Seluruh perhitungan dropout rate memakai view ini, sama seperti pada notebook:
-- mahasiswa Enrolled tidak diikutkan karena hasil akhirnya belum diketahui.
CREATE VIEW students_final AS
SELECT * FROM students WHERE status IN ('Dropout', 'Graduate');

-- Mahasiswa yang masih aktif kuliah -> sasaran prediksi & intervensi semester berjalan.
-- Skor mereka berasal dari prediksi model, bukan dari data historis berlabel.
CREATE VIEW students_active AS
SELECT * FROM students WHERE status = 'Enrolled';

CREATE INDEX idx_students_status ON students (status);
CREATE INDEX idx_students_band ON students (risk_band);
CREATE INDEX idx_students_prodi ON students (program_studi);
