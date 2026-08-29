"""Prototipe sistem deteksi dini dropout — Jaya Jaya Institut.

Aplikasi Streamlit ini membungkus model Random Forest hasil `notebook.ipynb` menjadi alat
yang bisa dipakai staf akademik tanpa menulis kode:

* **Prediksi Individu** — isi profil satu mahasiswa, dapatkan probabilitas dropout, band
  risiko, faktor risiko yang terdeteksi, dan rekomendasi tindakan.
* **Prediksi Massal** — unggah CSV satu angkatan, dapatkan hasil skoring lengkap yang bisa
  diunduh kembali.
* **Monitoring Angkatan** — ringkasan kondisi seluruh mahasiswa beserta daftar prioritas
  intervensi.
* **Tentang Model** — metrik evaluasi dan faktor penentu, agar keputusan model dapat diaudit.

Menjalankan secara lokal:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import preprocessing as pp

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "dropout_model.joblib"
META_PATH = BASE_DIR / "model" / "model_metadata.json"
SCORED_PATH = BASE_DIR / "data" / "students_scored.csv"

DROPOUT = "#E4572E"
NEUTRAL = "#2E86AB"
SAFE = "#2E9E5B"
ACCENT = "#F2A541"

BAND_COLORS = {
    "Rendah": SAFE,
    "Sedang": "#F2C14E",
    "Tinggi": ACCENT,
    "Sangat Tinggi": DROPOUT,
}

# Dropout rate historis per band (hasil validasi silang out-of-fold pada notebook) —
# dipakai agar angka yang ditampilkan aplikasi punya rujukan empiris, bukan sekadar label.
BAND_EVIDENCE = {
    "Rendah": "5,4% mahasiswa pada band ini benar-benar dropout secara historis",
    "Sedang": "12,3% mahasiswa pada band ini benar-benar dropout secara historis",
    "Tinggi": "39,6% mahasiswa pada band ini benar-benar dropout secara historis",
    "Sangat Tinggi": "93,0% mahasiswa pada band ini benar-benar dropout secara historis",
}

st.set_page_config(
    page_title="Deteksi Dini Dropout — Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
      .jji-header {
          background: linear-gradient(120deg, #123B5E 0%, #2E86AB 100%);
          padding: 1.6rem 1.9rem; border-radius: 14px; color: #FFFFFF; margin-bottom: 1.4rem;
      }
      .jji-header h1 {font-size: 1.65rem; margin: 0 0 .35rem 0; font-weight: 700; color: #FFFFFF;}
      .jji-header p {margin: 0; opacity: .92; font-size: .95rem;}
      .jji-card {
          border: 1px solid #E4E7EB; border-radius: 12px; padding: 1.1rem 1.25rem;
          background: #FFFFFF; height: 100%;
      }
      .jji-verdict {border-radius: 14px; padding: 1.4rem 1.6rem; color: #FFFFFF;}
      .jji-verdict h2 {margin: 0; font-size: 2.4rem; font-weight: 700; color: #FFFFFF;}
      .jji-verdict p {margin: .2rem 0 0 0; font-size: 1rem; opacity: .95;}
      .jji-chip {
          display: inline-block; padding: .18rem .7rem; border-radius: 999px;
          font-size: .78rem; font-weight: 600; margin-right: .35rem;
      }
      .jji-risk {background: #FDECE7; color: #B23A15; border: 1px solid #F5C6B5;}
      .jji-safe {background: #E8F5EE; color: #1F6E42; border: 1px solid #BFE3CE;}
      .jji-note {color: #5C6672; font-size: .85rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================================
# Pemuatan artefak
# ======================================================================================
def sidik_berkas(path: Path) -> tuple:
    """Sidik jari berkas (ukuran + waktu ubah) sebagai kunci cache.

    Tanpa ini, cache Streamlit hanya berkunci pada kode fungsinya. Ketika artefak
    diperbarui (model dilatih ulang lalu di-push), aplikasi yang sudah berjalan akan
    tetap memakai model dan metrik lama sampai container-nya benar-benar dimatikan.
    """
    if not path.exists():
        return (str(path), 0, 0)
    stat = path.stat()
    return (str(path), stat.st_size, int(stat.st_mtime))


@st.cache_resource(show_spinner="Memuat model ...")
def load_model(sidik: tuple):
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata(sidik: tuple) -> dict:
    if not META_PATH.exists():
        return {"ambang_operasional": 0.35, "metrik_data_uji": {}}
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_scored(sidik: tuple) -> pd.DataFrame | None:
    if not SCORED_PATH.exists():
        return None
    return pd.read_csv(SCORED_PATH)


model = load_model(sidik_berkas(MODEL_PATH))
meta = load_metadata(sidik_berkas(META_PATH))
AMBANG = float(meta.get("ambang_operasional", 0.35))

if model is None:
    st.error(
        "Berkas `model/dropout_model.joblib` tidak ditemukan. "
        "Jalankan `notebook.ipynb` sampai selesai untuk membuat artefak model."
    )
    st.stop()


def prediksi(frame: pd.DataFrame) -> pd.Series:
    """Jalankan feature engineering + model, kembalikan probabilitas dropout."""
    fitur = pp.prepare_for_model(frame)
    return pd.Series(model.predict_proba(fitur)[:, 1], index=frame.index)


def kartu_band(nilai: float) -> tuple[str, str]:
    band = pp.risk_band(nilai)
    return band, BAND_COLORS[band]


# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
    st.markdown("### 🎓 Jaya Jaya Institut")
    st.caption("Sistem Deteksi Dini Mahasiswa Berisiko Dropout")
    halaman = st.radio(
        "Menu",
        ["Prediksi Individu", "Prediksi Massal (CSV)", "Monitoring Angkatan", "Tentang Model"],
        label_visibility="collapsed",
    )
    st.divider()
    metrik = meta.get("metrik_data_uji", {})
    st.markdown("**Ringkasan model**")
    st.markdown(
        f"""
        - Algoritma: **{meta.get('nama_model', 'Random Forest')}**
        - ROC-AUC (data uji): **{metrik.get('roc_auc', 0):.4f}**
        - Recall dropout: **{metrik.get('recall_ambang_operasional', 0):.1%}**
        - Ambang intervensi: **{AMBANG:.2f}**
        """
    )
    st.caption(
        "Ambang sengaja diturunkan di bawah 0,50 agar mahasiswa berisiko tidak lolos "
        "dari deteksi — biaya melewatkan mahasiswa dropout jauh lebih besar daripada "
        "biaya satu sesi konseling tambahan."
    )


st.markdown(
    """
    <div class="jji-header">
      <h1>Sistem Deteksi Dini Mahasiswa Berisiko Dropout</h1>
      <p>Jaya Jaya Institut · dari mahasiswa yang perjalanan studinya sudah selesai, 39,15%
      berakhir dropout. Aplikasi ini menandai mereka sejak akhir tahun pertama agar bimbingan
      khusus bisa diberikan tepat waktu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================================================================================
# Halaman 1 — Prediksi Individu
# ======================================================================================
if halaman == "Prediksi Individu":
    st.subheader("Profil Mahasiswa")
    st.caption(
        "Isi 20 data berikut — seluruhnya sudah tersedia pada sistem akademik institut. "
        "Nilai bawaan menggambarkan mahasiswa dengan kondisi rata-rata."
    )

    with st.form("form_mahasiswa"):
        tab_akademik, tab_keuangan, tab_pendaftaran, tab_demografi = st.tabs(
            ["📚 Akademik", "💳 Keuangan", "📝 Pendaftaran", "👤 Demografi"]
        )

        with tab_akademik:
            st.markdown("**Semester 1**")
            c1, c2, c3 = st.columns(3)
            sem1_enrolled = c1.number_input("Mata kuliah diambil (sem 1)", 0, 30, 6)
            sem1_approved = c2.number_input("Mata kuliah lulus (sem 1)", 0, 30, 5)
            sem1_grade = c3.number_input("Rata-rata nilai (sem 1, skala 0-20)",
                                         0.0, 20.0, 12.5, step=0.1)
            st.markdown("**Semester 2**")
            c4, c5, c6 = st.columns(3)
            sem2_enrolled = c4.number_input("Mata kuliah diambil (sem 2)", 0, 30, 6)
            sem2_approved = c5.number_input("Mata kuliah lulus (sem 2)", 0, 30, 5)
            sem2_grade = c6.number_input("Rata-rata nilai (sem 2, skala 0-20)",
                                         0.0, 20.0, 12.5, step=0.1)
            st.caption(
                "Rasio mata kuliah yang lulus adalah faktor penentu nomor satu pada model. "
                "Mahasiswa yang lulus di bawah 50% mata kuliah dropout 88-99% secara historis."
            )

        with tab_keuangan:
            c1, c2, c3 = st.columns(3)
            ukt_lunas = c1.selectbox("Pembayaran UKT lunas?", ["Ya", "Tidak"])
            menunggak = c2.selectbox("Memiliki tunggakan (debtor)?", ["Tidak", "Ya"])
            beasiswa = c3.selectbox("Penerima beasiswa?", ["Tidak", "Ya"])
            st.caption(
                "Mahasiswa dengan pembayaran belum lunas dropout 94,03%, sementara penerima "
                "beasiswa hanya 13,83%. Kondisi keuangan adalah pemisah paling tajam di seluruh data."
            )

        with tab_pendaftaran:
            c1, c2 = st.columns(2)
            program = c1.selectbox("Program studi", list(pp.COURSE_LABELS.values()),
                                   index=list(pp.COURSE_LABELS.values()).index("Manajemen"))
            jalur = c2.selectbox("Jalur masuk", list(pp.APPLICATION_MODE_LABELS.values()))
            c3, c4, c5 = st.columns(3)
            urutan_pilihan = c3.number_input("Urutan pilihan program studi (0 = pilihan utama)",
                                             0, 9, 1)
            nilai_masuk = c4.number_input("Nilai seleksi masuk (0-200)", 0.0, 200.0, 127.0, step=0.1)
            nilai_sebelumnya = c5.number_input("Nilai pendidikan sebelumnya (0-200)",
                                               0.0, 200.0, 132.0, step=0.1)
            waktu_kuliah = st.radio("Waktu perkuliahan", ["Kelas Siang", "Kelas Malam"],
                                    horizontal=True)

        with tab_demografi:
            c1, c2, c3 = st.columns(3)
            usia = c1.number_input("Usia saat mendaftar", 16, 75, 20)
            gender = c2.selectbox("Jenis kelamin", ["Perempuan", "Laki-laki"])
            pernikahan = c3.selectbox("Status pernikahan", list(pp.MARITAL_STATUS_LABELS.values()))
            c4, c5 = st.columns(2)
            perantau = c4.selectbox("Perantau (displaced)?", ["Tidak", "Ya"])
            internasional = c5.selectbox("Mahasiswa internasional?", ["Tidak", "Ya"])
            st.caption(
                "Dropout naik seiring usia: 25,23% pada usia 17-19 tahun menjadi 70,25% "
                "pada usia 26-30 tahun."
            )

        submit = st.form_submit_button("🔎 Hitung Risiko Dropout", type="primary",
                                       use_container_width=True)

    if submit:
        kode_program = {v: k for k, v in pp.COURSE_LABELS.items()}[program]
        kode_jalur = {v: k for k, v in pp.APPLICATION_MODE_LABELS.items()}[jalur]
        kode_nikah = {v: k for k, v in pp.MARITAL_STATUS_LABELS.items()}[pernikahan]

        mahasiswa = pd.DataFrame([{
            "Marital_status": kode_nikah,
            "Application_mode": kode_jalur,
            "Application_order": urutan_pilihan,
            "Course": kode_program,
            "Daytime_evening_attendance": 1 if waktu_kuliah == "Kelas Siang" else 0,
            "Previous_qualification_grade": nilai_sebelumnya,
            "Admission_grade": nilai_masuk,
            "Displaced": 1 if perantau == "Ya" else 0,
            "Debtor": 1 if menunggak == "Ya" else 0,
            "Tuition_fees_up_to_date": 1 if ukt_lunas == "Ya" else 0,
            "Gender": 1 if gender == "Laki-laki" else 0,
            "Scholarship_holder": 1 if beasiswa == "Ya" else 0,
            "Age_at_enrollment": usia,
            "International": 1 if internasional == "Ya" else 0,
            "Curricular_units_1st_sem_enrolled": sem1_enrolled,
            "Curricular_units_1st_sem_approved": sem1_approved,
            "Curricular_units_1st_sem_grade": sem1_grade,
            "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
            "Curricular_units_2nd_sem_approved": sem2_approved,
            "Curricular_units_2nd_sem_grade": sem2_grade,
        }])

        peluang = float(prediksi(mahasiswa).iloc[0])
        band, warna = kartu_band(peluang)
        perlu_intervensi = peluang >= AMBANG

        st.divider()
        kiri, kanan = st.columns([1, 1.25])

        with kiri:
            st.markdown(
                f"""
                <div class="jji-verdict" style="background: {warna};">
                  <p>Probabilitas dropout</p>
                  <h2>{peluang:.1%}</h2>
                  <p>Band risiko: <strong>{band}</strong></p>
                  <p style="font-size:.85rem; margin-top:.5rem;">{BAND_EVIDENCE[band]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if perlu_intervensi:
                st.error(f"**Perlu intervensi** — risiko di atas ambang institut ({AMBANG:.0%}).")
            else:
                st.success(f"**Belum perlu intervensi khusus** — risiko di bawah ambang institut "
                           f"({AMBANG:.0%}). Tetap pantau pada evaluasi semester berikutnya.")

        with kanan:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=peluang * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": warna, "thickness": 0.72},
                    "steps": [
                        {"range": [0, 20], "color": "#E8F5EE"},
                        {"range": [20, 35], "color": "#FDF3D8"},
                        {"range": [35, 60], "color": "#FCE7CE"},
                        {"range": [60, 100], "color": "#FBDDD3"},
                    ],
                    "threshold": {"line": {"color": "#123B5E", "width": 3},
                                  "thickness": 0.85, "value": AMBANG * 100},
                },
            ))
            gauge.update_layout(height=270, margin={"t": 20, "b": 10, "l": 30, "r": 30})
            st.plotly_chart(gauge, use_container_width=True)
            st.caption(f"Garis biru = ambang intervensi institut ({AMBANG:.0%}).")

        # ---------------------------------------------------------------- faktor risiko
        rasio = ((sem1_approved + sem2_approved) / (sem1_enrolled + sem2_enrolled)
                 if (sem1_enrolled + sem2_enrolled) > 0 else 0)
        rata_nilai = (sem1_grade + sem2_grade) / 2
        tren = ((sem2_approved / sem2_enrolled) if sem2_enrolled else 0) - \
               ((sem1_approved / sem1_enrolled) if sem1_enrolled else 0)

        pemeriksaan = [
            (ukt_lunas == "Tidak", "Pembayaran belum lunas",
             "94,03% mahasiswa dengan pembayaran tertunggak berakhir dropout"),
            (menunggak == "Ya", "Memiliki tunggakan (debtor)",
             "75,54% mahasiswa debtor berakhir dropout"),
            (beasiswa == "Tidak", "Bukan penerima beasiswa",
             "Penerima beasiswa hanya dropout 13,83% vs 48,37% non-penerima"),
            (rasio < 0.5, f"Rasio kelulusan mata kuliah rendah ({rasio:.0%})",
             "Rasio di bawah 50% berkorelasi dengan dropout 88-99%"),
            (rata_nilai < 10, f"Rata-rata nilai rendah ({rata_nilai:.1f}/20)",
             "Nilai rata-rata di bawah 10 berkorelasi dengan dropout 99,07%"),
            (tren < -0.1, "Performa menurun dari semester 1 ke semester 2",
             "Tren kelulusan negatif termasuk 10 besar penentu model"),
            (usia >= 26, f"Mendaftar pada usia lanjut ({usia} tahun)",
             "Dropout usia 26-30 tahun mencapai 70,25%"),
            (waktu_kuliah == "Kelas Malam", "Mengambil kelas malam",
             "Dropout kelas malam 50,74% vs kelas siang 37,68%"),
        ]
        aktif = [p for p in pemeriksaan if p[0]]

        st.markdown("#### Pemeriksaan faktor risiko")
        st.caption(
            "Daftar berikut membandingkan profil mahasiswa dengan temuan analisis historis "
            "pada 4.424 mahasiswa. Faktor inilah yang perlu ditindaklanjuti pembimbing."
        )
        if aktif:
            for _, judul, bukti in aktif:
                st.markdown(
                    f"<span class='jji-chip jji-risk'>⚠ RISIKO</span> **{judul}** "
                    f"<span class='jji-note'>— {bukti}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span class='jji-chip jji-safe'>✓ AMAN</span> Tidak ada faktor risiko utama "
                "yang terdeteksi pada profil ini.",
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------- rekomendasi
        rekomendasi = []
        if ukt_lunas == "Tidak" or menunggak == "Ya":
            rekomendasi.append(
                "**Rujuk ke unit keuangan dalam 7 hari.** Tawarkan skema cicilan, penundaan "
                "pembayaran, atau beasiswa darurat sebelum masalah menjadi alasan berhenti kuliah."
            )
        if rasio < 0.5 or rata_nilai < 10:
            rekomendasi.append(
                "**Jadwalkan konseling akademik wajib.** Tinjau beban SKS semester berikutnya "
                "dan tawarkan program tutor sebaya untuk mata kuliah yang tidak lulus."
            )
        if tren < -0.1:
            rekomendasi.append(
                "**Telusuri penyebab penurunan performa.** Tren memburuk antar semester biasanya "
                "menandakan masalah non-akademik (pekerjaan, kesehatan, keluarga)."
            )
        if usia >= 26 or waktu_kuliah == "Kelas Malam" or pernikahan != "Lajang":
            rekomendasi.append(
                "**Tawarkan fleksibilitas jadwal.** Mahasiswa non-tradisional membutuhkan kelas "
                "daring/rekaman dan batas SKS yang lebih rendah, bukan sekadar bimbingan akademik."
            )
        if beasiswa == "Tidak" and (ukt_lunas == "Tidak" or menunggak == "Ya"):
            rekomendasi.append(
                "**Prioritaskan pada seleksi beasiswa berikutnya.** Beasiswa terbukti menurunkan "
                "dropout hingga sepertiganya."
            )
        if not rekomendasi:
            rekomendasi.append(
                "**Pertahankan pemantauan rutin.** Evaluasi ulang pada akhir semester berikutnya "
                "atau bila muncul tunggakan pembayaran."
            )

        st.markdown("#### Rekomendasi tindakan")
        for item in rekomendasi:
            st.markdown(f"- {item}")


# ======================================================================================
# Halaman 2 — Prediksi Massal
# ======================================================================================
elif halaman == "Prediksi Massal (CSV)":
    st.subheader("Skoring Satu Angkatan Sekaligus")
    st.caption(
        "Unggah berkas CSV berisi data mahasiswa untuk mendapatkan skor risiko seluruh angkatan. "
        "Berkas harus memuat 20 kolom mentah sesuai templat di bawah."
    )

    contoh = pd.read_csv(BASE_DIR / "data" / "data.csv", sep=";").head(5)[pp.RAW_INPUT_COLUMNS]
    c1, c2 = st.columns([1, 2])
    c1.download_button(
        "⬇️ Unduh templat CSV",
        contoh.to_csv(index=False).encode(),
        file_name="templat_data_mahasiswa.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.caption(
        "Templat berisi 5 baris contoh dari data institut. Kolom `student_id` bersifat opsional "
        "dan akan ikut ditampilkan pada hasil bila disertakan."
    )

    berkas = st.file_uploader("Unggah CSV data mahasiswa", type=["csv"])
    if berkas is not None:
        try:
            isi = pd.read_csv(berkas, sep=None, engine="python")
            hilang = [c for c in pp.RAW_INPUT_COLUMNS if c not in isi.columns]
            if hilang:
                st.error(f"Kolom berikut tidak ditemukan pada berkas: {', '.join(hilang)}")
            else:
                peluang = prediksi(isi)
                hasil = isi.copy()
                hasil["probabilitas_dropout"] = peluang.round(4)
                hasil["risk_score"] = (peluang * 100).round(1)
                hasil["band_risiko"] = [pp.risk_band(v) for v in peluang]
                hasil["perlu_intervensi"] = ["Ya" if v >= AMBANG else "Tidak" for v in peluang]

                ditandai = int((peluang >= AMBANG).sum())
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mahasiswa diproses", f"{len(hasil):,}")
                m2.metric("Perlu intervensi", f"{ditandai:,}",
                          f"{ditandai / len(hasil) * 100:.1f}% dari total", delta_color="off")
                m3.metric("Band Sangat Tinggi",
                          f"{int((hasil['band_risiko'] == 'Sangat Tinggi').sum()):,}")
                m4.metric("Rata-rata risiko", f"{peluang.mean() * 100:.1f}%")

                urut = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
                sebaran = (hasil["band_risiko"].value_counts().reindex(urut, fill_value=0)
                           .reset_index())
                sebaran.columns = ["Band Risiko", "Jumlah Mahasiswa"]
                fig = px.bar(sebaran, x="Band Risiko", y="Jumlah Mahasiswa",
                             color="Band Risiko", color_discrete_map=BAND_COLORS,
                             text="Jumlah Mahasiswa",
                             title="Sebaran Band Risiko pada Berkas yang Diunggah")
                fig.update_layout(showlegend=False, height=340)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Hasil skoring (diurutkan dari risiko tertinggi)")
                tampil = hasil.sort_values("risk_score", ascending=False)
                st.dataframe(tampil, use_container_width=True, height=420)
                st.download_button(
                    "⬇️ Unduh hasil skoring",
                    tampil.to_csv(index=False).encode(),
                    file_name="hasil_skoring_dropout.csv",
                    mime="text/csv",
                )
        except Exception as exc:  # noqa: BLE001 - tampilkan pesan ramah ke pengguna
            st.error(f"Berkas gagal diproses: {exc}")


# ======================================================================================
# Halaman 3 — Monitoring Angkatan
# ======================================================================================
elif halaman == "Monitoring Angkatan":
    scored = load_scored(sidik_berkas(SCORED_PATH))
    if scored is None:
        st.warning("Berkas `data/students_scored.csv` belum tersedia. "
                   "Jalankan `notebook.ipynb` terlebih dahulu.")
        st.stop()

    st.subheader("Kondisi Seluruh Mahasiswa Jaya Jaya Institut")
    total = len(scored)
    # Dropout rate dihitung hanya pada mahasiswa yang status akhirnya sudah pasti,
    # sama seperti pada notebook — mahasiswa Enrolled belum punya hasil akhir.
    final = scored[scored["status"].isin(["Dropout", "Graduate"])]
    dropout = int(final["is_dropout"].sum())
    aktif = scored[scored["status"] == "Enrolled"]
    aktif_ditandai = int((aktif["risk_score"] >= AMBANG * 100).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total mahasiswa", f"{total:,}")
    m2.metric("Dropout tercatat", f"{dropout:,}",
              f"{dropout / len(final) * 100:.2f}% dari {len(final):,} yang sudah selesai",
              delta_color="off")
    m3.metric("Masih aktif kuliah", f"{len(aktif):,}")
    m4.metric("Aktif & perlu intervensi", f"{aktif_ditandai:,}",
              f"{aktif_ditandai / len(aktif) * 100:.1f}% mahasiswa aktif", delta_color="off")
    st.caption(
        "Dropout rate dihitung pada mahasiswa yang status akhirnya sudah pasti (Dropout atau "
        "Graduate). Mahasiswa yang masih aktif kuliah tidak diikutkan karena hasil akhirnya "
        "belum diketahui — mereka justru menjadi sasaran prediksi pada tabel di bawah."
    )

    c1, c2 = st.columns(2)
    with c1:
        per_prodi = (final.groupby("program_studi")
                     .agg(jumlah=("student_id", "size"), dropout_rate=("is_dropout", "mean"))
                     .reset_index())
        per_prodi["dropout_rate"] = (per_prodi["dropout_rate"] * 100).round(2)
        per_prodi = per_prodi[per_prodi["jumlah"] >= 30].sort_values("dropout_rate")
        fig = px.bar(per_prodi, x="dropout_rate", y="program_studi", orientation="h",
                     text="dropout_rate", title="Dropout Rate per Program Studi (%)",
                     color_discrete_sequence=[DROPOUT])
        rata_rata = dropout / len(final) * 100
        fig.add_vline(x=rata_rata, line_dash="dash", line_color="#8A939B",
                      annotation_text=f"rata-rata kohort {rata_rata:.1f}%",
                      annotation_position="top")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=460, yaxis_title="", xaxis_title="Dropout Rate (%)",
                          xaxis_range=[0, per_prodi["dropout_rate"].max() * 1.22])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        nama_pendek = {
            "Aman (lunas & tidak menunggak)": "Aman",
            "Perlu perhatian (1 masalah)": "Perlu perhatian",
            "Bermasalah (menunggak & UKT tertunggak)": "Bermasalah",
        }
        per_keuangan = (final.groupby("status_keuangan")
                        .agg(jumlah=("student_id", "size"), dropout_rate=("is_dropout", "mean"))
                        .reset_index())
        per_keuangan["dropout_rate"] = (per_keuangan["dropout_rate"] * 100).round(2)
        per_keuangan["label"] = per_keuangan["status_keuangan"].map(nama_pendek)
        per_keuangan = per_keuangan.set_index("label").reindex(
            ["Aman", "Perlu perhatian", "Bermasalah"]).reset_index()
        fig2 = px.bar(per_keuangan, x="label", y="dropout_rate", text="dropout_rate",
                      title="Dropout Rate per Status Keuangan (%)",
                      color_discrete_sequence=[ACCENT])
        fig2.update_traces(textposition="outside", cliponaxis=False)
        fig2.update_layout(height=250, xaxis_title="", yaxis_title="Dropout Rate (%)",
                           yaxis_range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

        urut = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
        band_aktif = (aktif["risk_band"].value_counts().reindex(urut, fill_value=0).reset_index())
        band_aktif.columns = ["Band Risiko", "Jumlah"]
        fig3 = px.bar(band_aktif, x="Band Risiko", y="Jumlah", text="Jumlah", color="Band Risiko",
                      color_discrete_map=BAND_COLORS,
                      title="Band Risiko Mahasiswa Aktif (hasil prediksi model)")
        fig3.update_layout(height=220, showlegend=False, xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Daftar prioritas intervensi — mahasiswa aktif dengan risiko tertinggi")
    filter_band = st.multiselect("Saring band risiko", urut, default=["Sangat Tinggi", "Tinggi"])
    prioritas = (aktif[aktif["risk_band"].isin(filter_band)]
                 .sort_values("risk_score", ascending=False)
                 .loc[:, ["student_id", "program_studi", "jalur_masuk", "kelompok_usia",
                          "status_keuangan", "rasio_kelulusan", "rata_rata_nilai",
                          "risk_score", "risk_band"]])
    st.dataframe(prioritas, use_container_width=True, height=380)
    st.download_button(
        "⬇️ Unduh daftar prioritas",
        prioritas.to_csv(index=False).encode(),
        file_name="daftar_prioritas_intervensi.csv",
        mime="text/csv",
    )


# ======================================================================================
# Halaman 4 — Tentang Model
# ======================================================================================
else:
    st.subheader("Tentang Model dan Cara Membacanya")

    metrik = meta.get("metrik_data_uji", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{metrik.get('roc_auc', 0):.4f}")
    m2.metric("PR-AUC", f"{metrik.get('pr_auc', 0):.4f}")
    m3.metric("Recall dropout", f"{metrik.get('recall_ambang_operasional', 0):.1%}")
    m4.metric("Precision", f"{metrik.get('precision_ambang_operasional', 0):.1%}")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(
            f"""
            **Algoritma.** {meta.get('algoritma', 'RandomForestClassifier')} dengan
            `n_estimators={meta.get('hyperparameter', {}).get('n_estimators', '-')}`,
            `max_depth={meta.get('hyperparameter', {}).get('max_depth', '-')}`, dan
            `class_weight=balanced`. Model dipilih dari tiga kandidat (Logistic Regression,
            Random Forest, Hist Gradient Boosting) berdasarkan **ROC-AUC validasi silang 5-fold**,
            lalu disetel dengan `RandomizedSearchCV` (40 kombinasi).

            **Data latih.** {meta.get('jumlah_data_latih', 0):,} mahasiswa yang status akhirnya
            sudah pasti — **1 = Dropout, 0 = Graduate** — dengan proporsi dropout
            {meta.get('proporsi_kelas_positif', 0):.2%}.
            {meta.get('jumlah_mahasiswa_enrolled_dipisah', 0):,} mahasiswa berstatus *Enrolled*
            **sengaja tidak dilibatkan** dalam pelatihan karena hasil akhirnya belum diketahui;
            mereka hanya menjadi sasaran prediksi. ROC-AUC *out-of-fold*:
            **{meta.get('roc_auc_out_of_fold', 0):.4f}**.

            **Ambang {AMBANG:.2f}, bukan 0,50.** {meta.get('alasan_ambang', '')}
            Pada ambang ini model menangkap
            {metrik.get('recall_ambang_operasional', 0):.1%} mahasiswa dropout, dengan konsekuensi
            sekitar 1 dari 5 mahasiswa yang ditandai sebenarnya akan lulus.
            """
        )
        st.info(
            "**Batasan yang perlu diketahui.** Model memprediksi *risiko*, bukan kepastian. "
            "Karena dilatih pada dua kelompok dengan hasil yang sudah pasti dan saling "
            "berlawanan (dropout versus lulus), skornya cenderung terpolarisasi saat diterapkan "
            "pada mahasiswa yang masih kuliah — pakailah sebagai **urutan prioritas**, bukan "
            "vonis. Hasilnya adalah alat bantu menentukan prioritas pendampingan, bukan dasar "
            "untuk mengeluarkan atau memberi sanksi kepada mahasiswa. Model perlu dilatih ulang "
            "setiap akhir tahun akademik dengan data angkatan terbaru."
        )

    with c2:
        st.markdown("**Band risiko dan bukti historisnya**")
        band_df = pd.DataFrame({
            "Band": list(BAND_EVIDENCE),
            "Rentang probabilitas": ["< 20%", "20% – 35%", "35% – 60%", "≥ 60%"],
            "Dropout aktual": ["5,4%", "12,3%", "39,6%", "93,0%"],
        })
        st.dataframe(band_df, use_container_width=True, hide_index=True)
        st.caption(
            "Diukur dengan prediksi *out-of-fold* pada 3.630 mahasiswa berlabel, sehingga "
            "setiap mahasiswa dinilai oleh model yang tidak pernah melihat datanya."
        )

    fi_path = BASE_DIR / "data" / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).head(12).sort_values("importance")
        fig = px.bar(fi, x="importance", y="fitur", orientation="h",
                     title="12 Faktor Paling Berpengaruh (Permutation Importance)",
                     color_discrete_sequence=[NEUTRAL])
        fig.update_layout(height=440, yaxis_title="",
                          xaxis_title="Penurunan ROC-AUC saat fitur diacak")
        st.plotly_chart(fig, use_container_width=True)
