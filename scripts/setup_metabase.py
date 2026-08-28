"""Membangun business dashboard Jaya Jaya Institut di Metabase secara otomatis.

Script memakai Metabase REST API sehingga dashboard dapat dibangun ulang dari nol
(reproducible) tanpa satu pun klik manual:

1. menyelesaikan setup awal Metabase (membuat akun admin root@mail.com / root123),
2. mendaftarkan database PostgreSQL `jji_analytics` sebagai sumber data,
3. membuat seluruh kartu (question) berbasis SQL native,
4. menyusunnya menjadi satu dashboard bertajuk
   "Student Performance Dashboard - Jaya Jaya Institut".

Prasyarat: `docker compose up -d` sudah dijalankan dan Metabase hidup di localhost:3030.

Cara pakai:
    python scripts/setup_metabase.py
    python scripts/setup_metabase.py --url http://localhost:3030
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ADMIN_EMAIL = "root@mail.com"
ADMIN_PASSWORD = "root123"
ADMIN_FIRST = "Nafiul"
ADMIN_LAST = "Irsad"
SITE_NAME = "Jaya Jaya Institut Analytics"

DB_NAME = "Jaya Jaya Institut (PostgreSQL)"
DB_DETAILS = {
    "host": "postgres",  # nama service pada docker-compose
    "port": 5432,
    "dbname": "jji_analytics",
    "user": "jji_user",
    "password": "jji_password",
    "ssl": False,
    "tunnel-enabled": False,
}

DASHBOARD_NAME = "Student Performance Dashboard - Jaya Jaya Institut"
DASHBOARD_DESC = (
    "Monitoring performa mahasiswa dan deteksi dini dropout Jaya Jaya Institut. "
    "Sumber data: 4.424 mahasiswa (1.421 dropout / 32,12%). "
    "Panel terakhir menampilkan skor risiko hasil model machine learning "
    "beserta daftar prioritas intervensi."
)

RED = "#E4572E"
BLUE = "#2E86AB"
GREY = "#5C6672"
GREEN = "#2E9E5B"
ORANGE = "#F2A541"

AVG_LINE = 32.12  # dropout rate institut (garis pembanding pada setiap grafik)


# --------------------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------------------
class Metabase:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session_id: str | None = None

    def _request(self, method: str, path: str, payload=None, timeout=180):
        url = f"{self.base}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.session_id:
            req.add_header("X-Metabase-Session", self.session_id)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:500]
            raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from None

    def get(self, path):
        return self._request("GET", path)

    def post(self, path, payload=None):
        return self._request("POST", path, payload if payload is not None else {})

    def put(self, path, payload):
        return self._request("PUT", path, payload)

    # ---------------------------------------------------------------- lifecycle
    def wait_until_ready(self, timeout=420):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.get("/api/health").get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError("Metabase tidak siap dalam batas waktu yang ditentukan")

    def authenticate(self):
        props = self.get("/api/session/properties")
        setup_token = props.get("setup-token")
        if setup_token:
            print("  -> menjalankan setup awal (membuat akun admin)")
            res = self.post("/api/setup", {
                "token": setup_token,
                "user": {
                    "first_name": ADMIN_FIRST, "last_name": ADMIN_LAST,
                    "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
                    "password_confirm": ADMIN_PASSWORD, "site_name": SITE_NAME,
                },
                "prefs": {"site_name": SITE_NAME, "site_locale": "en", "allow_tracking": False},
            })
            self.session_id = res["id"] if isinstance(res, dict) else res
        else:
            print("  -> Metabase sudah ter-setup, melakukan login")
            self.session_id = self.post(
                "/api/session", {"username": ADMIN_EMAIL, "password": ADMIN_PASSWORD})["id"]
        print(f"  -> session aktif sebagai {ADMIN_EMAIL}")


# --------------------------------------------------------------------------------------
# Pembentuk definisi kartu
# --------------------------------------------------------------------------------------
def kpi(name, query, description, color=BLUE, suffix=""):
    settings = {"card.description": description, "scalar.field": name}
    if suffix:
        settings["scalar.suffix"] = suffix
    return {"name": name, "sql": query, "display": "scalar",
            "description": description, "settings": settings, "color": color}


def chart(name, query, dimension, metric, description, display="bar",
          color=RED, goal=None, extra=None):
    settings = {
        "graph.dimensions": [dimension],
        "graph.metrics": [metric],
        "graph.show_values": True,
        "graph.x_axis.title_text": dimension,
        "graph.y_axis.title_text": metric,
        "series_settings": {metric: {"color": color}},
        "card.description": description,
    }
    if goal is not None:
        settings.update({"graph.show_goal": True, "graph.goal_value": goal,
                         "graph.goal_label": " "})
    if extra:
        settings.update(extra)
    return {"name": name, "sql": query, "display": display,
            "description": description, "settings": settings}


RATE_TEMPLATE = """SELECT {col} AS "{dim}",
       ROUND(100.0 * SUM(is_dropout) / COUNT(*), 2) AS "Dropout Rate (%)",
       COUNT(*) AS "Jumlah Mahasiswa"
FROM students
WHERE {col} IS NOT NULL{extra}
GROUP BY 1
ORDER BY {order}"""


def rate_card(name, col, dim, description, order="2 DESC", display="bar",
              color=RED, extra_where="", order_expr=None):
    """Kartu dropout rate per kategori, lengkap dengan garis pembanding rata-rata institut."""
    sql = RATE_TEMPLATE.format(col=col, dim=dim, order=order, extra=extra_where)
    if order_expr:
        # kategori berjenjang diurutkan sesuai tingkatannya, bukan sesuai besar nilainya
        sql = sql.rsplit("ORDER BY", 1)[0] + f"ORDER BY {order_expr}"
    return chart(name, sql, dim, "Dropout Rate (%)", description,
                 display=display, color=color, goal=AVG_LINE)


# Urutan tampil untuk kategori berjenjang (ordinal), supaya grafik terbaca sebagai gradien.
ORDER_KELULUSAN = ("CASE band_kelulusan_sks WHEN '0% (tidak lulus satu pun)' THEN 1 "
                   "WHEN '1-50%' THEN 2 WHEN '51-80%' THEN 3 ELSE 4 END")
ORDER_NILAI = ("CASE band_nilai_semester WHEN 'Tanpa nilai' THEN 1 WHEN '0-10' THEN 2 "
               "WHEN '10-12' THEN 3 WHEN '12-14' THEN 4 ELSE 5 END")
ORDER_USIA = ("CASE kelompok_usia WHEN '17-19' THEN 1 WHEN '20-22' THEN 2 WHEN '23-25' THEN 3 "
              "WHEN '26-30' THEN 4 WHEN '31-40' THEN 5 ELSE 6 END")
ORDER_KEUANGAN = ("CASE status_keuangan WHEN 'Aman (lunas & tidak menunggak)' THEN 1 "
                  "WHEN 'Perlu perhatian (1 masalah)' THEN 2 ELSE 3 END")
ORDER_NILAI_MASUK = ("CASE band_nilai_masuk WHEN '<120' THEN 1 WHEN '120-130' THEN 2 "
                     "WHEN '130-140' THEN 3 WHEN '140-150' THEN 4 ELSE 5 END")
ORDER_BAND = ("CASE risk_band WHEN 'Rendah' THEN 1 WHEN 'Sedang' THEN 2 "
              "WHEN 'Tinggi' THEN 3 ELSE 4 END")

CARDS = [
    # -------------------------------------------------------------- Seksi 1: KPI
    kpi("Total Mahasiswa",
        'SELECT COUNT(*) AS "Total Mahasiswa" FROM students',
        "Seluruh mahasiswa pada basis data institut (angkatan yang tercatat).",
        color=BLUE),
    kpi("Mahasiswa Dropout",
        'SELECT SUM(is_dropout) AS "Mahasiswa Dropout" FROM students',
        "Jumlah mahasiswa yang berhenti kuliah sebelum lulus.",
        color=RED),
    kpi("Dropout Rate (%)",
        'SELECT ROUND(100.0 * SUM(is_dropout) / COUNT(*), 2) AS "Dropout Rate (%)" FROM students',
        "Rasio mahasiswa dropout terhadap seluruh mahasiswa. Target manajemen < 25%.",
        color=RED, suffix="%"),
    kpi("Mahasiswa Aktif Perlu Intervensi",
        'SELECT COUNT(*) AS "Mahasiswa Aktif Perlu Intervensi" '
        'FROM students_active WHERE is_flagged = 1',
        "Mahasiswa yang masih kuliah dengan skor risiko di atas ambang operasional 0,30.",
        color=ORANGE),

    # -------------------------------------------------------------- Seksi 2: akademik
    rate_card("Dropout % — Rasio Kelulusan Mata Kuliah", "band_kelulusan_sks",
              "Rasio Mata Kuliah Lulus",
              "Mahasiswa yang tidak lulus satu pun mata kuliah dropout 80,77%.",
              order_expr=ORDER_KELULUSAN),
    rate_card("Dropout % — Rata-rata Nilai Semester", "band_nilai_semester",
              "Rata-rata Nilai Semester",
              "Nilai rata-rata di bawah 10 berkorelasi dengan dropout 86,18%.",
              order_expr=ORDER_NILAI),
    chart("Rata-rata Performa Akademik per Status",
          """SELECT status AS "Status Akhir",
       ROUND(AVG(rasio_kelulusan) * 100, 2) AS "Rasio Kelulusan (%)"
FROM students
GROUP BY 1
ORDER BY 2 DESC""",
          "Status Akhir", "Rasio Kelulusan (%)",
          "Graduate 90%, Enrolled 67%, Dropout 34% — mahasiswa aktif berada di tengah.",
          display="bar", color=BLUE),
    chart("Dropout % — Tren Performa Antarsemester",
          """SELECT CASE WHEN tren_kelulusan < -0.1 THEN 'Memburuk'
            WHEN tren_kelulusan > 0.1 THEN 'Membaik'
            ELSE 'Stabil' END AS "Tren Semester 1 -> 2",
       ROUND(100.0 * SUM(is_dropout) / COUNT(*), 2) AS "Dropout Rate (%)",
       COUNT(*) AS "Jumlah Mahasiswa"
FROM students
GROUP BY 1
ORDER BY 2 DESC""",
          "Tren Semester 1 -> 2", "Dropout Rate (%)",
          "Performa yang memburuk antarsemester menaikkan risiko dropout secara tajam.",
          display="bar", color=RED, goal=AVG_LINE),

    # -------------------------------------------------------------- Seksi 3: keuangan
    rate_card("Dropout % — Status Keuangan", "status_keuangan", "Status Keuangan",
              "Kombinasi tunggakan dan UKT tertunggak menghasilkan dropout tertinggi.",
              order_expr=ORDER_KEUANGAN),
    rate_card("Dropout % — Kelunasan UKT", "ukt_lunas", "UKT Lunas",
              "Faktor tunggal terkuat: UKT belum lunas -> dropout 86,55%.",
              order="2 DESC"),
    rate_card("Dropout % — Tunggakan (Debtor)", "menunggak", "Memiliki Tunggakan",
              "Mahasiswa dengan tunggakan dropout 62,03% vs 28,28%."),
    rate_card("Dropout % — Beasiswa", "penerima_beasiswa", "Penerima Beasiswa",
              "Beasiswa bersifat protektif: dropout 12,19% vs 38,71%.", color=GREEN),

    # -------------------------------------------------------------- Seksi 4: demografi
    rate_card("Dropout % — Kelompok Usia", "kelompok_usia", "Kelompok Usia",
              "Dropout naik konsisten seiring usia saat mendaftar.",
              order_expr=ORDER_USIA),
    rate_card("Dropout % — Jenis Kelamin", "jenis_kelamin", "Jenis Kelamin",
              "Mahasiswa laki-laki dropout 45,05% vs perempuan 25,10%."),
    rate_card("Dropout % — Waktu Kuliah", "waktu_kuliah", "Waktu Perkuliahan",
              "Kelas malam dropout 42,86% vs kelas siang 30,80%."),
    rate_card("Dropout % — Status Pernikahan", "status_pernikahan", "Status Pernikahan",
              "Mahasiswa menikah dropout 47,23% vs lajang 30,21%.",
              extra_where="\n  AND status_pernikahan IN ('Lajang', 'Menikah', 'Bercerai')"),

    # -------------------------------------------------------------- Seksi 5: program & jalur
    # Catatan: dimensi dengan >12 kategori memakai bar chart vertikal, bukan row chart,
    # karena row chart Metabase melipat kategori berlebih menjadi satu batang "Other"
    # yang nilainya dijumlahkan sehingga menyesatkan untuk metrik persentase.
    rate_card("Dropout % — Program Studi", "program_studi", "Program Studi",
              "Selisih antar program studi mencapai 40 poin persentase.",
              order="2 DESC", display="bar"),
    rate_card("Dropout % — Jalur Masuk", "jalur_masuk", "Jalur Masuk",
              "Jalur non-reguler menyumbang risiko jauh lebih besar.",
              order="2 ASC", display="row", color=ORANGE),
    rate_card("Dropout % — Nilai Seleksi Masuk", "band_nilai_masuk", "Band Nilai Masuk",
              "Nilai masuk rendah menaikkan risiko, tetapi jauh lebih lemah dari faktor keuangan.",
              order_expr=ORDER_NILAI_MASUK, color=BLUE),

    # -------------------------------------------------------------- Seksi 6: machine learning
    chart("10 Faktor Paling Berpengaruh (Model ML)",
          """SELECT fitur AS "Fitur",
       ROUND(importance * 1000, 2) AS "Skor Pengaruh (x1000)"
FROM feature_importance
ORDER BY importance DESC
LIMIT 10""",
          "Fitur", "Skor Pengaruh (x1000)",
          "Permutation importance: penurunan ROC-AUC saat nilai fitur diacak.",
          display="row", color=BLUE),
    chart("Sebaran Band Risiko Seluruh Mahasiswa",
          f"""SELECT risk_band AS "Band Risiko",
       COUNT(*) AS "Jumlah Mahasiswa"
FROM students
GROUP BY 1
ORDER BY {ORDER_BAND}""",
          "Band Risiko", "Jumlah Mahasiswa",
          "Distribusi skor risiko model untuk seluruh 4.424 mahasiswa.",
          display="bar", color=ORANGE),
    chart("Validasi Kalibrasi: Dropout Aktual per Band Risiko",
          f"""SELECT risk_band AS "Band Risiko",
       ROUND(100.0 * SUM(is_dropout) / COUNT(*), 2) AS "Dropout Aktual (%)"
FROM students
GROUP BY 1
ORDER BY {ORDER_BAND}""",
          "Band Risiko", "Dropout Aktual (%)",
          "Bukti band risiko terkalibrasi: 4,8% pada band Rendah hingga 84,2% pada Sangat Tinggi.",
          display="bar", color=RED),
    chart("Rata-rata Skor Risiko Mahasiswa Aktif per Program Studi",
          """SELECT program_studi AS "Program Studi",
       ROUND(AVG(risk_score), 2) AS "Rata-rata Skor Risiko"
FROM students_active
GROUP BY 1
HAVING COUNT(*) >= 20
ORDER BY 2 DESC""",
          "Program Studi", "Rata-rata Skor Risiko",
          "Program studi dengan konsentrasi mahasiswa aktif berisiko tertinggi.",
          display="bar", color=GREEN),
    {
        "name": "Daftar Prioritas Intervensi — 15 Mahasiswa Aktif Paling Berisiko",
        "sql": """SELECT student_id AS "ID Mahasiswa",
       program_studi AS "Program Studi",
       jalur_masuk AS "Jalur Masuk",
       kelompok_usia AS "Usia",
       status_keuangan AS "Status Keuangan",
       ROUND(rasio_kelulusan * 100, 1) AS "Rasio Kelulusan (%)",
       rata_rata_nilai AS "Rata-rata Nilai",
       risk_score AS "Skor Risiko",
       risk_band AS "Band Risiko"
FROM students_active
ORDER BY risk_score DESC
LIMIT 15""",
        "display": "table",
        "description": ("15 mahasiswa yang masih aktif kuliah dengan skor risiko tertinggi — "
                        "kandidat utama bimbingan khusus semester berjalan."),
        "settings": {"card.description": "Prioritas tindak lanjut unit bimbingan bulan berjalan."},
    },
]

# Judul seksi (text card)
SECTIONS = [
    "## 1. Ringkasan Kondisi Mahasiswa Jaya Jaya Institut",
    "## 2. Performa Akademik Tahun Pertama\n"
    "*Garis putus-putus pada setiap grafik = dropout rate rata-rata institut (32,12%).*",
    "## 3. Kondisi Keuangan Mahasiswa",
    "## 4. Profil Demografi & Pola Perkuliahan",
    "## 5. Program Studi & Jalur Masuk",
    "## 6. Deteksi Dini Berbasis Machine Learning",
]


def build_layout(card_ids, text_cards):
    """Susun tata letak dashboard pada grid 24 kolom."""
    dashcards = []
    neg_id = -1
    row = 0

    def add_text(idx, height=1):
        nonlocal row, neg_id
        dashcards.append({
            "id": neg_id, "card_id": None, "row": row, "col": 0,
            "size_x": 24, "size_y": height,
            "visualization_settings": {
                "virtual_card": {"name": None, "display": "text",
                                 "visualization_settings": {}, "dataset_query": {},
                                 "archived": False},
                "text": text_cards[idx],
                "dashcard.background": False,
            },
            "parameter_mappings": [],
        })
        neg_id -= 1
        row += height

    def add(name, col, size_x, size_y):
        nonlocal neg_id
        dashcards.append({
            "id": neg_id, "card_id": card_ids[name], "row": row, "col": col,
            "size_x": size_x, "size_y": size_y,
            "visualization_settings": {}, "parameter_mappings": [],
        })
        neg_id -= 1

    # Seksi 1 - KPI
    add_text(0)
    for i, name in enumerate(["Total Mahasiswa", "Mahasiswa Dropout",
                              "Dropout Rate (%)", "Mahasiswa Aktif Perlu Intervensi"]):
        add(name, i * 6, 6, 3)
    row += 3

    # Seksi 2 - akademik
    add_text(1, height=2)
    add("Dropout % — Rasio Kelulusan Mata Kuliah", 0, 12, 6)
    add("Dropout % — Rata-rata Nilai Semester", 12, 12, 6)
    row += 6
    add("Rata-rata Performa Akademik per Status", 0, 12, 6)
    add("Dropout % — Tren Performa Antarsemester", 12, 12, 6)
    row += 6

    # Seksi 3 - keuangan
    add_text(2)
    add("Dropout % — Status Keuangan", 0, 9, 6)
    add("Dropout % — Kelunasan UKT", 9, 5, 6)
    add("Dropout % — Tunggakan (Debtor)", 14, 5, 6)
    add("Dropout % — Beasiswa", 19, 5, 6)
    row += 6

    # Seksi 4 - demografi
    add_text(3)
    add("Dropout % — Kelompok Usia", 0, 9, 6)
    add("Dropout % — Jenis Kelamin", 9, 5, 6)
    add("Dropout % — Waktu Kuliah", 14, 5, 6)
    add("Dropout % — Status Pernikahan", 19, 5, 6)
    row += 6

    # Seksi 5 - program studi & jalur masuk
    add_text(4)
    add("Dropout % — Program Studi", 0, 24, 9)
    row += 9
    add("Dropout % — Jalur Masuk", 0, 13, 7)
    add("Dropout % — Nilai Seleksi Masuk", 13, 11, 7)
    row += 7

    # Seksi 6 - machine learning
    add_text(5)
    add("10 Faktor Paling Berpengaruh (Model ML)", 0, 12, 7)
    add("Sebaran Band Risiko Seluruh Mahasiswa", 12, 6, 7)
    add("Validasi Kalibrasi: Dropout Aktual per Band Risiko", 18, 6, 7)
    row += 7
    add("Rata-rata Skor Risiko Mahasiswa Aktif per Program Studi", 0, 24, 8)
    row += 8
    add("Daftar Prioritas Intervensi — 15 Mahasiswa Aktif Paling Berisiko", 0, 24, 10)
    row += 10

    return dashcards


# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:3030",
                        help="Base URL instance Metabase (default: http://localhost:3030)")
    args = parser.parse_args()

    mb = Metabase(args.url)

    print("[1/5] Menunggu Metabase siap ...")
    mb.wait_until_ready()
    mb.authenticate()

    print("[2/5] Mendaftarkan database PostgreSQL ...")
    databases = mb.get("/api/database")
    items = databases["data"] if isinstance(databases, dict) else databases
    db = next((d for d in items if d["name"] == DB_NAME), None)
    if db is None:
        db = mb.post("/api/database", {
            "engine": "postgres", "name": DB_NAME, "details": DB_DETAILS,
            "is_full_sync": True, "auto_run_queries": True,
        })
        print(f"  -> database dibuat (id={db['id']})")
    else:
        print(f"  -> database sudah ada (id={db['id']})")
    db_id = db["id"]

    for _ in range(60):
        if mb.get(f"/api/database/{db_id}").get("initial_sync_status") == "complete":
            break
        time.sleep(3)
    print("  -> sinkronisasi metadata selesai")

    print("[3/5] Membuat kartu (question) ...")
    existing = {c["name"]: c["id"] for c in mb.get("/api/card") or []}
    card_ids = {}
    for spec in CARDS:
        name = spec["name"]
        payload = {
            "name": name,
            "description": spec.get("description"),
            "display": spec["display"],
            "visualization_settings": spec.get("settings", {}),
            "dataset_query": {
                "type": "native",
                "native": {"query": spec["sql"], "template-tags": {}},
                "database": db_id,
            },
            "type": "question",
        }
        card = (mb.put(f"/api/card/{existing[name]}", payload) if name in existing
                else mb.post("/api/card", payload))
        card_ids[name] = card["id"]
        print(f"  -> {name}")

    print("[4/5] Membuat dashboard ...")
    for d in mb.get("/api/dashboard") or []:
        if d["name"] == DASHBOARD_NAME:
            mb.put(f"/api/dashboard/{d['id']}", {"archived": True})
            print("  -> dashboard lama diarsipkan")
    dashboard = mb.post("/api/dashboard", {
        "name": DASHBOARD_NAME, "description": DASHBOARD_DESC, "parameters": []})
    dash_id = dashboard["id"]

    print("[5/5] Menyusun tata letak kartu ...")
    mb.put(f"/api/dashboard/{dash_id}", {"dashcards": build_layout(card_ids, SECTIONS)})

    print("\nDashboard berhasil dibuat.")
    print(f"  URL      : {args.url.rstrip('/')}/dashboard/{dash_id}")
    print(f"  Email    : {ADMIN_EMAIL}")
    print(f"  Password : {ADMIN_PASSWORD}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"GAGAL: {exc}", file=sys.stderr)
        sys.exit(1)
