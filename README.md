# Jivara AI API

REST API untuk estimasi nutrisi, analisis risiko interaksi obat-makanan, dan rekomendasi makanan aman.
API ini menjadi service AI terpisah yang dipanggil oleh backend utama.

> Catatan: file model (`.keras`, encoder, dan vocabulary) disediakan terpisah melalui Google Drive:
> https://drive.google.com/drive/folders/1FU4IeaMimdCzl_C2vnrXR96IcY6A2ahL?usp=sharing

## Arsitektur Sistem

```text
Foto makanan
     |
     v
[Computer Vision YOLOv11]  ->  nama makanan
     |
     +--> /nutrition
     |       Estimasi kalori, protein, lemak, dan karbohidrat.
     |
     +--> /interaction-check
     |       TensorFlow risk classifier untuk menghitung risiko
     |       interaksi obat-makanan.
     |
     +--> /recommend
             Menilai daftar makanan terhadap obat pasien,
             lalu mengurutkan makanan dari risiko terendah.
```

Penjelasan naratif untuk pengguna tidak dibuat di service ini. Jika dibutuhkan,
bagian tersebut ditangani oleh backend utama.

## Tech Stack

- Framework API: FastAPI + Uvicorn
- Model ML: TensorFlow/Keras Functional API
- Task model: klasifikasi 3 tingkat risiko
  - Aman
  - Perhatian
  - Bahaya
- Input model: kategori obat + bahan makanan dalam bentuk multi-hot vector
- Output API: severity score 0-5, risk level, rekomendasi makanan, dan alert
- Data: pasangan makanan-obat, kategori farmakologis, kelas makanan, dan data gizi

## API Endpoints

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| `GET` | `/health` | Health check |
| `POST` | `/nutrition` | Estimasi nilai gizi |
| `POST` | `/interaction-check` | Cek risiko interaksi obat-makanan |
| `POST` | `/recommend` | Rekomendasi makanan aman |
| `GET` | `/alerts` | Riwayat notifikasi risiko |
| `DELETE` | `/alerts` | Hapus riwayat alert |

## POST `/interaction-check`

Request:

```json
{
  "yolo_class": "tumis-kangkung",
  "patient_medications": ["WARFARIN"]
}
```

Response:

```json
{
  "detected_food": "tumis-kangkung",
  "highest_severity": 4.3,
  "status": "warning",
  "detailed_predictions": [
    {
      "medication": "WARFARIN",
      "matched_categories": ["antikoagulan"],
      "severity_score": 4.3,
      "risk_level": "tinggi",
      "risky_categories": ["antikoagulan"],
      "mechanisms": [
        "Meningkatkan efek pengencer darah / antagonis vitamin K"
      ]
    }
  ],
  "recommended_foods": [
    {
      "food_name": "apel",
      "severity_score": 0.5,
      "risk_level": "aman"
    }
  ],
  "alert_sent": true
}
```

## POST `/recommend`

Request:

```json
{
  "patient_medications": ["METFORMIN", "SIMVASTATIN"],
  "top_n": 5
}
```

Response:

```json
{
  "patient_medications": ["METFORMIN", "SIMVASTATIN"],
  "matched_categories": {
    "METFORMIN": ["antidiabetes"],
    "SIMVASTATIN": ["statin"]
  },
  "total_foods_analyzed": 61,
  "summary": {
    "safe": 25,
    "avoid": 36
  },
  "recommended_foods": [
    {
      "food_name": "ayam-betutu",
      "severity_score": 0.5,
      "risk_level": "aman"
    }
  ],
  "foods_to_avoid": [
    {
      "food_name": "kunyit-asam",
      "severity_score": 4.5,
      "risk_level": "tinggi",
      "worst_category": "antidiabetes"
    }
  ]
}
```

## POST `/nutrition`

Request:

```json
{
  "yolo_class": "rendang",
  "portion_grams": 150
}
```

Response:

```json
{
  "status": "success",
  "yolo_class": "rendang",
  "matched_food": "Rendang sapi masakan",
  "portion_grams": 150,
  "nutrition_facts": {
    "calories_kcal": 289.5,
    "proteins_g": 33.9,
    "fats_g": 11.85,
    "carbohydrates_g": 11.7
  }
}
```

## Setup Lokal

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

Untuk Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Server berjalan di `http://localhost:8000`.
Dokumentasi interaktif tersedia di `http://localhost:8000/docs`.

## Environment

Tidak ada API key eksternal yang wajib untuk menjalankan service ini.
File `.env.example` disediakan sebagai placeholder jika nanti ada konfigurasi tambahan.

## Model Artifacts

File yang dibutuhkan saat runtime:

```text
models/drug_food_risk_model.keras
models/drug_encoder.pkl
models/ingredient_vocab.json
```

## Deploy

1. Pastikan model artifacts tersedia di folder `models/`.
2. Install dependency dari `requirements.txt`.
3. Jalankan aplikasi dengan `python run.py` atau server ASGI sesuai platform deploy.
4. Pastikan endpoint `/health` mengembalikan status `ok`.

## Struktur Folder

```text
jivara-ai-api/
├── app/
│   ├── main.py
│   ├── model_inference.py
│   └── nutrition_service.py
├── data/
│   ├── drug_food_interactions.csv
│   ├── food_to_ingredient_kb.json
│   ├── obat_bpom_cleaned_full.csv
│   └── unified_nutrition.csv
├── models/
│   ├── drug_food_risk_model.keras
│   ├── drug_encoder.pkl
│   └── ingredient_vocab.json
├── notebooks/
├── scripts/
├── Dockerfile
├── Procfile
├── requirements.txt
├── run.py
└── .env.example
```
