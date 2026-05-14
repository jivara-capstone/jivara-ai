# Jivara AI API

REST API untuk analisis interaksi obat-makanan dan rekomendasi makanan aman.
Bagian dari Capstone Project Coding Camp 2026.

**Penanggung jawab:** Hanif Rifan Ash Shidiq (AI Engineer)

---

## Arsitektur Sistem

```
Foto Makanan
     |
     v
[YOLOv11 - Computer Vision]  <-- modul terpisah (tim CV)
     |
     v
  Nama Makanan (misal: "rendang")
     |
     +----> /nutrition
     |          Estimasi kalori, protein, lemak, karbo (sumber: TKPI)
     |
     +----> /interaction-check
     |          ExtraTrees model → severity score 0-5
     |          Gemini LLM → penjelasan bahasa awam (jika risiko tinggi)
     |
     +----> /recommend
                Score 61 makanan terhadap obat pasien
                Urutkan dari severity terendah → rekomendasi aman
```

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Model:** ExtraTrees Regressor (scikit-learn)
- **LLM:** Gemini 2.5 Flash (opsional, untuk penjelasan risiko tinggi)
- **Data:** 854 pasangan makanan-obat, 14 kategori farmakologis, 61 kelas makanan

## API Endpoints

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| `GET` | `/health` | Health check |
| `POST` | `/nutrition` | Estimasi nilai gizi |
| `POST` | `/interaction-check` | Cek interaksi obat-makanan |
| `POST` | `/recommend` | Rekomendasi makanan aman |
| `GET` | `/alerts` | Riwayat notifikasi risiko |
| `DELETE` | `/alerts` | Hapus riwayat alert |

### POST `/interaction-check`

**Request:**
```json
{
  "yolo_class": "tumis-kangkung",
  "patient_medications": ["WARFARIN"]
}
```

**Response:**
```json
{
  "detected_food": "tumis-kangkung",
  "highest_severity": 5.0,
  "status": "warning",
  "detailed_predictions": [
    {
      "medication": "WARFARIN",
      "matched_categories": ["antikoagulan"],
      "severity_score": 5.0,
      "risk_level": "tinggi",
      "risky_categories": ["antikoagulan"],
      "mechanisms": ["Meningkatkan efek pengencer darah / antagonis vitamin K"]
    }
  ],
  "llm_reasoning": "Peringatan: terdeteksi risiko interaksi...",
  "recommended_foods": [
    {"food_name": "apel", "severity_score": 0.0, "risk_level": "aman"}
  ],
  "alert_sent": true
}
```

### POST `/recommend`

**Request:**
```json
{
  "patient_medications": ["METFORMIN", "SIMVASTATIN"],
  "top_n": 5
}
```

**Response:**
```json
{
  "patient_medications": ["METFORMIN", "SIMVASTATIN"],
  "matched_categories": {
    "METFORMIN": ["antidiabetes"],
    "SIMVASTATIN": ["statin"]
  },
  "total_foods_analyzed": 61,
  "summary": {"safe": 25, "avoid": 36},
  "recommended_foods": [
    {"food_name": "ayam-betutu", "severity_score": 0.0, "risk_level": "aman"}
  ],
  "foods_to_avoid": [
    {"food_name": "kunyit-asam", "severity_score": 5.0, "risk_level": "tinggi", "worst_category": "antidiabetes"}
  ]
}
```

### POST `/nutrition`

**Request:**
```json
{
  "yolo_class": "rendang",
  "portion_grams": 150
}
```

**Response:**
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

## Nama Obat yang Didukung

| Kategori | Contoh Obat |
|----------|-------------|
| Antikoagulan | WARFARIN, CLOPIDOGREL, HEPARIN, RIVAROXABAN |
| Antidiabetes | METFORMIN, GLIBENCLAMIDE, INSULIN, ACARBOSE |
| ACE/ARB | CAPTOPRIL, LOSARTAN, VALSARTAN, CANDESARTAN |
| CCB | AMLODIPINE, NIFEDIPINE, DILTIAZEM, VERAPAMIL |
| Statin | SIMVASTATIN, ATORVASTATIN, ROSUVASTATIN |
| Antibiotik Tetrasiklin | DOXYCYCLINE, TETRACYCLINE |
| Antibiotik Fluorokuinolon | CIPROFLOXACIN, LEVOFLOXACIN |
| MAOI | SELEGILINE, MOCLOBEMIDE, LINEZOLID |
| Tiroid | LEVOTHYROXINE |
| NSAID | IBUPROFEN, DICLOFENAC, MELOXICAM, NAPROXEN |
| Antikonvulsan | PHENYTOIN, CARBAMAZEPINE, VALPROIC |
| Glikosida Jantung | DIGOXIN |
| Xantin | THEOPHYLLINE, AMINOPHYLLINE |
| Imunosupresan | CYCLOSPORINE, TACROLIMUS |

## Setup Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Konfigurasi API key Gemini (opsional)
cp .env.example .env

# Jalankan server
python run.py
```

Server berjalan di `http://localhost:8000`. Dokumentasi interaktif di `http://localhost:8000/docs`.

## Deploy ke Railway

1. Push ke GitHub
2. Railway → New Project → Deploy from GitHub
3. Set environment variable `GEMINI_API_KEY` (opsional)
4. Deploy otomatis dari `Dockerfile` atau `Procfile`

## Struktur Folder

```
jivara-ai-api/
├── app/
│   ├── main.py               # FastAPI server
│   ├── model_inference.py     # ExtraTrees inference + Gemini LLM
│   └── nutrition_service.py   # Estimasi gizi dari TKPI
├── data/
│   ├── drug_food_interactions.csv    # Ground truth 854 pasangan
│   ├── food_to_ingredient_kb.json    # 61 makanan + komposisi bahan
│   ├── obat_bpom_cleaned_full.csv    # 23.682 produk obat BPOM
│   └── unified_nutrition.csv         # 1.476 data gizi TKPI
├── models/
│   └── drug_interaction_tree_model.pkl  # ExtraTrees model (92% CV accuracy)
├── notebooks/
│   ├── 01_model_training.ipynb                    # Eksperimen Hybrid NCF
│   └── 02_tree_based_recommender_all_in_one.ipynb # Training ExtraTrees (model utama)
├── scripts/
│   ├── generate_drug_food_interactions.py  # Generate ground truth CSV
│   └── train_tree_interaction_model.py     # Training script
├── Dockerfile
├── Procfile
├── requirements.txt
├── run.py
└── .env.example
```

## Evaluasi Model

Model: **ExtraTrees Regressor** (5-Fold Stratified CV)

| Metrik | Nilai |
|--------|-------|
| MAE (0-5) | 0.164 |
| RMSE (0-5) | 0.633 |
| Risk Accuracy | 92.04% |

Risk category: aman, ringan, sedang, tinggi.
