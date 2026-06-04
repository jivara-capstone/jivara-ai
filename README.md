# Jivara AI API

REST API untuk analisis interaksi obat-makanan dan rekomendasi makanan aman.
Bagian dari Capstone Project Coding Camp 2026.


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
     |          TensorFlow risk classifier (3 tier) → severity 0-5
     |          Input: kategori obat + bahan makanan (multi-hot)
     |          Gemini LLM → penjelasan bahasa awam (jika risiko tinggi)
     |
     +----> /recommend
                Score 61 makanan terhadap obat pasien
                Urutkan dari severity terendah → rekomendasi aman
```

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Model ML:** TensorFlow/Keras Functional API (Deep Learning)
  - Klasifikasi 3 tingkat risiko (Aman / Perhatian / Bahaya)
  - Input: kategori obat (embedding) + bahan makanan (multi-hot)
  - Custom training loop (`tf.GradientTape`) + weighted cross-entropy
  - Output tier dikonversi ke severity 0-5 agar kompatibel dengan API
  - Accuracy 85.58%, dilatih pada 829 sampel
- **LLM:** Gemini 2.5 Flash (opsional, untuk penjelasan risiko tinggi)
- **Data:** 1,037 pasangan makanan-obat, 17 kategori farmakologis, 61 kelas makanan

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
  "highest_severity": 4.3,
  "status": "warning",
  "detailed_predictions": [
    {
      "medication": "WARFARIN",
      "matched_categories": ["antikoagulan"],
      "severity_score": 4.3,
      "risk_level": "tinggi",
      "risky_categories": ["antikoagulan"],
      "mechanisms": ["Meningkatkan efek pengencer darah / antagonis vitamin K"]
    }
  ],
  "llm_reasoning": "Peringatan: terdeteksi risiko interaksi...",
  "recommended_foods": [
    {"food_name": "apel", "severity_score": 0.5, "risk_level": "aman"}
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
    {"food_name": "ayam-betutu", "severity_score": 0.5, "risk_level": "aman"}
  ],
  "foods_to_avoid": [
    {"food_name": "kunyit-asam", "severity_score": 4.5, "risk_level": "tinggi", "worst_category": "antidiabetes"}
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

# Create Python 3.11 virtual environment (required for TensorFlow)
python3.11 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# Install packages
pip install -r requirements.txt

# Konfigurasi API key Gemini (opsional)
cp .env.example .env

# Jalankan server
python run.py
```

Server berjalan di `http://localhost:8000`. Dokumentasi interaktif di `http://localhost:8000/docs`.

### Training Model (untuk retrain)

Training dilakukan via notebook `notebooks/train_tf_reasoning_classification.ipynb`
(Google Colab, T4 GPU). Output:

```
models/drug_food_risk_model.keras   # model klasifikasi
models/drug_encoder.pkl             # encoder kategori obat
models/ingredient_vocab.json        # vocab bahan (urutan multi-hot)
```

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
│   ├── model_inference.py     # TensorFlow inference + Gemini LLM (UPDATED)
│   └── nutrition_service.py   # Estimasi gizi dari TKPI
├── data/
│   ├── drug_food_interactions.csv    # Ground truth 1,037 pasangan (UPDATED)
│   ├── food_to_ingredient_kb.json    # 61 makanan + komposisi bahan
│   ├── obat_bpom_cleaned_full.csv    # 23.682 produk obat BPOM
│   └── unified_nutrition.csv         # 1.476 data gizi TKPI
├── models/
│   ├── drug_food_risk_model.keras       # TF risk classifier 3-tier
│   ├── drug_encoder.pkl                 # LabelEncoder kategori obat
│   └── ingredient_vocab.json            # urutan bahan untuk multi-hot
├── notebooks/
│   └── train_tf_reasoning_classification.ipynb    # Drug-food risk classifier 3-kelas (GradientTape)
├── scripts/
│   └── generate_drug_food_interactions.py    # Generate ground truth CSV
├── Dockerfile
├── Procfile
├── requirements.txt
├── run.py
└── .env.example
```

## Evaluasi Model

### Drug-Food Risk Classifier — 3-Class (TensorFlow, June 2026)

Notebook: `notebooks/train_tf_reasoning_classification.ipynb`

Model klasifikasi tingkat risiko interaksi obat-makanan. Severity 0-5 dikelompokkan
menjadi 3 tingkat untuk mengatasi distribusi data yang tidak seimbang
(severity 0 mendominasi ~68% data).

**Arsitektur:**
- Input: kategori obat (Embedding 32) + bahan makanan (multi-hot)
- Dense 256 → 128 → 64 → Softmax(3)
- Custom training loop dengan `tf.GradientTape`
- Custom loss: weighted categorical cross-entropy
- Early stopping (restore best weights), logging TensorBoard

**Tingkat risiko:**
- Aman (severity 0-1)
- Perhatian (severity 2-3)
- Bahaya (severity 4-5)

**Hasil (validation set, stratified 80/20):**

| Metrik | Nilai |
|--------|-------|
| Accuracy | 85.58% |
| MAE (risk tier, skala 0-2) | 0.2389 |
| MAE (normalized 0-1) | 0.1195 |
| Baseline majority | 68.27% |
| Training samples | 829 |
| Validation samples | 208 |

**Per-class (validation):**

| Tingkat | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Aman | 0.91 | 0.90 | 0.90 | 145 |
| Perhatian | 0.70 | 0.75 | 0.72 | 40 |
| Bahaya | 0.82 | 0.78 | 0.80 | 23 |
| **Macro avg** | 0.81 | 0.81 | 0.81 | 208 |
| **Weighted avg** | 0.86 | 0.86 | 0.86 | 208 |

Catatan: model bekerja di tingkat **bahan**, sehingga dapat menilai makanan di luar
61 makanan dataset selama komposisi bahannya diketahui (pada uji food-level split,
accuracy ~83% untuk makanan yang tidak pernah dilihat saat training).
