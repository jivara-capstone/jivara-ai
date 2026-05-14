# Jivara AI — Drug-Food Interaction Risk Scoring & Nutrition Analysis

Modul AI dari sistem Jivara yang bertanggung jawab untuk menganalisis potensi interaksi obat-makanan dan mengestimasi nilai gizi. Dibangun sebagai bagian dari Capstone Project Coding Camp 2026.

**Penanggung jawab:** Hanif Rifan Ash Shidiq (AI Agent Reasoning & Nutrition Analysis)

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
     +----> /nutrition -----> mapping_image_nutrition.csv
     |                             |
     |                        unified_nutrition.csv (TKPI)
     |                             |
     |                        Estimasi kalori, protein, lemak, karbo
     |
     +----> /interaction-check
     |          |
     |          +---> Hybrid NCF Model (risk scoring)
     |          |         Input: food_id + drug_id + nutrition_features[4]
     |          |         Output: severity score 0-1
     |          |
     |          +---> Knowledge Base (rule-based lookup)
     |          |         Detail mekanisme, tipe interaksi
     |          |
     |          +---> Gemini LLM (jika risiko tinggi)
     |                    Penjelasan bahasa awam untuk pasien
     |
     +----> /recommend
                |
                +---> Score semua 35 makanan terhadap obat pasien
                +---> Urutkan dari risiko terendah (paling aman)
                +---> Return daftar rekomendasi + makanan yang harus dihindari
```

## Komponen AI

### 1. Hybrid NCF (Neural Collaborative Filtering) Risk Scorer

Model prediksi skor risiko interaksi obat-makanan menggunakan pendekatan Hybrid NCF — menggabungkan sinyal collaborative filtering (embedding interaksi) dengan content-based features (nutrisi makanan).

**Arsitektur (TensorFlow Functional API, 3 input):**
```
input_food_id ──┐
                ├── InteractionEmbeddingLayer(embed_dim=32) ── Flatten ──┐
input_drug_id ──┘                                                       │
                                                                        ├── Concatenate ── Dense(128) ── Dense(64) ── Dense(32) ── sigmoid
input_nutrition[4] ── Dense(16, relu) "nutrition_encoder" ──────────────┘
```

- **Collaborative signal:** Embedding makanan & obat (dim=32) digabung via element-wise multiply
- **Content signal:** 4 fitur nutrisi (kalori, protein, lemak, karbo) dinormalisasi MinMaxScaler, di-encode jadi 16-dim
- **Output:** skor severity ternormalisasi (0-1, dikali 5 untuk skala asli 0-5)

**Custom Components:**
| Komponen | Nama | Fungsi |
|----------|------|--------|
| Custom Layer | `InteractionEmbeddingLayer` | Embedding makanan & obat + interaction fusion |
| Custom Loss | `MedicalAsymmetricLoss` | Penalti 1.3x untuk under-prediction (keamanan medis) |
| Custom Callback | `RiskThresholdMonitor` | Early stopping saat val_mae <= target |

**Validasi:** Stratified 5-Fold Cross Validation (karena dataset kecil, ~67 pasangan interaksi).

**Format ekspor:** `.keras` + `nutrition_scaler.pkl` (MinMaxScaler untuk normalisasi fitur nutrisi)

### 2. Nutrition Estimation Pipeline

Menghitung estimasi nilai gizi berdasarkan hasil deteksi makanan:
1. Nama kelas YOLO dipetakan ke key database via `mapping_image_nutrition.csv`
2. Nilai gizi per 100g diambil dari `unified_nutrition.csv` (sumber: TKPI)
3. Disesuaikan dengan berat porsi yang diminta

### 3. LLM-Based Reasoning (Gemini)

Jika model Deep Learning mendeteksi risiko tinggi (severity >= 3.0), sistem memanggil Gemini 2.5 Flash untuk memberikan penjelasan dalam bahasa Indonesia yang mudah dipahami pasien awam, termasuk alasan risiko dan saran alternatif.

### 4. Food Recommendation System

Sistem rekomendasi makanan aman berdasarkan obat pasien:
1. Semua 35 makanan Indonesia di-score terhadap seluruh obat pasien menggunakan model Hybrid NCF
2. Diambil skor risiko tertinggi per makanan (worst-case scenario)
3. Makanan diurutkan dari skor terendah dan dikategorikan: Low Risk, Moderate, High Risk
4. Dikembalikan top-N rekomendasi beserta informasi nutrisi, serta daftar makanan yang perlu dihindari

### 5. Warning & Notification System

Setiap interaksi berisiko tinggi otomatis dicatat ke sistem alert. Riwayat alert bisa diakses via endpoint `/alerts` untuk monitoring oleh perawat atau dokter.

## Dataset

| File | Isi | Dipakai untuk |
|------|-----|---------------|
| `indonesian_food_drug_interactions.json` | 35 makanan Indonesia + interaksi obat + severity | Training model DL, knowledge base |
| `Drug to Food interactions Dataset.json` | 1400+ obat dari DrugBank | Referensi pendukung |
| `drug_food_kb_final.json` | Knowledge base (format dict) | Rule-based lookup saat inference |
| `mapping_image_nutrition.csv` | Mapping nama YOLO → nama di DB gizi | Pipeline estimasi gizi |
| `unified_nutrition.csv` | 1468 bahan makanan (TKPI) | Database nilai gizi |

## API Endpoints

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| GET | `/` | Root, link ke dokumentasi |
| GET | `/health` | Status server |
| POST | `/detect` | Deteksi makanan dari foto (integrasi YOLO) |
| POST | `/nutrition` | Estimasi nilai gizi |
| POST | `/interaction-check` | Analisis interaksi obat-makanan + reasoning |
| POST | `/recommend` | Rekomendasi makanan aman berdasarkan obat pasien |
| GET | `/alerts` | Riwayat notifikasi risiko tinggi |
| DELETE | `/alerts` | Hapus riwayat alert |

### Contoh Request `/interaction-check`

```json
{
  "yolo_class": "rendang",
  "patient_medications": ["Anticoagulants", "Antidiabetics"]
}
```

### Contoh Response

```json
{
  "detected_food": "rendang",
  "highest_severity_score": 4.0,
  "status": "warning",
  "detailed_predictions": [
    {
      "medication": "Anticoagulants",
      "severity_score": 3.95,
      "risk_level": "High Risk",
      "interaction_type": "AVOID",
      "mechanism": "CYP450_inhibition + pharmacodynamic_additive",
      "description": "Turmeric (curcumin) inhibits CYP3A4..."
    }
  ],
  "alert_sent": true,
  "alert_message": "Peringatan: interaksi berisiko tinggi...",
  "llm_reasoning": "Penjelasan dari Gemini dalam bahasa awam..."
}
```

### Contoh Request `/recommend`

```json
{
  "patient_medications": ["Anticoagulants", "Antihypertensives"],
  "top_n": 5
}
```

### Contoh Response `/recommend`

```json
{
  "patient_medications": ["Anticoagulants", "Antihypertensives"],
  "total_foods_analyzed": 35,
  "summary": {
    "safe": 20,
    "moderate": 10,
    "high_risk": 5
  },
  "recommended_foods": [
    {
      "food_name": "nasi-putih",
      "max_severity_score": 0.85,
      "risk_level": "Low Risk",
      "nutrition": {
        "calories_kcal": 175.0,
        "proteins_g": 4.0,
        "fats_g": 0.3,
        "carbohydrates_g": 40.0
      }
    }
  ],
  "foods_to_avoid": [
    {
      "food_name": "rendang",
      "max_severity_score": 3.95,
      "risk_level": "High Risk",
      "nutrition": { "..." }
    }
  ]
}
```

## Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi API Key Gemini

```bash
cp .env.example .env
# Edit .env dan masukkan API key Gemini yang valid
```

### 3. Training Model (opsional, model sudah tersedia)

Buka dan jalankan seluruh cell di `notebooks/01_model_training.ipynb`. Model akan disimpan ke `models/drug_risk_scorer.keras`.

### 4. Jalankan API Server

```bash
python app/main.py
```

Server berjalan di `http://localhost:8000`. Dokumentasi interaktif tersedia di `http://localhost:8000/docs`.

## Struktur Folder

```
ai engineer/
├── app/
│   ├── main.py                 # FastAPI server + endpoints
│   ├── model_inference.py      # Model loading + prediksi + LLM reasoning
│   └── nutrition_service.py    # Pipeline estimasi gizi
├── data/
│   ├── indonesian_food_drug_interactions.json
│   ├── Drug to Food interactions Dataset.json
│   ├── drug_food_kb_final.json
│   ├── mapping_image_nutrition.csv
│   └── unified_nutrition.csv
├── models/
│   ├── drug_risk_scorer.keras  # Model Hybrid NCF terlatih
│   └── nutrition_scaler.pkl    # MinMaxScaler untuk fitur nutrisi
├── notebooks/
│   └── 01_model_training.ipynb # Notebook training + evaluasi
├── logs/                       # TensorBoard logs
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Evaluasi Model

Hasil evaluasi dengan Stratified 5-Fold Cross Validation dan model final dilaporkan di notebook `01_model_training.ipynb`.

Target performa:
- MAE <= 0.02 (pada skala 0-1)
- Akurasi klasifikasi risiko >= 85%

Catatan: Jalankan ulang notebook untuk melihat angka evaluasi terbaru setelah perubahan arsitektur.

## TensorBoard

Log training tersimpan di folder `logs/`. Untuk memvisualisasikan:

```bash
tensorboard --logdir=logs
```

Buka `http://localhost:6006` di browser.
