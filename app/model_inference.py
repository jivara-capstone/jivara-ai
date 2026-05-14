import json
import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# Global State
# =====================================================================

tree_pipeline = None
drug_categories: dict = {}
active_keywords: list[str] = []
food_to_ingredients: dict = {}
food_classes: list[str] = []
df_bpom: pd.DataFrame | None = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENABLE_GEMINI_REASONING = os.getenv("ENABLE_GEMINI_REASONING", "false").lower() == "true"

DEFAULT_DRUG_CATEGORIES = {
    "antikoagulan": {
        "keywords": ["WARFARIN", "CLOPIDOGREL", "HEPARIN", "ENOXAPARIN",
                     "RIVAROXABAN", "APIXABAN", "TICAGRELOR", "ACENOCOUMAROL"],
        "mechanism": "Meningkatkan efek pengencer darah / antagonis vitamin K",
    },
    "antidiabetes": {
        "keywords": ["METFORMIN", "GLIBENCLAMIDE", "GLIMEPIRIDE", "GLIPIZIDE",
                     "GLICLAZIDE", "INSULIN", "ACARBOSE", "PIOGLITAZONE",
                     "SITAGLIPTIN", "VILDAGLIPTIN"],
        "mechanism": "Makanan tinggi gula mengganggu kontrol glikemik",
    },
    "ace_arb": {
        "keywords": ["CAPTOPRIL", "ENALAPRIL", "LISINOPRIL", "RAMIPRIL",
                     "LOSARTAN", "VALSARTAN", "IRBESARTAN", "CANDESARTAN",
                     "TELMISARTAN", "PERINDOPRIL"],
        "mechanism": "Makanan tinggi kalium meningkatkan risiko hiperkalemia",
    },
    "ccb": {
        "keywords": ["AMLODIPINE", "NIFEDIPINE", "DILTIAZEM", "VERAPAMIL",
                     "FELODIPINE"],
        "mechanism": "Jeruk/grapefruit menghambat CYP3A4, meningkatkan kadar obat",
    },
    "statin": {
        "keywords": ["SIMVASTATIN", "ATORVASTATIN", "LOVASTATIN",
                     "ROSUVASTATIN", "PRAVASTATIN", "FLUVASTATIN"],
        "mechanism": "CYP3A4 inhibitor dan lemak tinggi meningkatkan absorpsi berlebih",
    },
    "antibiotik_tetrasiklin": {
        "keywords": ["DOXYCYCLINE", "TETRACYCLINE", "MINOCYCLINE",
                     "OXYTETRACYCLINE"],
        "mechanism": "Kalsium dan mineral mengikat antibiotik, mengurangi absorpsi",
    },
    "antibiotik_fluorokuinolon": {
        "keywords": ["CIPROFLOXACIN", "LEVOFLOXACIN", "MOXIFLOXACIN",
                     "OFLOXACIN", "NORFLOXACIN"],
        "mechanism": "Kation divalen (Ca, Mg, Fe) mengurangi absorpsi antibiotik",
    },
    "maoi": {
        "keywords": ["SELEGILINE", "MOCLOBEMIDE", "LINEZOLID",
                     "TRANYLCYPROMINE", "PHENELZINE", "RASAGILINE"],
        "mechanism": "Tyramine dalam makanan fermentasi menyebabkan krisis hipertensi",
    },
    "tiroid": {
        "keywords": ["LEVOTHYROXINE", "LIOTHYRONINE", "THYROXINE",
                     "LEVOTIROKSIN"],
        "mechanism": "Kedelai dan kalsium mengganggu absorpsi hormon tiroid",
    },
    "nsaid": {
        "keywords": ["IBUPROFEN", "DIKLOFENAK", "DICLOFENAC", "MELOXICAM",
                     "PIROXICAM", "KETOROLAC", "NAPROXEN",
                     "ASAM MEFENAMAT", "INDOMETASIN", "CELECOXIB"],
        "mechanism": "Asam memperburuk iritasi lambung yang disebabkan NSAID",
    },
    "antikonvulsan": {
        "keywords": ["PHENYTOIN", "FENITOIN", "CARBAMAZEPINE",
                     "KARBAMAZEPIN", "VALPROIC", "PHENOBARBITAL"],
        "mechanism": "Kalsium dan protein tinggi mengubah absorpsi antikonvulsan",
    },
    "glikosida_jantung": {
        "keywords": ["DIGOXIN", "DIGOKSIN"],
        "mechanism": "Perubahan kadar kalium mempengaruhi toksisitas digitalis",
    },
    "xantin": {
        "keywords": ["THEOPHYLLINE", "TEOFILIN", "AMINOPHYLLINE", "AMINOFILIN"],
        "mechanism": "Kafein berkompetisi; lemak tinggi mengubah farmakokinetik",
    },
    "imunosupresan": {
        "keywords": ["CYCLOSPORINE", "SIKLOSPORIN", "TACROLIMUS",
                     "SIROLIMUS", "EVEROLIMUS", "MYCOPHENOLATE"],
        "mechanism": "CYP3A4 inhibitor meningkatkan kadar dan toksisitas obat",
    },
}


# =====================================================================
# Inisialisasi Model
# =====================================================================

def init_model() -> bool:
    """Load model ExtraTrees dan semua artifacts yang dibutuhkan."""
    global tree_pipeline, drug_categories, active_keywords
    global food_to_ingredients, food_classes, df_bpom

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "drug_interaction_tree_model.pkl")
    food_kb_path = os.path.join(base_dir, "data", "food_to_ingredient_kb.json")
    bpom_path = os.path.join(base_dir, "data", "obat_bpom_cleaned_full.csv")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return False

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    tree_pipeline = bundle["pipeline"]
    active_keywords = bundle.get("active_keywords", [])
    food_classes = bundle.get("food_classes", [])
    drug_categories = bundle.get("drug_categories") or DEFAULT_DRUG_CATEGORIES

    if os.path.exists(food_kb_path):
        with open(food_kb_path, "r", encoding="utf-8") as f:
            food_to_ingredients = json.load(f)["food_to_ingredients"]

    if os.path.exists(bpom_path):
        df_bpom = pd.read_csv(bpom_path)

    print(f"[OK] Model ExtraTrees dimuat — {len(food_classes)} makanan, {len(drug_categories)} kategori obat")
    return True


# =====================================================================
# Utilitas
# =====================================================================

def _severity_to_risk(severity: float) -> str:
    if severity < 1.0:
        return "aman"
    if severity < 2.5:
        return "ringan"
    if severity < 3.5:
        return "sedang"
    return "tinggi"


def _get_ingredient_features(food_name: str) -> list[float]:
    ingredients = food_to_ingredients.get(food_name, [])
    return [
        1.0 if any(kw.lower() in ingr.lower() for ingr in ingredients) else 0.0
        for kw in active_keywords
    ]


def _predict_severity(food_name: str, drug_category: str) -> float:
    features = {"food_class": food_name, "drug_category": drug_category}
    for kw, val in zip(active_keywords, _get_ingredient_features(food_name)):
        features[f"kw_{kw}"] = val
    df = pd.DataFrame([features])
    severity = float(tree_pipeline.predict(df)[0])
    return float(np.clip(severity, 0.0, 5.0))


def map_drug_to_categories(drug_name: str) -> list[str]:
    """Map nama obat ke kategori interaksi (keyword match + BPOM fallback)."""
    drug_upper = drug_name.strip().upper()
    matched = []

    for cat, info in drug_categories.items():
        for kw in info["keywords"]:
            if kw in drug_upper:
                matched.append(cat)
                break

    if not matched and df_bpom is not None:
        bpom_match = df_bpom[
            df_bpom["Nama Produk"].str.upper().str.contains(drug_upper, na=False)
        ]
        if not bpom_match.empty:
            composition = str(bpom_match.iloc[0]["Komposisi"]).upper()
            for cat, info in drug_categories.items():
                for kw in info["keywords"]:
                    if kw in composition:
                        matched.append(cat)
                        break

    return list(set(matched))


# =====================================================================
# Prediksi Interaksi & Rekomendasi
# =====================================================================

def check_interaction_with_reasoning(
    food_name: str,
    medications: list[str],
) -> dict:
    """Cek interaksi obat-makanan + LLM reasoning untuk risiko tinggi."""
    if tree_pipeline is None:
        init_model()

    all_cats: set[str] = set()
    med_cat_map: dict[str, list[str]] = {}
    for med in medications:
        cats = map_drug_to_categories(med)
        med_cat_map[med] = cats if cats else ["(tidak ditemukan)"]
        all_cats.update(cats)

    predictions = []
    highest_severity = 0.0
    high_risk_detected = False

    for med in medications:
        cats = [c for c in med_cat_map[med] if c != "(tidak ditemukan)"]
        max_severity = 0.0
        risky_cats = []

        for cat in cats:
            severity = _predict_severity(food_name, cat)
            if severity > max_severity:
                max_severity = severity
            if severity >= 2.5:
                risky_cats.append(cat)

        risk_level = _severity_to_risk(max_severity)
        if max_severity >= 3.0:
            high_risk_detected = True
        if max_severity > highest_severity:
            highest_severity = max_severity

        entry: dict = {
            "medication": med,
            "matched_categories": med_cat_map[med],
            "severity_score": round(max_severity, 2),
            "risk_level": risk_level,
        }
        if risky_cats:
            entry["risky_categories"] = risky_cats
            entry["mechanisms"] = [
                drug_categories.get(c, {}).get("mechanism", "") for c in risky_cats
            ]
        predictions.append(entry)

    result: dict = {
        "detected_food": food_name,
        "highest_severity": round(highest_severity, 2),
        "status": "warning" if high_risk_detected else "safe",
        "detailed_predictions": predictions,
    }

    # LLM reasoning untuk risiko tinggi
    if high_risk_detected and GEMINI_API_KEY and ENABLE_GEMINI_REASONING:
        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            high_risk_meds = [
                p["medication"]
                for p in predictions
                if p["risk_level"] in ("sedang", "tinggi")
            ]
            llm = genai.GenerativeModel("gemini-2.5-flash")
            prompt = (
                f"Anda adalah asisten kesehatan virtual bernama Jivara.\n"
                f"Pasien ingin makan '{food_name}' dan mengonsumsi obat: {', '.join(high_risk_meds)}.\n"
                f"Sistem mendeteksi risiko interaksi (severity {highest_severity:.1f}/5).\n\n"
                f"Jelaskan:\n1. Mengapa kombinasi ini berisiko\n"
                f"2. Apa yang bisa terjadi\n3. Saran alternatif\n\n"
                f"Bahasa Indonesia, ramah, 3-4 paragraf pendek."
            )
            result["llm_reasoning"] = llm.generate_content(prompt).text
        except Exception as e:
            result["llm_reasoning"] = f"Gagal memuat penjelasan AI: {e}"
    elif high_risk_detected:
        result["llm_reasoning"] = (
            "Peringatan: terdeteksi risiko interaksi obat-makanan tingkat tinggi. "
            "Silakan konsultasikan dengan dokter atau apoteker Anda."
        )
    else:
        result["llm_reasoning"] = (
            "Kombinasi makanan dan obat ini diprediksi aman. "
            "Tetap perhatikan porsi dan waktu konsumsi obat sesuai anjuran dokter."
        )

    return result


def get_food_recommendations(medications: list[str], top_n: int = 10) -> dict:
    """Rekomendasi makanan aman berdasarkan predicted severity."""
    if tree_pipeline is None:
        init_model()

    all_cats: set[str] = set()
    med_cat_map: dict[str, list[str]] = {}
    for med in medications:
        cats = map_drug_to_categories(med)
        med_cat_map[med] = cats if cats else ["(tidak ditemukan)"]
        all_cats.update(cats)

    if not all_cats:
        return {
            "patient_medications": medications,
            "matched_categories": med_cat_map,
            "status": "no_interaction_data",
            "message": "Obat tidak ditemukan dalam database.",
            "total_foods_analyzed": len(food_classes),
            "summary": {"safe": len(food_classes), "avoid": 0},
            "recommended_foods": [
                {"food_name": f, "severity_score": 0.0, "risk_level": "aman"}
                for f in food_classes[:top_n]
            ],
            "foods_to_avoid": [],
        }

    food_scores = []
    for food in food_classes:
        max_severity = 0.0
        worst_cat = None
        for cat in all_cats:
            severity = _predict_severity(food, cat)
            if severity > max_severity:
                max_severity = severity
                worst_cat = cat

        food_scores.append({
            "food_name": food,
            "severity_score": round(max_severity, 2),
            "risk_level": _severity_to_risk(max_severity),
            "worst_category": worst_cat,
        })

    food_scores.sort(key=lambda x: x["severity_score"])

    safe_foods = [f for f in food_scores if f["risk_level"] in ("aman", "ringan")]
    avoid_foods = [f for f in food_scores if f["risk_level"] in ("sedang", "tinggi")]

    return {
        "patient_medications": medications,
        "matched_categories": med_cat_map,
        "total_foods_analyzed": len(food_classes),
        "summary": {"safe": len(safe_foods), "avoid": len(avoid_foods)},
        "recommended_foods": food_scores[:top_n],
        "foods_to_avoid": list(reversed(avoid_foods)),
    }
