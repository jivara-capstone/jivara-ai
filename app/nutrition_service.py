import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUTRITION_CSV_PATH = os.path.join(BASE_DIR, "data", "unified_nutrition.csv")

try:
    df_nut = pd.read_csv(NUTRITION_CSV_PATH)
    df_nut["food_name_lower"] = df_nut["food_name"].str.lower().str.strip()
except Exception as e:
    print(f"[WARNING] Gagal memuat data gizi: {e}")
    df_nut = pd.DataFrame()


def get_nutrition_estimate(yolo_class_name: str, portion_grams: float = 100.0) -> dict:
    """
    Estimasi nilai gizi berdasarkan nama makanan hasil deteksi YOLO.
    Matching: "nasi-putih" -> "nasi putih" (replace hyphen, case-insensitive).
    """
    if df_nut.empty:
        return {"error": "Database gizi gagal dimuat."}

    search_key = yolo_class_name.lower().replace("-", " ").strip()

    match = df_nut[df_nut["food_name_lower"] == search_key]
    if match.empty:
        match = df_nut[df_nut["food_name_lower"].str.contains(search_key, na=False)]
    if match.empty:
        return {"error": f"Data gizi untuk '{yolo_class_name}' tidak ditemukan."}

    nut = match.iloc[0]
    multiplier = portion_grams / 100.0

    def safe_mul(value):
        return round(float(value) * multiplier, 2) if pd.notna(value) else 0.0

    return {
        "status": "success",
        "yolo_class": yolo_class_name,
        "matched_food": nut["food_name"],
        "portion_grams": portion_grams,
        "nutrition_facts": {
            "calories_kcal": safe_mul(nut.get("calories", 0)),
            "proteins_g": safe_mul(nut.get("proteins", 0)),
            "fats_g": safe_mul(nut.get("fat", 0)),
            "carbohydrates_g": safe_mul(nut.get("carbohydrate", 0)),
        },
    }
