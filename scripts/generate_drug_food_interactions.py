"""
Generate drug_food_interactions.csv — Ground Truth Interaksi Obat-Makanan
=========================================================================
Script ini menghasilkan dataset interaksi obat-makanan dengan severity 0-5
berdasarkan pengetahuan farmakologi klinis.

Perbedaan dengan rule-based binary sebelumnya:
1. Severity berskala 0-5 (bukan 0/1)
2. Ingredient yang sama bisa punya severity berbeda per kategori obat
3. Context-aware: "daun jeruk" != jeruk (grapefruit), "jeruk nipis" severity lebih rendah
4. Compounding risk: 3+ bahan bermasalah di satu makanan menaikkan severity
5. Menyertakan mekanisme interaksi dan tipe farmakologis

Output: data/drug_food_interactions.csv (854 baris = 61 makanan x 14 kategori obat)

CATATAN: Dataset ini di-generate LLM-assisted dan HARUS direview oleh
tim data science / farmasis sebelum digunakan sebagai ground truth final.
"""

import json
import csv
import os
from collections import Counter

# ============================================================
# Load food ingredient data
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOOD_KB_PATH = os.path.join(BASE_DIR, 'data', 'food_to_ingredient_kb.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'drug_food_interactions.csv')

with open(FOOD_KB_PATH, 'r', encoding='utf-8') as f:
    food_kb = json.load(f)
food_to_ingredients = food_kb['food_to_ingredients']

# ============================================================
# Ingredient-level severity mapping per drug category
#
# Setiap rule: (keyword, severity, exclude_if_contains)
#   - keyword: substring match terhadap daftar bahan makanan
#   - severity: 0-5 berdasarkan signifikansi klinis
#   - exclude_if_contains: skip jika ingredient mengandung string ini
#
# Severity scale:
#   0 = tidak ada interaksi
#   1 = teoretis/minimal, tidak signifikan secara klinis
#   2 = ringan, minor clinical significance
#   3 = sedang, perlu monitoring atau penyesuaian waktu
#   4 = signifikan, sebaiknya dihindari
#   5 = berat/berbahaya, kontraindikasi
# ============================================================

DRUG_SEVERITY = {
    "antikoagulan": {
        "rules": [
            ("kangkung", 5, []),       # vitamin K tinggi → antagonis langsung warfarin
            ("bayam", 5, []),          # vitamin K tinggi
            ("kemangi", 4, []),        # vitamin K moderat
            ("kunyit", 3, []),         # antiplatelet (curcumin)
            ("jahe", 3, []),           # antiplatelet (gingerol)
            ("bawang putih", 3, []),   # antiplatelet (allicin)
            ("lengkuas", 2, []),       # antiplatelet ringan
            ("daun bawang", 2, []),    # vitamin K minor
        ],
        "interaction_type": "pharmacodynamic",
        "mechanism_template": "Vitamin K dan/atau senyawa antiplatelet dalam {matched} mengganggu efek antikoagulan, meningkatkan risiko perdarahan atau mengurangi efektivitas obat",
    },
    "antidiabetes": {
        "rules": [
            ("gula pasir", 4, []),     # sukrosa langsung naikkan gula darah
            ("gula merah", 4, []),     # gula + indeks glikemik tinggi
            ("gula aren", 4, []),      # gula sederhana
            ("gula halus", 3, []),     # confectioner's sugar, porsi kecil
            ("kental manis", 4, []),   # susu kental manis = gula tinggi
            ("madu", 3, []),           # fruktosa + glukosa
            ("brown sugar", 3, []),    # English variant in recipes
            ("honey", 3, []),          # English variant
            ("gula", 2, ["gula pasir", "gula merah", "gula aren", "gula halus"]),  # generic "gula" minor
        ],
        "interaction_type": "pharmacodynamic",
        "mechanism_template": "Kandungan gula tinggi ({matched}) meningkatkan kadar glukosa darah, mengurangi efektivitas obat antidiabetes dan mengganggu kontrol glikemik",
    },
    "ace_arb": {
        "rules": [
            ("pisang", 4, []),         # kalium tinggi (~400mg/buah)
            ("kentang", 3, []),        # kalium moderat (~900mg/biji)
        ],
        "interaction_type": "pharmacodynamic",
        "mechanism_template": "Kalium tinggi dalam {matched} meningkatkan risiko hiperkalemia saat dikombinasikan dengan ACE inhibitor/ARB yang sudah menurunkan ekskresi kalium",
    },
    "ccb": {
        "rules": [
            # "daun jeruk" (kaffir lime leaf) TIDAK mengandung furanocoumarin
            # hanya jeruk/grapefruit/pomelo yang relevan
            ("jeruk", 4, ["daun jeruk"]),
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Furanocoumarin dalam {matched} menghambat CYP3A4 intestinal secara ireversibel, meningkatkan bioavailabilitas CCB hingga 2-3x lipat",
    },
    "statin": {
        "rules": [
            ("jeruk", 3, ["daun jeruk"]),   # CYP3A4 inhibisi
            ("santan", 3, []),               # lemak tinggi meningkatkan absorpsi
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Inhibisi CYP3A4 oleh jeruk dan/atau lemak tinggi dari santan ({matched}) meningkatkan bioavailabilitas statin, meningkatkan risiko miopati dan rhabdomyolysis",
    },
    "antibiotik_tetrasiklin": {
        "rules": [
            ("susu", 4, []),               # kalsium chelation
            ("kental manis", 4, []),        # susu kental = kalsium
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Kalsium dalam {matched} membentuk kompleks chelat dengan tetrasiklin yang tidak dapat diabsorpsi, mengurangi bioavailabilitas antibiotik 50-80%",
    },
    "antibiotik_fluorokuinolon": {
        "rules": [
            ("susu", 4, []),               # Ca2+ chelation
            ("kental manis", 4, []),        # susu kental = kalsium
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Kation divalen (Ca2+, Mg2+) dalam {matched} membentuk chelat dengan fluorokuinolon, mengurangi absorpsi dan efektivitas antibiotik",
    },
    "maoi": {
        "rules": [
            ("petis", 5, []),              # fermentasi tinggi → tyramine sangat tinggi
            ("tempe", 4, []),              # fermentasi kedelai → tyramine
            ("kecap", 4, []),              # fermentasi kedelai → tyramine
            ("terasi", 4, []),             # fermentasi udang → tyramine
            ("cuka", 2, []),               # asam asetat, tyramine minimal
        ],
        "interaction_type": "pharmacodynamic",
        "mechanism_template": "Tyramine dalam makanan fermentasi ({matched}) tidak dapat dimetabolisme saat MAO diinhibisi, menyebabkan pelepasan norepinefrin masif dan krisis hipertensi",
    },
    "tiroid": {
        "rules": [
            ("tahu", 4, []),               # isoflavon kedelai
            ("tempe", 4, []),              # isoflavon kedelai
            ("susu", 3, []),               # kalsium mengganggu absorpsi
            ("kental manis", 3, []),       # kalsium
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Isoflavon kedelai dan/atau kalsium dalam {matched} mengganggu absorpsi levothyroxine di saluran cerna, mengurangi efektivitas terapi tiroid",
    },
    "nsaid": {
        "rules": [
            ("cuka", 3, []),               # asam → iritasi GI aditif
            ("jeruk", 2, ["daun jeruk"]),   # asam sitrat → iritasi ringan
        ],
        "interaction_type": "pharmacodynamic",
        "mechanism_template": "Asam dalam {matched} memperburuk iritasi dan erosi mukosa lambung yang sudah diinduksi oleh penghambatan COX oleh NSAID",
    },
    "antikonvulsan": {
        "rules": [
            ("susu", 3, []),               # kalsium mengubah absorpsi
            ("kental manis", 3, []),        # kalsium
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Kalsium dan protein dalam {matched} membentuk kompleks dengan antikonvulsan, mengubah laju absorpsi dan kadar plasma obat",
    },
    "glikosida_jantung": {
        "rules": [
            ("pisang", 4, []),             # kalium tinggi → ubah toksisitas digitalis
            ("kangkung", 3, []),           # kalium moderat
            ("kentang", 3, []),            # kalium moderat
        ],
        "interaction_type": "pharmacodynamic",
        "mechanism_template": "Fluktuasi kadar kalium dari {matched} mempengaruhi sensitivitas miokardium terhadap glikosida jantung, meningkatkan risiko aritmia dan toksisitas digitalis",
    },
    "xantin": {
        "rules": [
            ("kopi", 4, []),               # kafein berkompetisi pada CYP1A2
            ("santan", 2, []),             # lemak tinggi mengubah farmakokinetik
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Kafein dalam {matched} berkompetisi dengan teofilin pada metabolisme CYP1A2, meningkatkan risiko toksisitas; lemak tinggi mengubah profil absorpsi",
    },
    "imunosupresan": {
        "rules": [
            ("jeruk", 4, ["daun jeruk"]),  # CYP3A4 inhibisi
            ("kunyit", 3, []),             # modulasi CYP3A4
            ("bawang putih", 3, []),       # modulasi CYP3A4 dan P-glikoprotein
        ],
        "interaction_type": "pharmacokinetic",
        "mechanism_template": "Senyawa dalam {matched} menghambat CYP3A4 dan/atau P-glikoprotein, meningkatkan kadar plasma imunosupresan dan risiko nefrotoksisitas",
    },
}

# ============================================================
# Jeruk nipis / jeruk limau → reduced severity
# (lime ≠ grapefruit, minimal furanocoumarin content)
# ============================================================
LIME_SEVERITY_OVERRIDE = {
    "ccb": 1,            # dari 4 → 1 (teoritis saja)
    "statin": 1,         # dari 3 → 1
    "nsaid": 1,          # dari 2 → 1
    "imunosupresan": 1,  # dari 4 → 1
}


def check_match(ingredient_lower, keyword_lower, excludes):
    """Check substring match with exclusion patterns."""
    for ex in excludes:
        if ex.lower() in ingredient_lower:
            return False
    return keyword_lower in ingredient_lower


def is_lime(ingredient_lower):
    """Cek apakah ingredient adalah jeruk nipis/limau (bukan grapefruit)."""
    return "jeruk nipis" in ingredient_lower or "jeruk limau" in ingredient_lower


def compute_severity(food_ingredients, drug_cat):
    """Hitung severity dan matched ingredients untuk satu pasangan makanan-obat."""
    info = DRUG_SEVERITY[drug_cat]
    matched = []
    max_severity = 0

    for keyword, base_sev, excludes in info["rules"]:
        for ingr in food_ingredients:
            ingr_lower = ingr.lower().strip()
            if check_match(ingr_lower, keyword.lower(), excludes):
                actual_sev = base_sev

                # Special case: jeruk nipis/limau mendapat severity lebih rendah
                if keyword == "jeruk" and is_lime(ingr_lower):
                    actual_sev = LIME_SEVERITY_OVERRIDE.get(drug_cat, base_sev)

                matched.append((keyword, actual_sev))
                if actual_sev > max_severity:
                    max_severity = actual_sev
                break  # satu keyword cukup match sekali

    # Compounding risk: 3+ bahan bermasalah → +1 severity (max 5)
    if len(matched) >= 3 and max_severity >= 2:
        max_severity = min(5, max_severity + 1)

    return max_severity, [m[0] for m in matched]


# ============================================================
# Generate 854 baris
# ============================================================
drug_categories = list(DRUG_SEVERITY.keys())
rows = []

for food in sorted(food_to_ingredients.keys()):
    ingredients = food_to_ingredients[food]

    for drug_cat in drug_categories:
        severity, matched = compute_severity(ingredients, drug_cat)
        info = DRUG_SEVERITY[drug_cat]

        has_interaction = 1 if severity >= 2 else 0

        if severity >= 2 and matched:
            interaction_type = info["interaction_type"]
            mechanism = info["mechanism_template"].format(matched=", ".join(matched))
        elif severity == 1 and matched:
            interaction_type = info["interaction_type"]
            mechanism = f"Interaksi minimal/teoretis: {', '.join(matched)} dalam jumlah kecil, tidak signifikan secara klinis"
        else:
            interaction_type = ""
            mechanism = ""

        rows.append({
            "food_class": food,
            "drug_category": drug_cat,
            "has_interaction": has_interaction,
            "severity": severity,
            "interaction_type": interaction_type,
            "mechanism": mechanism,
            "matched_ingredients": "|".join(matched) if matched else "",
            "source": "LLM-assisted curation (Claude) + pharmacology literature",
        })

# ============================================================
# Write CSV
# ============================================================
fieldnames = [
    "food_class", "drug_category", "has_interaction", "severity",
    "interaction_type", "mechanism", "matched_ingredients", "source",
]

with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ============================================================
# Summary
# ============================================================
total = len(rows)
n_interact = sum(1 for r in rows if r["has_interaction"] == 1)
n_safe = total - n_interact

print(f"{'=' * 60}")
print(f"  drug_food_interactions.csv — GENERATED")
print(f"{'=' * 60}")
print(f"  Total pasangan    : {total}")
print(f"  Interaksi (>=2)   : {n_interact} ({n_interact/total*100:.1f}%)")
print(f"  Aman (<2)         : {n_safe} ({n_safe/total*100:.1f}%)")

print(f"\n  Distribusi severity:")
sev_dist = Counter(r["severity"] for r in rows)
labels = {0: "aman", 1: "minimal", 2: "ringan", 3: "sedang", 4: "signifikan", 5: "berat"}
for s in sorted(sev_dist.keys()):
    print(f"    Severity {s} ({labels.get(s, '?'):11s}): {sev_dist[s]:4d} pasangan")

print(f"\n  Per kategori obat:")
for cat in drug_categories:
    cat_rows = [r for r in rows if r["drug_category"] == cat]
    n_int = sum(1 for r in cat_rows if r["has_interaction"] == 1)
    avg_sev = sum(r["severity"] for r in cat_rows if r["severity"] > 0) / max(1, sum(1 for r in cat_rows if r["severity"] > 0))
    print(f"    {cat:30s}: {n_int:2d}/61 interaksi, avg severity {avg_sev:.1f}")

print(f"\n  Output: {OUTPUT_PATH}")
print(f"{'=' * 60}")
