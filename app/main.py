from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model_inference import (
    check_interaction_with_reasoning,
    get_food_recommendations,
    init_model,
)
from app.nutrition_service import get_nutrition_estimate


# =====================================================================
# Startup: load model saat server dimulai
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield


app = FastAPI(
    title="Jivara AI Services API",
    description="REST API untuk analisis interaksi obat-makanan dan rekomendasi makanan aman.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

alert_history: list[dict] = []


# =====================================================================
# Request Schemas
# =====================================================================

class InteractionRequest(BaseModel):
    yolo_class: str
    patient_medications: list[str]


class NutritionRequest(BaseModel):
    yolo_class: str
    portion_grams: float = 100.0


class RecommendRequest(BaseModel):
    patient_medications: list[str]
    top_n: int = 10


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/")
def root():
    return {"message": "Jivara AI Services API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/nutrition", summary="Estimasi Nilai Gizi")
def nutrition(req: NutritionRequest):
    """Estimasi kalori, protein, lemak, karbohidrat berdasarkan makanan dan porsi."""
    result = get_nutrition_estimate(req.yolo_class, req.portion_grams)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/interaction-check", summary="Analisis Interaksi Obat-Makanan")
def interaction_check(req: InteractionRequest):
    """
    Cek risiko interaksi antara makanan dan obat pasien.
    Jika risiko tinggi, sertakan rekomendasi alternatif makanan aman.
    """
    result = check_interaction_with_reasoning(req.yolo_class, req.patient_medications)

    if result.get("status") == "warning":
        recs = get_food_recommendations(req.patient_medications, top_n=100)
        safe = [
            f for f in recs.get("recommended_foods", [])
            if f.get("risk_level") == "aman" and f.get("food_name") != req.yolo_class
        ]
        result["recommended_foods"] = safe[:10]

        alert = {
            "timestamp": datetime.now().isoformat(),
            "food": req.yolo_class,
            "medications": req.patient_medications,
            "severity_score": result["highest_severity"],
            "message": (
                f"Interaksi berisiko: '{req.yolo_class}' dengan "
                f"{', '.join(req.patient_medications)}. "
                f"Severity: {result['highest_severity']}/5.0"
            ),
        }
        alert_history.append(alert)
        result["alert_sent"] = True
    else:
        result["recommended_foods"] = []

    return result


@app.post("/recommend", summary="Rekomendasi Makanan Aman")
def recommend(req: RecommendRequest):
    """Daftar makanan diurutkan dari severity terendah berdasarkan obat pasien."""
    return get_food_recommendations(req.patient_medications, req.top_n)


@app.get("/alerts", summary="Riwayat Alert")
def get_alerts(limit: int = 20):
    recent = alert_history[-limit:]
    return {"total_alerts": len(alert_history), "alerts": list(reversed(recent))}


@app.delete("/alerts", summary="Hapus Alert")
def clear_alerts():
    alert_history.clear()
    return {"message": "Riwayat alert dihapus."}
