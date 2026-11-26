"""
FastAPI Application Entry Point:
Includes CORS, static file hosting for Radiologist Cockpit, and API routing.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.health import router as health_router
from api.routes.predict import router as predict_router

app = FastAPI(
    title="NeuroVision Elite - Clinical Brain Tumor Diagnostic Service",
    description=(
        "State-of-the-Art Dual-Task Brain Tumor AI: Multi-task Segmentation, "
        "Diagnostic Subtyping, Out-of-Distribution Safety, and Saliency Explainability (Grad-CAM)."
    ),
    version="2.0.0",
)

# CORS configuration for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(health_router)
app.include_router(predict_router)

# Mount frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
