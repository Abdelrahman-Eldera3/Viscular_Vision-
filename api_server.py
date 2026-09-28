"""
Vascular Vision — Vessel Segmentation & Video Analysis REST API
==============================================================
FastAPI Server powered by pad_pipeline_colab.py, medical_utils.py & sensitivity_analysis.py

Features:
- Direct video & DICOM (.dcm) multi-frame cine extraction
- Automatic medical fluoroscopy / angiography validation check
- Millimeter calibration (DICOM PixelSpacing & Catheter French size)
- FFR physiological limitations & uncertainty reporting
- Renamed tracking split events & static pre-existing branching detection
- Threshold sensitivity analysis endpoint
- Security: API key auth, streaming max upload size (500MB), CORS fix, and TTL cleanup
"""

import os
import time
import shutil
import uuid
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends, Security, Request, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import pipeline components
from pad_pipeline_frameselect import run_inference_frameselect as run_inference, format_summary_frameselect as format_summary_text
from pad_pipeline_colab import get_model, _get_device
from medical_utils import is_dicom_file, dicom_to_video, validate_medical_angiography
from sensitivity_analysis import run_threshold_sensitivity

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VascularVisionAPI")

# ── Security & Environment Config ─────────────────────────────────────────
DEBUG_MODE = os.environ.get("DEBUG", "true").lower() in ["true", "1", "yes"]
CONFIGURED_API_KEY = os.environ.get("API_KEY")
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 2048)) * 1024 * 1024  # 2048 MB (2 GB)
OUTPUT_TTL_SECONDS = int(os.environ.get("OUTPUT_TTL_SECONDS", 7200))  # 2 Hours

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Directories ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "api_outputs"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)):
    """
    Enforce API key validation.
    Fails in production if API_KEY is unset.
    Allows unauthenticated requests only if explicit DEBUG=true is set for local dev.
    """
    if DEBUG_MODE:
        return True

    if not CONFIGURED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Misconfiguration: API_KEY is required in production mode.",
        )

    if not api_key or api_key != CONFIGURED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing 'X-API-Key' header.",
        )
    return True


# ── FastAPI App Setup ──────────────────────────────────────────────────────
app = FastAPI(
    title="Vascular Vision API & Web Dashboard",
    description=(
        "Production-ready REST API & Medical Dashboard for Peripheral Angiography Vessel Segmentation, "
        "DICOM Cine Processing, QCA Stenosis Quantification, and FFR Estimation."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Secure CORS: Disallow wildcard origins when credentials are enabled
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False if "*" in ALLOWED_ORIGINS else True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount Static Files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Response Schemas ──────────────────────────────────────────────────────
class FFRUncertainty(BaseModel):
    real_std_dev_pct: Optional[float] = None
    empirical_range_pct: Optional[List[float]] = None
    n_frames_analyzed: Optional[int] = None
    note: Optional[str] = None


class VesselDetail(BaseModel):
    vessel_id: int
    original_tracking_id: Optional[int] = None
    anatomical_segment: Optional[str] = None
    anatomical_segment_ar: Optional[str] = None
    anatomical_region: Optional[str] = None
    anatomical_short: Optional[str] = None
    is_branch: bool
    parent_id: Optional[int] = None
    frame_count: int
    mean_ffr_pct: Optional[float] = None
    min_ffr_pct: Optional[float] = None
    hemodynamic_risk_estimate: Optional[float] = None
    hemodynamic_risk_label: Optional[str] = None
    min_ffr_uncertainty: Optional[FFRUncertainty] = None
    mean_DS_pct: Optional[float] = None
    max_DS_pct: Optional[float] = None
    max_DS_frame: Optional[int] = None
    MLD_px: Optional[float] = None
    RVD_px: Optional[float] = None
    stenosis_length_px: Optional[float] = None
    MLD_mm: Optional[float] = None
    RVD_mm: Optional[float] = None
    stenosis_length_mm: Optional[float] = None
    is_calibrated: Optional[bool] = False
    calibration_tag: Optional[str] = None
    interpretation: str
    recommendation: str
    ds_classification: Optional[str] = None
    stenosis_status: Optional[str] = None
    anatomical_functional_disagreement: bool = False
    low_contrast_only: bool = False
    appearance_ratio: Optional[float] = None
    tracking_coverage: Optional[float] = None
    quality_gate_passed: Optional[bool] = True
    confidence_level: Optional[str] = "STANDARD"
    confidence_reason: Optional[str] = None
    is_borderline: Optional[bool] = False
    is_late_washout_bone: Optional[bool] = False
    is_transient_flicker: Optional[bool] = False
    transient_low_confidence: bool = False
    branch_classification: Optional[str] = None
    branch_stability_ratio: Optional[float] = None
    parent_frame_count_during_overlap: Optional[int] = None
    cto_suspected: Optional[bool] = False
    cto_trigger_frame: Optional[int] = None
    cto_narrowest_pt: Optional[List[float]] = None
    is_total_occlusion: Optional[bool] = False
    best_measurement_frame: Optional[int] = None
    best_measurement_frame_score_breakdown: Optional[Dict[str, float]] = None


class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float


class ValidationResult(BaseModel):
    is_valid: bool
    confidence: float
    reason: str
    modality_type: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    success: bool
    job_id: str
    input_type: str
    is_medical_angiography: bool
    validation: ValidationResult
    total_frames: int
    analyzed_frames: int
    n_confirmed_vessels: int
    per_vessel: Dict[str, VesselDetail]
    bifurcation_events_detected_during_tracking: int
    possible_preexisting_bifurcations: int
    loops_excluded: int
    stenosis_method: str
    ffr_formula: str
    ffr_estimation_method: str
    physiological_limitations_note: str
    calibration_status: str
    calibration_note: str
    scale_mm_per_px: Optional[float] = None
    video_info: VideoInfo
    dicom_metadata: Optional[Dict[str, Any]] = None
    text_report: str
    downloads: Dict[str, str]
    diagnostic_frames: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    cuda_available: bool
    dicom_support: bool
    debug_mode: bool
    auth_enabled: bool


# ── Storage TTL & Cleanup ─────────────────────────────────────────────────
def cleanup_expired_jobs(max_age_seconds: int = OUTPUT_TTL_SECONDS):
    """Clean up old job output folders that have exceeded the TTL."""
    now = time.time()
    try:
        for job_folder in STORAGE_DIR.iterdir():
            if job_folder.is_dir():
                folder_age = now - job_folder.stat().st_mtime
                if folder_age > max_age_seconds:
                    shutil.rmtree(job_folder, ignore_errors=True)
                    logger.info(f"[TTL Cleanup] Removed expired job folder: {job_folder.name}")
    except Exception as e:
        logger.warning(f"[TTL Cleanup] Failed during cleanup: {e}")


async def save_upload_with_size_limit(upload_file: UploadFile, dest_path: Path, max_bytes: int = MAX_UPLOAD_SIZE_BYTES):
    """Stream and save uploaded file while strictly enforcing size limit before disk overflow."""
    bytes_written = 0
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = await upload_file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                buffer.close()
                if dest_path.exists():
                    dest_path.unlink()
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Uploaded file exceeds the maximum allowed size limit of {max_bytes // (1024*1024)} MB.",
                )
            buffer.write(chunk)


async def periodic_ttl_cleanup_worker(interval_seconds: int = 600):
    """
    Scheduled background worker task that periodically runs every interval_seconds (default: 10 mins)
    and removes any job directories in api_outputs/ exceeding OUTPUT_TTL_SECONDS.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            cleanup_expired_jobs()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[Scheduled TTL Cleanup Worker] Error during cycle: {e}")


# ── App Lifespan / Startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Startup initialization, security validation, model warm-up, and background workers."""
    logger.info("Initializing Vascular Vision API...")
    if not DEBUG_MODE and not CONFIGURED_API_KEY:
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: 'API_KEY' environment variable is required in production mode. "
            "To run locally for testing, set environment variable DEBUG=true."
        )

    # Initial sweep
    cleanup_expired_jobs()

    # Launch scheduled periodic cleanup worker
    asyncio.create_task(periodic_ttl_cleanup_worker(interval_seconds=600))
    logger.info(f"[Scheduled TTL Cleanup] Periodic background worker started (interval: 600s, TTL: {OUTPUT_TTL_SECONDS}s).")

    try:
        model = get_model()
        device = _get_device()
        model.to(device)
        logger.info(f"Model successfully loaded on: {device} | Debug Mode: {DEBUG_MODE}")
    except Exception as e:
        logger.error(f"Error loading model during startup: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Web Dashboard"], response_class=FileResponse)
def serve_dashboard():
    """Serves the interactive web medical dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(index_path, media_type="text/html")


@app.get("/api", tags=["General"])
def api_info():
    """API overview and endpoints discovery."""
    return {
        "message": "Welcome to Vascular Vision Vessel Analysis API v2.0",
        "web_dashboard": "/",
        "docs": "/docs",
        "health": "/api/v1/health",
        "analyze": "/api/v1/analyze",
        "validate": "/api/v1/validate",
        "sensitivity_analysis": "/api/v1/sensitivity-analysis",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Check API health and GPU/CPU/DICOM availability."""
    device = _get_device()
    try:
        import pydicom
        has_pydicom = True
    except ImportError:
        has_pydicom = False

    return HealthResponse(
        status="healthy",
        device=str(device),
        model_loaded=True,
        cuda_available=torch.cuda.is_available(),
        dicom_support=has_pydicom,
        debug_mode=DEBUG_MODE,
        auth_enabled=bool(CONFIGURED_API_KEY),
    )


@app.post("/api/v1/validate", response_model=ValidationResult, tags=["Validation"], dependencies=[Depends(verify_api_key)])
async def validate_input_video(
    file: UploadFile = File(..., description="Video or DICOM file to validate as medical angiography"),
):
    """
    Validates whether an uploaded file is a genuine medical fluoroscopy/angiography sequence.
    Detects if the video is arbitrary color video, cartoons, noise, or invalid format.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="val_"))
    temp_file_path = temp_dir / file.filename

    try:
        await save_upload_with_size_limit(file, temp_file_path)

        if is_dicom_file(str(temp_file_path)):
            converted_mp4 = temp_dir / "dcm_preview.mp4"
            dicom_to_video(str(temp_file_path), str(converted_mp4))
            video_to_check = str(converted_mp4)
        else:
            video_to_check = str(temp_file_path)

        val_result = validate_medical_angiography(video_to_check)
        return ValidationResult(**val_result)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def analyze_video(
    file: UploadFile = File(..., description="Angiography video (MP4, AVI, MOV) or DICOM file (.dcm)"),
    threshold: float = Form(0.50, ge=0.1, le=0.9, description="Segmentation probability threshold (default: 0.50)"),
    force_analysis: bool = Form(False, description="Bypass medical angiography validation check"),
    pixel_spacing_mm: Optional[float] = Form(None, description="Known pixel spacing in mm/px for millimeter calibration"),
    catheter_french_size: Optional[float] = Form(None, description="Catheter size in French (e.g. 5, 6, 7, 8 Fr) for calibration"),
    catheter_diameter_px: Optional[float] = Form(None, description="Measured catheter diameter in pixels for calibration"),
):
    """
    Full End-to-End Angiography Analysis:
    1. Multi-format ingestion: DICOM (.dcm) & standard video formats (.mp4, .avi, .mov).
    2. Real millimeter calibration via DICOM PixelSpacing or catheter French size ($1\\text{ Fr} = 0.333\\text{ mm}$).
    3. Medical angiography validation check.
    4. VGG16-UNet++ segmentation, multi-frame tracking, static pre-existing branching detection, and QCA quantification.
    5. FFR calculation with uncertainty bounds and physiological limitation disclosure.
    """
    cleanup_expired_jobs()

    job_id = str(uuid.uuid4())
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    clean_filename = Path(file.filename or "upload.mp4").name
    input_temp_path = job_dir / f"input_{clean_filename}"

    # Stream save with size limit
    await save_upload_with_size_limit(file, input_temp_path)

    dicom_meta = None
    input_type = "video"
    actual_video_path = str(input_temp_path)
    effective_pixel_spacing = pixel_spacing_mm

    # 1. Check if file is DICOM
    if is_dicom_file(str(input_temp_path)):
        input_type = "dicom"
        converted_mp4 = str(job_dir / "dicom_extracted.mp4")
        logger.info(f"[Job {job_id}] Detected DICOM file. Converting to MP4...")
        try:
            dicom_meta = dicom_to_video(str(input_temp_path), converted_mp4)
            actual_video_path = converted_mp4
            if not effective_pixel_spacing and dicom_meta.get("pixel_spacing_mm"):
                effective_pixel_spacing = dicom_meta["pixel_spacing_mm"]
        except Exception as e:
            shutil.rmtree(job_dir, ignore_errors=True)
            err_msg = str(e)
            if not err_msg.startswith("Failed to decode DICOM") and not err_msg.startswith("Could not decode DICOM") and not err_msg.startswith("The uploaded DICOM"):
                err_msg = f"Failed to decode DICOM pixel data: {err_msg}"
            raise HTTPException(status_code=400, detail=err_msg)
    else:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".mp4", ".avi", ".mov", ".mkv"]:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{file_ext}'. Allowed formats: .dcm, .mp4, .avi, .mov, .mkv",
            )

    # 2. Medical Angiography / Fluoroscopy Validation Check
    logger.info(f"[Job {job_id}] Validating if video is medical fluoroscopy/angiography...")
    validation_info = validate_medical_angiography(actual_video_path)

    if not validation_info["is_valid"] and not force_analysis:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.warning(f"[Job {job_id}] Validation failed: {validation_info['reason']}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Medical Validation Failed",
                "message": validation_info["reason"],
                "modality_detected": validation_info["modality_type"],
                "confidence": validation_info["confidence"],
                "metrics": validation_info["metrics"],
                "tip": "If you are sure this is valid medical footage, you can set force_analysis=true.",
            },
        )

    # 3. Run inference pipeline with calibration
    try:
        loop = asyncio.get_running_loop()
        overlay_path, mask_path, csv_path, summary = await loop.run_in_executor(
            None,
            lambda: run_inference(
                input_video_path=actual_video_path,
                threshold=threshold,
                work_dir=str(job_dir),
                pixel_spacing_mm=effective_pixel_spacing,
                catheter_french_size=catheter_french_size,
                catheter_diameter_px=catheter_diameter_px,
            )
        )
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.error(f"[Job {job_id}] Inference failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

    # Format text report summary
    text_report = format_summary_text(summary)

    # Build download and streaming URLs
    downloads = {
        "overlay_video": f"/api/v1/download/{job_id}/overlay",
        "mask_video": f"/api/v1/download/{job_id}/mask",
        "stats_csv": f"/api/v1/download/{job_id}/csv",
        "stream_overlay": f"/api/v1/stream/{job_id}/overlay",
        "stream_mask": f"/api/v1/stream/{job_id}/mask",
    }
    if input_type == "dicom":
        downloads["dicom_video"] = f"/api/v1/download/{job_id}/converted_video"
        downloads["stream_dicom"] = f"/api/v1/stream/{job_id}/converted_video"

    # Structure per-vessel results matching Pydantic schema
    per_vessel_data = {}
    for vid, vdata in summary.get("per_vessel", {}).items():
        per_vessel_data[vid] = VesselDetail(
            vessel_id=vdata["vessel_id"],
            is_branch=vdata["is_branch"],
            parent_id=vdata["parent_id"],
            frame_count=vdata["frame_count"],
            mean_ffr_pct=vdata["mean_ffr_pct"],
            min_ffr_pct=vdata["min_ffr_pct"],
            min_ffr_uncertainty=FFRUncertainty(**vdata["min_ffr_uncertainty"]),
            mean_DS_pct=vdata["mean_DS_pct"],
            max_DS_pct=vdata["max_DS_pct"],
            max_DS_frame=vdata["max_DS_frame"],
            MLD_px=vdata["MLD_px"],
            RVD_px=vdata["RVD_px"],
            stenosis_length_px=vdata["stenosis_length_px"],
            MLD_mm=vdata.get("MLD_mm"),
            RVD_mm=vdata.get("RVD_mm"),
            stenosis_length_mm=vdata.get("stenosis_length_mm"),
            interpretation=vdata["interpretation"],
            recommendation=vdata["recommendation"],
            ds_classification=vdata.get("ds_classification"),
            stenosis_status=vdata.get("stenosis_status", vdata.get("ds_classification")),
            anatomical_functional_disagreement=vdata.get("anatomical_functional_disagreement", False),
            low_contrast_only=vdata.get("low_contrast_only", False),
            appearance_ratio=vdata.get("appearance_ratio"),
            transient_low_confidence=vdata.get("transient_low_confidence", False),
            branch_classification=vdata.get("branch_classification"),
            branch_stability_ratio=vdata.get("branch_stability_ratio"),
            parent_frame_count_during_overlap=vdata.get("parent_frame_count_during_overlap"),
            original_tracking_id=vdata.get("original_tracking_id"),
            anatomical_segment=vdata.get("anatomical_segment"),
            anatomical_segment_ar=vdata.get("anatomical_segment_ar"),
            anatomical_region=vdata.get("anatomical_region"),
            anatomical_short=vdata.get("anatomical_short"),
            tracking_coverage=vdata.get("tracking_coverage"),
            confidence_level=vdata.get("confidence_level", "STANDARD"),
            confidence_reason=vdata.get("confidence_reason"),
            quality_gate_passed=vdata.get("quality_gate_passed", True),
            is_borderline=vdata.get("is_borderline", False),
            is_late_washout_bone=vdata.get("is_late_washout_bone", False),
            is_transient_flicker=vdata.get("is_transient_flicker", False),
            cto_suspected=vdata.get("cto_suspected", False),
            cto_trigger_frame=vdata.get("cto_trigger_frame"),
            cto_narrowest_pt=list(vdata.get("cto_narrowest_pt")) if vdata.get("cto_narrowest_pt") else None,
            is_total_occlusion=vdata.get("is_total_occlusion", False),
            best_measurement_frame=vdata.get("best_measurement_frame"),
            best_measurement_frame_score_breakdown=vdata.get("best_measurement_frame_score_breakdown"),
        )

    response = AnalysisResponse(
        success=summary["success"],
        job_id=job_id,
        input_type=input_type,
        is_medical_angiography=validation_info["is_valid"],
        validation=ValidationResult(**validation_info),
        total_frames=summary["total_frames"],
        analyzed_frames=summary["analyzed_frames"],
        n_confirmed_vessels=len(per_vessel_data) if per_vessel_data else summary.get("n_confirmed_vessels", 0),
        per_vessel=per_vessel_data,
        bifurcation_events_detected_during_tracking=summary["bifurcation_events_detected_during_tracking"],
        possible_preexisting_bifurcations=summary["possible_preexisting_bifurcations"],
        loops_excluded=summary["loops_excluded"],
        stenosis_method=summary["stenosis_method"],
        ffr_formula=summary["ffr_formula"],
        ffr_estimation_method=summary["ffr_estimation_method"],
        physiological_limitations_note=summary["physiological_limitations_note"],
        calibration_status=summary["calibration_status"],
        calibration_note=summary["calibration_note"],
        scale_mm_per_px=summary["scale_mm_per_px"],
        video_info=VideoInfo(**summary["video_info"]),
        dicom_metadata=dicom_meta,
        text_report=text_report,
        downloads=downloads,
        diagnostic_frames=summary.get("diagnostic_frames", []),
    )

    logger.info(f"[Job {job_id}] Finished successfully. Calibration: {summary['calibration_status']}")
    return response


@app.post("/api/v1/sensitivity-analysis", tags=["Sensitivity Analysis"], dependencies=[Depends(verify_api_key)])
async def sensitivity_analysis_endpoint(
    file: UploadFile = File(..., description="Angiography video or DICOM file"),
    thresholds_csv: Optional[str] = Form("0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70", description="Comma-separated threshold list"),
):
    """
    Executes threshold sensitivity analysis across probability thresholds (0.30 to 0.70)
    to assess stability of MLD, DS%, and FFR measurements.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="sens_"))
    temp_file_path = temp_dir / file.filename

    try:
        await save_upload_with_size_limit(file, temp_file_path)

        if is_dicom_file(str(temp_file_path)):
            converted_mp4 = temp_dir / "converted.mp4"
            dicom_to_video(str(temp_file_path), str(converted_mp4))
            target_video = str(converted_mp4)
        else:
            target_video = str(temp_file_path)

        thresholds = [float(x.strip()) for x in thresholds_csv.split(",") if x.strip()]
        report = run_threshold_sensitivity(target_video, thresholds=thresholds)
        return report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/api/v1/download/{job_id}/{file_type}", tags=["Downloads"])
def download_result_file(job_id: str, file_type: str):
    """
    Download analysis output files:
    - `overlay`: Annotated visualization video with QCA & FFR overlays (.mp4)
    - `mask`: Segmentation binary mask video (.mp4)
    - `csv`: Full per-frame analysis statistics (.csv)
    - `converted_video`: Extracted MP4 from original DICOM file (.mp4)
    """
    job_dir = STORAGE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job ID not found or expired")

    file_mapping = {
        "overlay": (job_dir / "output_OVERLAY.mp4", "video/mp4", "analysis_overlay.mp4"),
        "mask": (job_dir / "output_MASK.mp4", "video/mp4", "vessel_mask.mp4"),
        "csv": (job_dir / "output_stats.csv", "text/csv", "vessel_stats.csv"),
        "converted_video": (job_dir / "dicom_extracted.mp4", "video/mp4", "dicom_extracted.mp4"),
    }

    if file_type not in file_mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_type}'. Allowed types: 'overlay', 'mask', 'csv', 'converted_video'",
        )

    file_path, media_type, download_name = file_mapping[file_type]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_type}' not found for this job.")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=download_name,
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'}
    )


@app.get("/api/v1/stream/{job_id}/{file_type}", tags=["Streaming"])
def stream_video_file(job_id: str, file_type: str):
    """
    Stream video directly for browser HTML5 video players (no attachment header, supports 206 Partial Content).
    """
    job_dir = STORAGE_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job ID not found or expired")

    file_mapping = {
        "overlay": job_dir / "output_OVERLAY.mp4",
        "mask": job_dir / "output_MASK.mp4",
        "converted_video": job_dir / "dicom_extracted.mp4",
    }
    file_path = file_mapping.get(file_type)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_type}' not found.")

    return FileResponse(path=str(file_path), media_type="video/mp4")


@app.get("/api/v1/stream/{job_id}/frame/{frame_idx}/{file_type}", tags=["Streaming"])
def stream_diagnostic_frame_image(job_id: str, frame_idx: int, file_type: str):
    """
    Stream individual diagnostic frame, mask, or QCA overlay image.
    `file_type`: 'raw', 'mask', or 'overlay_v{vid}' / 'overlay'
    """
    job_dir = STORAGE_DIR / job_id / "diagnostic_frames"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Diagnostic frames directory not found")

    target = None
    if file_type == "raw":
        target = job_dir / f"raw_{frame_idx}.jpg"
    elif file_type == "mask":
        target = job_dir / f"mask_{frame_idx}.png"
    elif file_type.startswith("overlay"):
        target = job_dir / f"{file_type}_{frame_idx}.jpg"
        if not target.exists():
            target = job_dir / f"{file_type}.jpg"
        if not target.exists():
            for f in job_dir.glob(f"overlay_{frame_idx}*.jpg"):
                target = f
                break

    if not target or not target.exists():
        for f in job_dir.glob(f"*{frame_idx}*"):
            if file_type in f.name:
                target = f
                break

    if not target or not target.exists():
        raise HTTPException(status_code=404, detail=f"Diagnostic frame image '{file_type}' for frame #{frame_idx} not found")

    media_type = "image/png" if target.suffix == ".png" else "image/jpeg"
    return FileResponse(path=str(target), media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    # Production entrypoint with reload=False
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
