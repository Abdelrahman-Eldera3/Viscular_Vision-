"""
PAD Angiography Vessel Measurement Pipeline - Frame Selection Architecture
========================================================================
Standalone, parallel pipeline for peripheral artery disease (PAD)
angiography vessel measurement using quality-gated frame selection instead
of continuous multi-frame identity tracking.

Key Design Principles:
  1. Reuses shared mathematical and segmentation primitives from pad_pipeline_colab.
  2. Eliminates temporal tracking state machines (VesselTracker) for clinical numbers.
  3. Evaluates candidate frames against a diagnostic eligibility gate.
  4. Ranks eligible frames via a multi-component quality index (contrast, sharpness, stability).
  5. Selects the top-k clean frames per vessel and aggregates worst-case stenosis.
  6. Computes the Angiographic Severity Index (ASI) incorporating lesion length.
  7. Evaluates outcomes in strict priority order:
     - Priority 1: INSUFFICIENT_DIAGNOSTIC_QUALITY (occlusion_flag=False, numeric fields=None)
     - Priority 2: LIKELY_OCCLUSION_LUMEN_UNRESOLVED (occlusion_flag=True, numeric fields=None)
     - Priority 3: Standard measurement (NORMAL, MILD, MODERATE, SEVERE STENOSIS)
"""

import os
import json
import sys
import argparse
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import cv2
from skimage.morphology import skeletonize
import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────────────────
#  REUSE EXISTING BUILDING BLOCKS (Unmodified Import)
# ──────────────────────────────────────────────────────────
from pad_pipeline_colab import (
    get_model,
    _get_device,
    preprocess,
    postprocess,
    calculate_stenosis_qca,
    _measure_orthogonal_lumen_profile,
    extract_vessel_centerline,
    qualify_vessels,
    _merge_collinear_vessels,
    _find_contrast_bridge_path,
    _get_fragment_endpoint_and_direction,
    _is_diffuse_collateral_mesh,
    compute_contrast_score,
    get_calibration_for_video,
    WIBMER_ALPHA,
    WIBMER_BETA,
    CUTOFF_MAX_TOP_RATIO,
    CUTOFF_MAX_BOT_RATIO,
    CUTOFF_MIN_HEIGHT_RATIO,
    CUTOFF_MIN_RVD_PX,
    DS_TOTAL_OCCLUSION_THRESHOLD,
    DS_SEVERE_THRESHOLD,
    DS_MODERATE_THRESHOLD,
    IMAGE_SIZE,
    BATCH_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pad_frameselect")

# ──────────────────────────────────────────────────────────
#  TUNABLE PARAMETER DECLARATIONS
#  (Untuned starting defaults requiring clinical validation before diagnostic use)
# ──────────────────────────────────────────────────────────
K_BEST_FRAMES: int = 5
W_CONTRAST: float = 1.0 / 3.0
W_SHARPNESS: float = 1.0 / 3.0
W_STABILITY: float = 1.0 / 3.0

NOMINAL_REFERENCE_LENGTH_MM: float = 20.0       # Reference lesion length for Poiseuille resistance scaling
LENGTH_FACTOR_COEFF: float = 0.15               # Severity growth factor per nominal reference excess
NEIGHBOR_DS_DIFF_NORM_DIVISOR: float = 25.0     # Normalization divisor for DS% neighbor deviation
FOV_EDGE_MARGIN_RATIO: float = 0.05             # 5% margin from FOV boundaries for stenosis location
DIAG_MIN_HEIGHT_RATIO: float = 0.60             # Diagnostic gate: height >= 60% of vessel's cine peak height
DIAG_MIN_AREA_RATIO: float = 0.45               # Diagnostic gate: area >= 45% of vessel's cine peak area
MIN_RESOLVABLE_LUMEN_PX: float = 2.0            # Minimum orthogonal lumen width (px) considered resolvable

# Vessel Significance Gate (Tunable starting defaults; flagged as untuned, requiring validation against confirmed real vessels vs artifacts):
MIN_VESSEL_APPEARANCE_FRAMES: int = 5         # Cluster must appear in at least this many raw frames across cine to qualify as a vessel
MIN_VESSEL_PEAK_HEIGHT_PX: float = 60.0       # Minimum peak bounding height in pixels (flagged as untuned)
MIN_VESSEL_PEAK_HEIGHT_RATIO: float = 0.05    # Alternative minimum peak height ratio: 0.05 * frame_height (flagged as untuned)
MIN_VESSEL_PEAK_AREA_PX: float = 300.0        # Minimum peak area in pixels^2 (flagged as untuned)
MIN_VESSEL_PEAK_CONTRAST: float = 5.0         # Minimum peak contrast score across cine to qualify as contrast-enhanced vessel
CONTRAST_SCORE_MIN_THRESHOLD: float = 5.0     # Diagnostic hard gate contrast floor to reject bone cortex/shadows

# Segment-interruption analysis (mid-vessel gap between two aligned stumps).
# ALL values below are untuned starting defaults — validate against clinician-confirmed
# occlusions (e.g. video 141) AND confirmed-normal fragmented vessels before clinical use.
INTERRUPTION_MIN_GAP_PX: float = 12.0           # Gaps shorter than this are ordinary segmentation jitter
INTERRUPTION_MIN_GAP_RVD_MULT: float = 1.0      # ...and must also be >= 1.0 x local vessel width
INTERRUPTION_MAX_GAP_RATIO: float = 0.35        # Gaps longer than 35% of frame height are too ambiguous to call
INTERRUPTION_MAX_LATERAL_OFFSET_MULT: float = 2.0  # Stump exit/entry lateral offset <= 2 x vessel width
INTERRUPTION_MIN_LATERAL_OFFSET_PX: float = 25.0   # ...with a floor for very thin vessels
INTERRUPTION_MIN_FRAMES: int = 5                # Gap must be seen in at least this many frames
INTERRUPTION_MIN_PERSISTENCE: float = 0.50      # ...and in >= 50% of frames where both stumps are visible
SEGMENTATION_GAP_CONTRAST_RATIO: float = 0.60   # gap contrast >= 60% of vessel contrast -> dye present, model dropout

# Evidence triage for weakly supported vessels. Untuned defaults — chosen on 12 videos, validate on new ones.
MIN_EVIDENCE_FRAMES_MAX: int = 7         # a vessel needs >= 7 frames passing the diagnostic gate ...
MIN_EVIDENCE_FRACTION: float = 0.30      # ... or 30% of the video's frames, whichever is smaller (short cines)
MIN_EVIDENCE_FRAMES_FLOOR: int = 3       # never below 3 frames
DUPLICATE_FOOTPRINT_OVERLAP: float = 0.50  # weak vessel lying >= 50% inside a reliable vessel's footprint = duplicate
DUPLICATE_FOOTPRINT_DILATE_PX: int = 7
# Smart temporal linking (v1.2): pieces of ONE artery seen at DIFFERENT times are merged only if they overlap
# in space (e.g. the bolus front inside the later, fully filled artery) or one continues the other along its
# own direction (straight or curved) across a reasonable gap and within a short time. Untuned defaults.
ENABLE_SMART_TEMPORAL_MERGE: bool = True
ENABLE_PATH_CONSISTENT_TRACKING: bool = True   # first-pass grouping: join a vessel only on its path, or next to it recently
TRACK_MIN_PATH_SHARE: float = 0.60             # >= 60% of the new piece's centerline lies inside the vessel seen so far
# Delayed filling (v1.2): a vessel that STARTS mid-image is called a retrograde occlusion only if it filled clearly
# later than the other vessels at the same level (reconstitution through collaterals). 48 (true occlusion): +25 frames;
# 631 (branch whose connection was not segmented): +8 frames; normal vessels: 0 to +3. Based on 2 cases: validate.
ENABLE_DELAYED_FILLING_CHECK: bool = True
# Patency seen elsewhere (v1.2): no cutoff is called where ANOTHER reliable vessel is seen passing through the same
# point and continuing beyond it at some other time in the run (123: the main artery was fully visible in frames
# 37-57; after patient motion its upper half alone formed a separate "vessel" that ended mid-image).
ENABLE_PATENCY_ELSEWHERE_CHECK: bool = True
PATENCY_MIN_GOOD_FRAMES: int = 7          # only reliable vessels count (weak pieces must never cancel an occlusion)
PATENCY_MIN_CONTINUATION_PX: int = 100    # the other vessel must continue >= 100 px beyond the end point
RETRO_MIN_DELAY_SEC: float = 1.0
VIDEO_FPS: float = 15.0                        # set from the video when known
LINK_MIN_OVERLAP: float = 0.30          # >= 30% of the smaller piece lies on the other piece
LINK_MAX_GAP_RATIO: float = 0.35        # continuation gap <= 35% of the image height (same limit as interruptions)
LINK_MAX_TIME_GAP_FRAMES: int = 10      # a continuation must appear within 10 frames of the other piece
LINK_MIN_LATERAL_TOL_PX: float = 25.0   # continuation must land within max(25 px, 2 x width) (+10% of the gap)

# Main-artery scope (v1.2): the report covers the main arteries only (pelvis 1, thigh 2, leg 3 -> at most 3).
ENABLE_MAIN_ARTERY_FILTER: bool = True
MAX_MAIN_ARTERIES: int = 3
MAIN_MIN_REL_CALIBER: float = 0.40       # caliber >= 40% of the thickest vessel in the SAME video (leg arteries are thinner)
MAIN_MIN_SPAN: float = 0.25              # visible over >= 25% of the image height
MAIN_KEEP_LINKED_BRANCHES: bool = True   # a RELIABLE branch linked to a main artery counts as main (profunda, leg arteries)
GAP_REVIEW_MIN_GOOD_FRAMES: int = 2      # mask broke along the vessel in >= 2 diagnostic frames -> review flag
ENABLE_CUTOFF_GATED_EXTENT: bool = True  # cutoff rule reads a vessel's extent from diagnostic-quality pieces only
CUTOFF_MIN_GATED_PIECES: int = 5         # ... when at least 5 exist; otherwise it falls back to all pieces
SIDE_BY_SIDE_FRAME_WINDOW: int = 2      # compare with reliable vessels within +/- 2 frames
SIDE_BY_SIDE_MIN_VERTICAL: float = 0.50  # weak piece shares >= 50% of its height with a reliable vessel ...
SIDE_BY_SIDE_MAX_OVERLAP: float = 0.30   # ... but <= 30% of its pixels lie on that vessel -> runs BESIDE it (branch)
SIDE_BY_SIDE_MIN_FRACTION: float = 0.50  # branch-like in >= 50% of comparable frames -> never removed as a duplicate
SIDE_BY_SIDE_MAX_GAP_PX: int = 25        # "beside" means within 25 px of the reliable vessel (a branch touches its trunk)
ENABLE_BRANCH_DIVERGENCE: bool = False   # OFF: on the full 12-video run (2026-09-26) this rule removed the REAL
                                         # branch in 39 and did not remove the extras in 123/141 (it had been checked
                                         # on top-k frames only). Kept for future work, disabled by default.
BRANCH_MIN_DIVERGENCE_PX: float = 10.0   # a branch must move AWAY from the trunk along its length by >= 10 px
BRANCH_MIN_DIVERGENCE_WIDTH_MULT: float = 0.5  # ... and by >= half its own width (a glued parallel copy does not)
BONE_EDGE_HIDE_RIM: float = 1.0          # bright rim STRONGER than the dark core: dye alone cannot do that -> not an
                                         # artery (subtraction edge of bone/tissue). Real arteries measured 0.22-0.65
                                         # (0.65 = a real branch beside a dark trunk); bone in 110 = 3.26.
EDGE_ARTIFACT_RIM_RATIO: float = 0.40    # bright rim / dark core >= 0.40 -> looks like a subtraction edge.
                                         # On real data: real vessels 0.22-0.34, junk 0.42-0.72 (123, 48). The margin is
                                         # too thin to DELETE on, so it only FLAGS a low-evidence vessel for review.

# Bifurcation (trunk/branch) geometry. Untuned defaults.
BIFURCATION_MAX_JUNCTION_PX: float = 45.0       # Hard ceiling on takeoff-to-parent distance
BIFURCATION_JUNCTION_WIDTH_MULT: float = 2.0    # Takeoff must lie within 2 x parent width of the parent centerline
BIFURCATION_MIN_ANGLE_DEG: float = 10.0         # Real divergence: >= 10 deg ...
BIFURCATION_MIN_SEPARATION_GROWTH: float = 1.5  # ... or downstream separation >= 1.5 x junction distance (+10px)

# Spatial clustering thresholds (reused from legacy tracker; require independent validation in this context):
SPATIAL_CLUSTER_CX_TOLERANCE_PX: float = 60.0   # Horizontal centroid tolerance for spatial grouping
SPATIAL_CLUSTER_CX_RATIO: float = 0.070         # Horizontal centroid tolerance as ratio of frame width
SPATIAL_CLUSTER_IOU_THRESHOLD: float = 0.20     # Bounding box IoU threshold for spatial grouping


@dataclass
class CandidateDetection:
    """Detection for a single vessel component in a single frame."""
    frame_idx: int
    mask_region: np.ndarray
    top: int
    bot: int
    left: int
    right: int
    w: int
    h: int
    cx: int
    cy: int
    area: float
    continuity: float
    # QCA metrics computed if available
    qca: Optional[Dict[str, Any]] = None
    lumen_widths: List[float] = field(default_factory=list)
    lumen_ys: List[int] = field(default_factory=list)
    sten_x: Optional[float] = None
    sten_y: Optional[float] = None
    DS_pct: Optional[float] = None
    MLD_px: Optional[float] = None
    RVD_px: Optional[float] = None
    stenosis_length_px: Optional[float] = None
    # Gating and ranking scores
    passed_hard_gate: bool = False
    gate_rejection_reason: str = ""
    contrast_score_raw: float = 0.0
    sharpness_score_raw: float = 0.0
    stability_score_raw: float = 0.0
    total_score: float = 0.0
    # Mid-vessel gap evidence attached to the UPPER stump of a gap in this frame
    gap_records: List[Dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
#  SCORING SUB-FUNCTIONS
# ──────────────────────────────────────────────────────────

def compute_contrast_component(
    gray_frame: np.ndarray,
    vessel_mask: np.ndarray,
    baseline_gray: Optional[np.ndarray] = None,
) -> float:
    """Computes opacification contrast relative to background/baseline."""
    return float(compute_contrast_score(gray_frame, vessel_mask, baseline_frame=baseline_gray))


def compute_sharpness_component(gray_frame: np.ndarray) -> float:
    """
    Computes image sharpness using the variance of the Laplacian.
    Higher variance reflects sharper vessel contours and less motion blur.
    """
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def compute_neighbor_stability_component(
    candidate_ds_by_frame: Dict[int, float],
    frame_idx: int,
    norm_divisor: float = NEIGHBOR_DS_DIFF_NORM_DIVISOR,
) -> float:
    """
    Computes neighbor stability score based on the deviation of this frame's raw DS%
    from the median DS% of its immediate temporal neighbors (frame - 1, frame + 1)
    that also passed the hard gate.
    Smaller difference = higher stability score (frame is not an isolated spike).
    Normalized to [0.0, 1.0].
    """
    this_ds = candidate_ds_by_frame.get(frame_idx)
    if this_ds is None:
        return 0.0

    neighbor_ds = []
    for offset in (-1, 1):
        n_idx = frame_idx + offset
        if n_idx in candidate_ds_by_frame:
            neighbor_ds.append(candidate_ds_by_frame[n_idx])

    if not neighbor_ds:
        return 1.0  # No adjacent gated neighbors; neutral/default stability

    med_neighbor = float(np.median(neighbor_ds))
    diff = abs(this_ds - med_neighbor)
    stability = 1.0 - min(1.0, diff / max(1e-3, norm_divisor))
    return float(np.clip(stability, 0.0, 1.0))


# ──────────────────────────────────────────────────────────
#  SPATIAL GROUPING (Stateless Region Clustering)
# ──────────────────────────────────────────────────────────

def _compute_bbox_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes IoU between two (top, bot, left, right) bounding boxes."""
    t1, b1, l1, r1 = box1
    t2, b2, l2, r2 = box2

    inter_top = max(t1, t2)
    inter_bot = min(b1, b2)
    inter_left = max(l1, l2)
    inter_right = min(r1, r2)

    if inter_bot <= inter_top or inter_right <= inter_left:
        return 0.0

    inter_area = (inter_bot - inter_top) * (inter_right - inter_left)
    area1 = (b1 - t1) * (r1 - l1)
    area2 = (b2 - t2) * (r2 - l2)
    union_area = area1 + area2 - inter_area
    return float(inter_area / max(1.0, union_area))


def _ensure_full_mask(d: "CandidateDetection", orig_w: int, orig_h: int) -> np.ndarray:
    """Returns a full-frame mask for a detection (bbox fill as fallback)."""
    m = d.mask_region
    if m is None or m.shape[0] < 2 or m.shape[1] < 2:
        m = np.zeros((orig_h, orig_w), dtype=np.uint8)
        m[max(0, d.top):min(orig_h, d.bot), max(0, d.left):min(orig_w, d.right)] = 1
    return m


def _measure_stump_gap(
    upper: "CandidateDetection",
    lower: "CandidateDetection",
    gray_frame: Optional[np.ndarray],
    orig_w: int,
    orig_h: int,
) -> Optional[Dict[str, Any]]:
    """
    Measures the gap between two aligned pieces of ONE vessel seen in the SAME frame
    (upper piece ends, lower piece starts below it on the same axis).

    Geometry decides "is this the same vessel?" (alignment, gap length).
    Contrast INSIDE the gap decides what the gap is:
      SEGMENTATION_GAP : dye clearly present in the gap (>= SEGMENTATION_GAP_CONTRAST_RATIO of the
                         vessel's own contrast) -> lumen patent, the model simply dropped pixels.
      FAINT_FLOW       : some dye, well below the vessel's own level -> suspected subtotal occlusion.
      NO_FLOW          : no dye above the bone/tissue floor -> suspected total occlusion.
      STUMPS_NOT_OPACIFIED : one of the stumps is itself poorly opacified (inflow/washout frame);
                         this frame cannot be used as evidence either way.
    Returns None when the two pieces are not an aligned, measurable gap.
    """
    if upper.top > lower.top:
        upper, lower = lower, upper
    m_up = _ensure_full_mask(upper, orig_w, orig_h)
    m_lo = _ensure_full_mask(lower, orig_w, orig_h)
    try:
        exit_pt, _ = _get_fragment_endpoint_and_direction(m_up, upper.top, upper.bot, upper.left, upper.right, mode="exit")
        entry_pt, _ = _get_fragment_endpoint_and_direction(m_lo, lower.top, lower.bot, lower.left, lower.right, mode="entry")
    except Exception:
        exit_pt = (float(upper.cx), float(upper.bot))
        entry_pt = (float(lower.cx), float(lower.top))

    # Gap LENGTH comes from the real mask edges. The endpoint helper places its points ~30-40px
    # inside each fragment (it averages the tail to get a stable direction), which is right for
    # alignment and for seeding the bridge path, but would overstate the gap length.
    vertical_gap = float(lower.top - upper.bot)
    if vertical_gap <= 0:
        return None  # pieces overlap vertically -> not a gap
    gap_px = float(np.hypot(float(entry_pt[0]) - float(exit_pt[0]), vertical_gap))

    widths = [v for v in (upper.RVD_px, lower.RVD_px) if v is not None and v > 0]
    local_w = float(np.median(widths)) if widths else float(np.median([upper.w, lower.w]))

    lateral = abs(float(entry_pt[0]) - float(exit_pt[0]))
    if lateral > max(INTERRUPTION_MIN_LATERAL_OFFSET_PX, INTERRUPTION_MAX_LATERAL_OFFSET_MULT * local_w):
        return None  # not on the same axis -> two different vessels
    if gap_px > INTERRUPTION_MAX_GAP_RATIO * orig_h:
        return None  # too long to call it one interrupted vessel
    if gap_px < max(INTERRUPTION_MIN_GAP_PX, INTERRUPTION_MIN_GAP_RVD_MULT * local_w):
        return None  # ordinary segmentation jitter

    record: Dict[str, Any] = {
        "frame_idx": int(upper.frame_idx),
        "gap_px": round(gap_px, 1),
        "gap_top_y": float(upper.bot),
        "gap_bottom_y": float(lower.top),
        "local_width_px": round(local_w, 1),
        "gap_contrast": None,
        "vessel_contrast": None,
        "contrast_ratio": None,
        "classification": "UNKNOWN_NO_IMAGE",
    }
    if gray_frame is None:
        return record

    c_up = upper.contrast_score_raw if upper.contrast_score_raw > 0 else compute_contrast_score(gray_frame, m_up)
    c_lo = lower.contrast_score_raw if lower.contrast_score_raw > 0 else compute_contrast_score(gray_frame, m_lo)
    vessel_contrast = float(min(c_up, c_lo))
    record["vessel_contrast"] = round(vessel_contrast, 2)
    if vessel_contrast < CONTRAST_SCORE_MIN_THRESHOLD:
        record["classification"] = "STUMPS_NOT_OPACIFIED"
        return record

    _, gap_contrast = _find_contrast_bridge_path(gray_frame, exit_pt, entry_pt, mask1=m_up, mask2=m_lo)
    gap_contrast = float(gap_contrast)
    ratio = gap_contrast / vessel_contrast if vessel_contrast > 0 else 0.0
    record["gap_contrast"] = round(gap_contrast, 2)
    record["contrast_ratio"] = round(ratio, 3)
    if ratio >= SEGMENTATION_GAP_CONTRAST_RATIO:
        record["classification"] = "SEGMENTATION_GAP"
    elif gap_contrast >= CONTRAST_SCORE_MIN_THRESHOLD:
        record["classification"] = "FAINT_FLOW"
    else:
        record["classification"] = "NO_FLOW"
    return record


def _merge_same_frame_collinear_candidates(
    frame_dets: List[CandidateDetection],
    orig_w: int,
    orig_h: int,
    gray_frame: Optional[np.ndarray] = None,
) -> List[CandidateDetection]:
    """
    Merges disconnected collinear vessel fragments within a single frame using
    _merge_collinear_vessels() before assigning to temporal/spatial clusters.
    """
    if len(frame_dets) <= 1:
        return frame_dets

    max_gap_px = int(0.08 * orig_h)
    vessels_dict = []
    for d in frame_dets:
        m = d.mask_region
        if m is None or m.shape[0] < 2 or m.shape[1] < 2:
            m = np.zeros((orig_h, orig_w), dtype=np.uint8)
            m[max(0, d.top):min(orig_h, d.bot), max(0, d.left):min(orig_w, d.right)] = 1
        vessels_dict.append({
            "top": d.top,
            "bot": d.bot,
            "left": d.left,
            "right": d.right,
            "w": d.w,
            "h": d.h,
            "cx": d.cx,
            "cy": d.cy,
            "area": d.area,
            "continuity": d.continuity,
            "mask_region": m,
            "_det": d,
        })

    merged_dicts = _merge_collinear_vessels(vessels_dict, max_gap_px=max_gap_px, gray_frame=gray_frame)

    merged_candidates: List[CandidateDetection] = []
    for vd in merged_dicts:
        orig_det = vd.get("_det")
        if orig_det is not None and vd["top"] == orig_det.top and vd["bot"] == orig_det.bot and vd["left"] == orig_det.left:
            merged_candidates.append(orig_det)
        else:
            # Reconstruct merged CandidateDetection combining properties
            constituents = [
                d for d in frame_dets 
                if vd["top"] <= d.top and d.bot <= vd["bot"] and vd["left"] <= d.left and d.right <= vd["right"]
            ]
            ds_candidates = [d.DS_pct for d in constituents if d.DS_pct is not None]
            best_ds = max(ds_candidates) if ds_candidates else (orig_det.DS_pct if orig_det else None)
            mld_candidates = [d.MLD_px for d in constituents if d.MLD_px is not None]
            best_mld = min(mld_candidates) if mld_candidates else (orig_det.MLD_px if orig_det else None)
            rvd_candidates = [d.RVD_px for d in constituents if d.RVD_px is not None]
            best_rvd = max(rvd_candidates) if rvd_candidates else (orig_det.RVD_px if orig_det else None)
            c_score = max((d.contrast_score_raw for d in constituents), default=(orig_det.contrast_score_raw if orig_det else 20.0))
            f_idx = constituents[0].frame_idx if constituents else (orig_det.frame_idx if orig_det else 0)

            gap_recs: List[Dict[str, Any]] = []
            ordered = sorted(constituents, key=lambda d: d.top)
            for up_d, lo_d in zip(ordered, ordered[1:]):
                rec = _measure_stump_gap(up_d, lo_d, gray_frame, orig_w, orig_h)
                if rec is not None:
                    gap_recs.append(rec)

            merged_candidates.append(CandidateDetection(
                gap_records=gap_recs,
                frame_idx=f_idx,
                mask_region=vd["mask_region"],
                top=vd["top"],
                bot=vd["bot"],
                left=vd["left"],
                right=vd["right"],
                w=vd["w"],
                h=vd["h"],
                cx=vd["cx"],
                cy=vd["cy"],
                area=float(vd["area"]),
                continuity=float(vd["continuity"]),
                DS_pct=best_ds,
                MLD_px=best_mld,
                RVD_px=best_rvd,
                stenosis_length_px=float(vd["h"]),
                contrast_score_raw=c_score,
            ))

    return merged_candidates


def _small_mask(d: CandidateDetection, orig_w: int, orig_h: int, scale: int = 4) -> np.ndarray:
    cached = getattr(d, "_small_mask_cache", None)
    if cached is not None:
        return cached
    m = _ensure_full_mask(d, orig_w, orig_h)
    sm = cv2.resize((m > 0).astype(np.uint8), (orig_w // scale, orig_h // scale), interpolation=cv2.INTER_NEAREST) > 0
    try:
        d._small_mask_cache = sm
    except Exception:
        pass
    return sm


def _centerline_share(piece_small: np.ndarray, footprint_small: np.ndarray) -> float:
    """Share of a piece's centerline lying inside a footprint (both at the same reduced scale)."""
    if not piece_small.any() or not footprint_small.any():
        return 0.0
    ys, xs = np.where(piece_small)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sk = skeletonize(piece_small[y0:y1, x0:x1])
    if not sk.any():
        return 0.0
    fp = cv2.dilate(footprint_small.astype(np.uint8), np.ones((3, 3), np.uint8))[y0:y1, x0:x1] > 0
    return float((sk & fp).sum()) / float(sk.sum())


def _cluster_footprint(dets: List[CandidateDetection], orig_w: int, orig_h: int, scale: int = 4) -> np.ndarray:
    """Union of a cluster's masks at 1/scale resolution (bool)."""
    h, w = orig_h // scale, orig_w // scale
    fp = np.zeros((h, w), np.uint8)
    for d in dets:
        m = _ensure_full_mask(d, orig_w, orig_h)
        fp |= (cv2.resize((m > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return fp > 0


def _temporal_link_ok(
    dets1: List[CandidateDetection], dets2: List[CandidateDetection], orig_w: int, orig_h: int, scale: int = 4,
) -> Tuple[bool, str]:
    """
    May two pieces seen at DIFFERENT times be the same artery?
      overlap      : they cover the same place (bolus front inside the later, fully filled artery);
      continuation : the lower piece starts where the upper piece's own direction leads (straight or curved),
                     across a gap <= LINK_MAX_GAP_RATIO of the image, within LINK_MAX_TIME_GAP_FRAMES frames.
    Anything else (different places, not on one line, or far apart in time) is NOT merged.
    """
    fp1 = _cluster_footprint(dets1, orig_w, orig_h, scale); fp2 = _cluster_footprint(dets2, orig_w, orig_h, scale)
    if not fp1.any() or not fp2.any():
        return False, "empty"
    k = np.ones((3, 3), np.uint8)
    d1 = cv2.dilate(fp1.astype(np.uint8), k) > 0; d2 = cv2.dilate(fp2.astype(np.uint8), k) > 0
    small = min(int(fp1.sum()), int(fp2.sum()))
    if small and max((fp1 & d2).sum(), (fp2 & d1).sum()) / small >= LINK_MIN_OVERLAP:
        return True, "overlap"

    r1 = np.where(fp1.any(axis=1))[0]; r2 = np.where(fp2.any(axis=1))[0]
    (up, up_dets, lo, lo_dets) = (fp1, dets1, fp2, dets2) if r1.min() <= r2.min() else (fp2, dets2, fp1, dets1)
    ru = np.where(up.any(axis=1))[0]; rl = np.where(lo.any(axis=1))[0]
    gap = (rl.min() - ru.max()) * scale
    if gap < -0.05 * orig_h or gap > LINK_MAX_GAP_RATIO * orig_h:
        return False, f"gap {gap:.0f}px"
    fu = sorted({d.frame_idx for d in up_dets}); fl = sorted({d.frame_idx for d in lo_dets})
    t_gap = max(0, max(fu[0], fl[0]) - min(fu[-1], fl[-1]))
    if t_gap > LINK_MAX_TIME_GAP_FRAMES:
        return False, f"{t_gap} frames apart"
    tail = [r for r in ru if r >= ru.max() - 80 // scale]               # direction of the upper piece near its end
    xs = np.array([np.where(up[r])[0].mean() for r in tail]) * scale; ys = np.array(tail, float) * scale
    widths = [np.count_nonzero(up[r]) * scale for r in tail]
    slope = float(np.polyfit(ys, xs, 1)[0]) if len(tail) >= 3 and np.ptp(ys) > 0 else 0.0
    x_end = float(xs[-1]); y_end = float(ys[-1])
    head = [r for r in rl if r <= rl.min() + 20 // scale]
    x_start = float(np.mean([np.where(lo[r])[0].mean() for r in head])) * scale; y_start = float(rl.min() * scale)
    predicted = x_end + slope * (y_start - y_end)
    tol = max(LINK_MIN_LATERAL_TOL_PX, 2.0 * float(np.median(widths))) + 0.10 * max(0.0, gap)
    miss = min(abs(x_start - predicted), abs(x_start - x_end))
    return bool(miss <= tol), f"continuation miss {miss:.0f}px (tol {tol:.0f})"


def group_vessels_spatially(
    detections_by_frame: List[List[CandidateDetection]],
    orig_w: int,
    orig_h: int,
    gray_frames: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[int, List[CandidateDetection]]:
    """
    Groups frame detections into distinct physical vessel clusters across the cine
    using lateral proximity and bounding box overlap.

    Pre-merges same-frame collinear fragments with _merge_collinear_vessels(),
    prevents intra-frame duplicate assignment, tracks vertical envelope, and unifies
    temporal bolus propagation phases (early inflow and peak filling of the same artery).
    """
    clusters: List[Dict[str, Any]] = []
    # clusters entry schema:
    # {
    #    "cluster_id": int,
    #    "detections": List[CandidateDetection],
    #    "median_cx": float,
    #    "min_top": int,
    #    "max_bot": int,
    #    "last_bbox": Tuple[int, int, int, int],
    # }

    cx_tol = max(SPATIAL_CLUSTER_CX_TOLERANCE_PX, SPATIAL_CLUSTER_CX_RATIO * orig_w)

    for raw_frame_dets in detections_by_frame:
        if not raw_frame_dets:
            continue
        f_idx = raw_frame_dets[0].frame_idx
        gf = gray_frames.get(f_idx) if gray_frames is not None else None

        # Same-frame pre-merge step: merge collinear fragments aligned on shared vessel axis
        frame_dets = _merge_same_frame_collinear_candidates(raw_frame_dets, orig_w, orig_h, gray_frame=gf)

        # Sort detections in frame by area descending so dominant trunks get priority
        sorted_dets = sorted(frame_dets, key=lambda d: d.area, reverse=True)
        matched_in_frame: Set[int] = set()

        for det in sorted_dets:
            det_box = (det.top, det.bot, det.left, det.right)
            best_c_idx = -1
            best_c_dist = float("inf")

            for c_idx, cluster in enumerate(clusters):
                if c_idx in matched_in_frame:
                    continue  # Each cluster can contain at most one detection per frame

                # Calculate centroid horizontal distance
                dx = abs(det.cx - cluster["median_cx"])
                iou = _compute_bbox_iou(det_box, cluster["last_bbox"])

                # Check vertical overlap with cluster accumulated vertical envelope
                v_overlap = max(0, min(det.bot, cluster["max_bot"]) - max(det.top, cluster["min_top"]))
                v_ratio = v_overlap / max(1.0, det.h)

                old_ok = (dx <= cx_tol and v_ratio >= 0.15) or iou >= SPATIAL_CLUSTER_IOU_THRESHOLD or dx <= (0.50 * cx_tol)
                if ENABLE_PATH_CONSISTENT_TRACKING:
                    # Lateral position alone is not enough (631: a bone edge 26 frames later joined an artery;
                    # 110: three unrelated pieces in one column). Join only on the vessel's own path, or next to it
                    # while it is still being seen (the bolus front advancing).
                    recent = (f_idx - cluster.get("last_frame", f_idx)) <= LINK_MAX_TIME_GAP_FRAMES
                    accept = recent and old_ok
                    if not accept and dx <= cx_tol * 2:
                        accept = _centerline_share(_small_mask(det, orig_w, orig_h), cluster["fp"]) >= TRACK_MIN_PATH_SHARE
                else:
                    accept = old_ok
                if accept:
                    if dx < best_c_dist:
                        best_c_dist = dx
                        best_c_idx = c_idx

            if best_c_idx >= 0:
                c = clusters[best_c_idx]
                c["detections"].append(det)
                c["median_cx"] = float(np.median([d.cx for d in c["detections"]]))
                c["min_top"] = min(c["min_top"], det.top)
                c["max_bot"] = max(c["max_bot"], det.bot)
                c["last_bbox"] = det_box
                c["last_frame"] = f_idx
                c["fp"] = c["fp"] | _small_mask(det, orig_w, orig_h)
                matched_in_frame.add(best_c_idx)
            else:
                new_id = len(clusters) + 1
                clusters.append({
                    "cluster_id": new_id,
                    "detections": [det],
                    "median_cx": float(det.cx),
                    "min_top": det.top,
                    "max_bot": det.bot,
                    "last_bbox": det_box,
                    "last_frame": f_idx,
                    "fp": _small_mask(det, orig_w, orig_h).copy(),
                })
                matched_in_frame.add(len(clusters) - 1)

    # Post-merging pass: Merge clusters that represent the same physical vessel column across time
    # (e.g. early bolus wavefront inflow + peak full fill of the same trunk)
    merged_clusters: List[Dict[str, Any]] = []
    used: Set[int] = set()

    for i, c1 in enumerate(clusters):
        if i in used:
            continue
        dets1 = list(c1["detections"])
        cx1 = float(np.median([d.cx for d in dets1]))
        frames1 = set(d.frame_idx for d in dets1)

        for j in range(i + 1, len(clusters)):
            if j in used:
                continue
            c2 = clusters[j]
            dets2 = c2["detections"]
            cx2 = float(np.median([d.cx for d in dets2]))
            frames2 = set(d.frame_idx for d in dets2)

            dx = abs(cx1 - cx2)
            common_frames = len(frames1.intersection(frames2))
            common_ratio = common_frames / max(1.0, min(len(frames1), len(frames2)))

            # Check if co-occurring detections are vertically non-overlapping (stenotic gap / flow interruption)
            is_vertically_stacked = True
            if common_frames > 0:
                f_map1 = {}
                for d in dets1: f_map1.setdefault(d.frame_idx, []).append(d)
                f_map2 = {}
                for d in dets2: f_map2.setdefault(d.frame_idx, []).append(d)
                for cf in frames1.intersection(frames2):
                    for d1 in f_map1[cf]:
                        for d2 in f_map2[cf]:
                            v_overlap = max(0, min(d1.bot, d2.bot) - max(d1.top, d2.top))
                            overlap_ratio = v_overlap / max(1.0, min(d1.h, d2.h))
                            if overlap_ratio > 0.20:
                                is_vertically_stacked = False
                                break
                        if not is_vertically_stacked:
                            break
                    if not is_vertically_stacked:
                        break

            # (a) Seen TOGETHER in the same frames, one above the other (stumps around a gap / occlusion): merged as before.
            # (b) Seen at DIFFERENT times: previously merged on lateral position alone (is_vertically_stacked defaults to
            #     True when they never co-occur), which glued unrelated pieces together (video 110). Now they must overlap
            #     in space or continue each other along one line, close in time.
            co_stacked = common_frames > 0 and is_vertically_stacked
            temporal = common_ratio <= 0.15 or common_frames == 0
            link_ok = co_stacked
            if not link_ok and dx <= cx_tol and temporal:
                if ENABLE_SMART_TEMPORAL_MERGE:
                    link_ok, why = _temporal_link_ok(dets1, dets2, orig_w, orig_h)
                    if not link_ok:
                        logger.debug(f"[SpatialGrouping] NOT merging #{c2['cluster_id']} into #{c1['cluster_id']}: {why}")
                else:
                    link_ok = True
            if dx <= cx_tol and link_ok:
                # Two aligned pieces visible TOGETHER in the same frames: the space between them
                # is a real finding (occlusion) or a segmentation dropout — measure it, never discard it.
                if common_frames > 0 and is_vertically_stacked:
                    for cf in frames1.intersection(frames2):
                        gf_cf = gray_frames.get(cf) if gray_frames is not None else None
                        for d1 in f_map1[cf]:
                            for d2 in f_map2[cf]:
                                rec = _measure_stump_gap(d1, d2, gf_cf, orig_w, orig_h)
                                if rec is not None:
                                    (d1 if d1.top <= d2.top else d2).gap_records.append(rec)
                dets1.extend(dets2)
                frames1.update(frames2)
                cx1 = float(np.median([d.cx for d in dets1]))
                used.add(j)
                logger.debug(f"[SpatialGrouping] Merged temporal/collinear sub-cluster #{c2['cluster_id']} into main cluster #{c1['cluster_id']} (dx={dx:.1f}px, stacked={is_vertically_stacked})")

        c1["detections"] = dets1
        c1["median_cx"] = cx1
        merged_clusters.append(c1)
        used.add(i)

    # Re-index clusters cleanly 1..N
    return {idx + 1: c["detections"] for idx, c in enumerate(merged_clusters)}


# ──────────────────────────────────────────────────────────
#  VESSEL SIGNIFICANCE GATE (PRE-FILTERING NOISE & FRAGMENTS)
# ──────────────────────────────────────────────────────────

def _rotated_fill_ratio(d: CandidateDetection, orig_w: int, orig_h: int) -> float:
    """area / area of the minimum rotated rectangle around the detection's mask."""
    m = d.mask_region
    if m is None or m.shape[0] < 2 or m.shape[1] < 2:
        return d.area / max(1.0, float(d.w * d.h))
    crop = (m[max(0, d.top):min(orig_h, d.bot), max(0, d.left):min(orig_w, d.right)] > 0).astype(np.uint8)
    pts = cv2.findNonZero(crop)
    if pts is None or len(pts) < 5:
        return d.area / max(1.0, float(d.w * d.h))
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    return float(len(pts)) / max(1.0, float(rw * rh))


def filter_significant_vessels(
    clusters: Dict[int, List[CandidateDetection]],
    orig_w: int,
    orig_h: int,
) -> Dict[int, List[CandidateDetection]]:
    """
    Vessel Significance Gate (Section 2):
    Applied immediately after spatial grouping and BEFORE any per-frame hard-gate
    filtering, ranking scoring, or outcome evaluation to prevent small segmentation
    artifacts/fragments from being promoted to a reportable vessel.

    A spatial cluster is promoted to a reportable vessel only if it satisfies ALL of:
      1. total_raw_appearances >= MIN_VESSEL_APPEARANCE_FRAMES (default: 5 frames)
      2. peak_height_px >= MIN_VESSEL_PEAK_HEIGHT_PX (or >= MIN_VESSEL_PEAK_HEIGHT_RATIO * orig_h)
      3. peak_area_px >= MIN_VESSEL_PEAK_AREA_PX (default: 300.0 px^2)

    Failing clusters are discarded entirely:
      - Never assigned a vessel_id
      - Not reported in output CSV
      - Not counted in Vessel #N numbering (surviving vessels are cleanly re-indexed 1..M)
      - Do not trigger any outcome (treated as segmentation noise, NOT INSUFFICIENT_DIAGNOSTIC_QUALITY)
      - Logged at debug level with count and peak size for auditability.
    """
    promoted_clusters: Dict[int, List[CandidateDetection]] = {}
    vessel_idx = 1

    for raw_id, dets in clusters.items():
        if not dets:
            continue

        n_appearances = len(dets)
        peak_h = float(max(d.h for d in dets))
        peak_area = float(max(d.area for d in dets))
        peak_contrast = float(max((d.contrast_score_raw for d in dets), default=0.0))

        cond_appearances = bool(n_appearances >= MIN_VESSEL_APPEARANCE_FRAMES)
        # Pass if either absolute height threshold or height ratio is satisfied
        cond_height = bool(peak_h >= MIN_VESSEL_PEAK_HEIGHT_PX or peak_h >= (MIN_VESSEL_PEAK_HEIGHT_RATIO * float(orig_h)))
        cond_area = bool(peak_area >= MIN_VESSEL_PEAK_AREA_PX)
        cond_contrast = bool(peak_contrast >= MIN_VESSEL_PEAK_CONTRAST)

        # Diffuse collateral meshwork / multi-strand porosity filter
        mesh_count = sum(
            1 for d in dets
            if d.mask_region is not None and d.mask_region.shape[0] > 1 and _is_diffuse_collateral_mesh(
                d.mask_region[max(0, d.top):min(orig_h, d.bot), max(0, d.left):min(orig_w, d.right)],
                d.w, d.h, int(d.area), orig_w, orig_h
            )
        )
        is_meshwork = bool(mesh_count / max(1, len(dets)) >= 0.50)
        # Fill ratio against the ROTATED bounding rectangle. The axis-aligned box made any oblique
        # vessel look "hollow" (a diagonal branch fills ~10% of its upright box) and silently discarded
        # genuine oblique branches as collateral mesh. A real tubular vessel fills its rotated
        # rectangle well regardless of angle; a spider-web meshwork still does not.
        med_fill = float(np.median([_rotated_fill_ratio(d, orig_w, orig_h) for d in dets]))
        is_hollow_mesh = bool(med_fill < 0.20 and peak_h < 0.60 * orig_h)
        cond_not_mesh = not (is_meshwork or is_hollow_mesh)

        if cond_appearances and cond_height and cond_area and cond_contrast and cond_not_mesh:
            promoted_clusters[vessel_idx] = dets
            vessel_idx += 1
        else:
            reasons = []
            if not cond_appearances:
                reasons.append(f"appearances {n_appearances} < {MIN_VESSEL_APPEARANCE_FRAMES}")
            if not cond_height:
                reasons.append(f"peak_h {peak_h:.1f}px < {MIN_VESSEL_PEAK_HEIGHT_PX}px")
            if not cond_area:
                reasons.append(f"peak_area {peak_area:.1f}px^2 < {MIN_VESSEL_PEAK_AREA_PX}px^2")
            if not cond_contrast:
                reasons.append(f"peak_contrast {peak_contrast:.1f} < {MIN_VESSEL_PEAK_CONTRAST} (bone cortex / non-vascular)")
            if not cond_not_mesh:
                reasons.append(f"diffuse collateral meshwork (mesh_ratio={mesh_count/max(1, len(dets)):.2f}, med_fill={med_fill:.2f})")
            reason_str = ", ".join(reasons)
            logger.debug(
                f"[VesselSignificanceGate] Discarded noise fragment/bone cluster #{raw_id}: {reason_str} "
                f"(peak_h={peak_h:.1f}, peak_area={peak_area:.1f}, peak_contrast={peak_contrast:.1f}, frames={n_appearances})"
            )

    return promoted_clusters


# ──────────────────────────────────────────────────────────
#  HARD GATE EVALUATION & CANDIDATE SCORING
# ──────────────────────────────────────────────────────────

def evaluate_and_score_candidates(
    cluster_detections: List[CandidateDetection],
    orig_w: int,
    orig_h: int,
    gray_frames: Dict[int, np.ndarray],
    baseline_gray: Optional[np.ndarray] = None,
) -> Tuple[List[CandidateDetection], float, float]:
    """
    Evaluates candidate detections for a spatial vessel cluster:
      1. Determines cine-wide peak height and peak area.
      2. Enforces the Hard Gate:
         - height >= DIAG_MIN_HEIGHT_RATIO * peak_height
         - area >= DIAG_MIN_AREA_RATIO * peak_area
         - stenosis coordinate (sten_x, sten_y) within [0.05, 0.95] FOV boundary
      3. Computes normalized ranking scores across frames passing the hard gate.

    Returns:
      (evaluated_detections, peak_height, peak_area)
    """
    if not cluster_detections:
        return [], 0.0, 0.0

    peak_height = float(max(d.h for d in cluster_detections))
    peak_area = float(max(d.area for d in cluster_detections))

    # Pass 1: QCA analysis and Hard Gate evaluation
    margin_x = FOV_EDGE_MARGIN_RATIO * orig_w
    margin_y = FOV_EDGE_MARGIN_RATIO * orig_h

    for det in cluster_detections:
        # Run orthogonal lumen profile measurement if not already provided (e.g. from raw video segmentation)
        if det.DS_pct is None and det.mask_region is not None and det.mask_region.shape[0] > 1:
            vessel_mask_crop = det.mask_region[det.top:det.bot, det.left:det.right]
            widths, ys = _measure_orthogonal_lumen_profile(vessel_mask_crop, det.top, num_samples=30)

            # Fallback to horizontal slices if orthogonal profiling yields insufficient slices
            if len(widths) < 8:
                widths, ys = [], []
                step = max(1, det.h // 30)
                for y_rel in range(0, det.h, step):
                    w_row = int(np.sum(vessel_mask_crop[y_rel, :] > 0))
                    if w_row > 0:
                        widths.append(float(w_row))
                        ys.append(y_rel + det.top)

            det.lumen_widths = widths
            det.lumen_ys = ys

            if len(widths) >= 5:
                qca = calculate_stenosis_qca(np.array(widths, dtype=float))
                det.qca = qca
                if qca is not None:
                    det.DS_pct = float(qca["DS_pct"])
                    det.MLD_px = float(qca["MLD"])
                    det.RVD_px = float(qca["RVD"])
                    mid_w = np.array(widths)[qca["prox_end"]:qca["dist_start"]] if qca["dist_start"] > qca["prox_end"] else np.array(widths)
                    step_h = max(1, det.h // 30)
                    det.stenosis_length_px = float(int(np.sum(mid_w < qca["RVD"] * 0.70)) * step_h) if len(mid_w) else 0.0

                    # Compute exact (sten_x, sten_y)
                    sten_idx = min(qca["sten_idx"], len(ys) - 1)
                    sten_y = float(ys[sten_idx])
                    y_crop = int(np.clip(sten_y - det.top, 0, det.h - 1))
                    cols = np.where(vessel_mask_crop[y_crop, :] > 0)[0]
                    sten_x = float(np.mean(cols) + det.left) if len(cols) > 0 else float(det.cx)
                    det.sten_x = sten_x
                    det.sten_y = sten_y

        # Ensure contrast score is computed if not already done
        gf = gray_frames.get(det.frame_idx)
        if det.contrast_score_raw == 0.0 and gf is not None and det.mask_region is not None and det.mask_region.shape[0] > 1:
            det.contrast_score_raw = compute_contrast_component(gf, det.mask_region, baseline_gray=baseline_gray)

        # Hard Gate Checks
        h_ok = bool(det.h >= DIAG_MIN_HEIGHT_RATIO * peak_height)
        a_ok = bool(det.area >= DIAG_MIN_AREA_RATIO * peak_area)
        c_ok = bool(det.contrast_score_raw >= CONTRAST_SCORE_MIN_THRESHOLD)

        fov_ok = True
        if det.sten_x is not None and det.sten_y is not None:
            if not (margin_x <= det.sten_x <= orig_w - margin_x and margin_y <= det.sten_y <= orig_h - margin_y):
                fov_ok = False

        if not h_ok:
            det.passed_hard_gate = False
            det.gate_rejection_reason = f"Height ratio {det.h / max(1.0, peak_height):.2f} < {DIAG_MIN_HEIGHT_RATIO}"
        elif not a_ok:
            det.passed_hard_gate = False
            det.gate_rejection_reason = f"Area ratio {det.area / max(1.0, peak_area):.2f} < {DIAG_MIN_AREA_RATIO}"
        elif not c_ok:
            det.passed_hard_gate = False
            det.gate_rejection_reason = f"Contrast score {det.contrast_score_raw:.1f} < {CONTRAST_SCORE_MIN_THRESHOLD} (bone cortex / non-vascular)"
        elif not fov_ok:
            det.passed_hard_gate = False
            det.gate_rejection_reason = f"Stenosis position ({det.sten_x:.1f}, {det.sten_y:.1f}) on FOV edge"
        elif det.DS_pct is None:
            det.passed_hard_gate = False
            det.gate_rejection_reason = "QCA failed to resolve stenosis parameters"
        else:
            det.passed_hard_gate = True
            det.gate_rejection_reason = "PASSED"

    # Pass 2: Ranking Score Computation on Passed Candidates
    gated_passed = [d for d in cluster_detections if d.passed_hard_gate]
    if not gated_passed:
        return cluster_detections, peak_height, peak_area

    # Precompute raw components
    raw_contrast = []
    raw_sharpness = []
    ds_by_frame = {d.frame_idx: d.DS_pct for d in gated_passed if d.DS_pct is not None}

    for det in gated_passed:
        gf = gray_frames.get(det.frame_idx)
        if gf is not None:
            c = compute_contrast_component(gf, det.mask_region, baseline_gray=baseline_gray)
            s = compute_sharpness_component(gf)
        else:
            c = 10.0
            s = 100.0
        det.contrast_score_raw = c
        det.sharpness_score_raw = s
        det.stability_score_raw = compute_neighbor_stability_component(ds_by_frame, det.frame_idx)
        raw_contrast.append(c)
        raw_sharpness.append(s)

    max_c = max(raw_contrast) if raw_contrast else 1.0
    min_c = min(raw_contrast) if raw_contrast else 0.0
    c_denom = max_c - min_c if (max_c - min_c) > 1e-3 else 1.0

    max_s = max(raw_sharpness) if raw_sharpness else 1.0
    min_s = min(raw_sharpness) if raw_sharpness else 0.0
    s_denom = max_s - min_s if (max_s - min_s) > 1e-3 else 1.0

    for det in gated_passed:
        c_norm = float(np.clip((det.contrast_score_raw - min_c) / c_denom, 0.0, 1.0))
        s_norm = float(np.clip((det.sharpness_score_raw - min_s) / s_denom, 0.0, 1.0))
        stab_norm = float(np.clip(det.stability_score_raw, 0.0, 1.0))

        det.total_score = float(W_CONTRAST * c_norm + W_SHARPNESS * s_norm + W_STABILITY * stab_norm)

    return cluster_detections, peak_height, peak_area


# ──────────────────────────────────────────────────────────
#  FRAME SELECTION
# ──────────────────────────────────────────────────────────

def select_best_frames(
    scored_candidates: List[CandidateDetection],
    k: int = K_BEST_FRAMES,
) -> List[CandidateDetection]:
    """
    Selects top-k frames by total_score among candidates that passed the hard gate.
    If fewer than k passed, returns all that passed.
    """
    valid = [d for d in scored_candidates if d.passed_hard_gate]
    if not valid:
        return []
    valid_sorted = sorted(valid, key=lambda d: d.total_score, reverse=True)
    return valid_sorted[:k]


# ──────────────────────────────────────────────────────────
#  CUTOFF / OCCLUSION EVALUATION
# ──────────────────────────────────────────────────────────

def _extent_pieces(dets: List[CandidateDetection]) -> List[CandidateDetection]:
    """
    Pieces used to judge where a vessel starts/ends. Inflow/washout pieces (failed the diagnostic gate)
    are excluded: in video 48 three late-washout pieces (frames 83/88/96, all gate-failed) lying along the
    occluded segment made the reconstituted artery look as if it started near the top, which blocked the
    occlusion call supported by all 59 diagnostic pieces. Falls back to all pieces if too few passed.
    """
    if not ENABLE_CUTOFF_GATED_EXTENT:
        return dets
    gated = [d for d in dets if d.passed_hard_gate]
    return gated if len(gated) >= CUTOFF_MIN_GATED_PIECES else dets


def evaluate_cutoff_occlusion(
    cluster_detections: List[CandidateDetection],
    all_clusters_detections: Dict[int, List[CandidateDetection]],
    cluster_id: int,
    orig_w: int,
    orig_h: int,
    has_confirmed_parent: bool = False,
) -> bool:
    """
    Evaluates whether the vessel exhibits a blunt cutoff stump indicative
    of complete occlusion (antegrade or retrograde cutoff).

    has_confirmed_parent: the vessel is a geometrically confirmed branch. Its inflow is the parent
    trunk it grows out of, so it must never be called a retrograde cutoff for lacking an upstream
    segment that ENDS above it (a branch's feeder runs beside its takeoff, not above it).
    """
    evaluate_cutoff_occlusion.last_detail = {"antegrade": False, "retrograde": False, "skipped": "fewer than 5 detections"}
    if len(cluster_detections) < 5:
        return False

    ext = _extent_pieces(cluster_detections)
    min_top_px = float(min(d.top for d in ext))
    max_bot_px = float(max(d.bot for d in ext))
    max_h_px = float(max(d.h for d in ext))

    rvd_vals = [d.RVD_px for d in cluster_detections if d.RVD_px is not None and d.RVD_px > 0]
    med_rvd = float(np.median(rvd_vals)) if rvd_vals else 0.0
    med_cx = float(np.median([d.cx for d in ext]))

    # Check downstream runoff in other clusters
    other_downstream = False
    other_upstream = False
    align_tol = min(55.0, max(35.0, 1.8 * med_rvd))
    max_gap = 0.25 * orig_h

    for other_id, other_dets in all_clusters_detections.items():
        if other_id == cluster_id or len(other_dets) < 5:
            continue
        o_ext = _extent_pieces(other_dets)
        o_top = float(min(d.top for d in o_ext))
        o_bot = float(max(d.bot for d in o_ext))
        o_cx = float(np.median([d.cx for d in o_ext]))

        # Downstream check
        gap_down = o_top - max_bot_px
        if -40.0 <= gap_down <= max_gap and abs(o_cx - med_cx) <= align_tol:
            other_downstream = True

        # Upstream check
        gap_up = min_top_px - o_bot
        if -40.0 <= gap_up <= max_gap and abs(o_cx - med_cx) <= align_tol:
            other_upstream = True

    # Check natural taper vs blunt stump
    # In blunt stump, distal diameter remains substantial relative to RVD
    dist_ratios = []
    for d in cluster_detections:
        if d.lumen_widths and d.RVD_px and d.RVD_px > 0:
            distal_slice_w = float(np.median(d.lumen_widths[-max(1, len(d.lumen_widths) // 5):]))
            dist_ratios.append(distal_slice_w / d.RVD_px)
    med_dist_ratio = float(np.median(dist_ratios)) if dist_ratios else 1.0
    is_naturally_tapered = bool(med_dist_ratio < 0.45)

    # Antegrade cutoff (e.g. Video 141)
    is_antegrade = bool(
        min_top_px <= CUTOFF_MAX_TOP_RATIO * orig_h
        and max_bot_px < CUTOFF_MAX_BOT_RATIO * orig_h
        and not other_downstream
        and not is_naturally_tapered
        and max_h_px >= CUTOFF_MIN_HEIGHT_RATIO * orig_h
        and med_rvd >= CUTOFF_MIN_RVD_PX
    )

    if has_confirmed_parent:
        other_upstream = True
    # Retrograde cutoff
    is_retrograde = bool(
        min_top_px >= CUTOFF_MAX_TOP_RATIO * orig_h
        and max_bot_px >= CUTOFF_MAX_BOT_RATIO * orig_h
        and not other_upstream
        and not is_naturally_tapered
        and max_h_px >= CUTOFF_MIN_HEIGHT_RATIO * orig_h
        and med_rvd >= CUTOFF_MIN_RVD_PX
    )

    patent_elsewhere = None
    if (is_antegrade or is_retrograde) and ENABLE_PATENCY_ELSEWHERE_CHECK:
        ext_dets = _extent_pieces(cluster_detections)
        scale = 4
        for oid, od in all_clusters_detections.items():
            if oid == cluster_id or not od:
                continue
            good = [o for o in od if o.passed_hard_gate]
            if len({o.frame_idx for o in good}) < PATENCY_MIN_GOOD_FRAMES:
                continue
            fp = cv2.dilate(_cluster_footprint(good, orig_w, orig_h, scale).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            checks = []
            if is_antegrade:     # does the other vessel cover this vessel's lower END and continue below it?
                end = max(ext_dets, key=lambda d: d.bot); y = int(end.bot) - 5
                checks.append((y, int(end.cx), +1))
            if is_retrograde:    # does the other vessel cover this vessel's upper START and continue above it?
                start = min(ext_dets, key=lambda d: d.top); y = int(start.top) + 5
                checks.append((y, int(start.cx), -1))
            for (y, x, direction) in checks:
                ys, xs = y // scale, x // scale
                if not (0 <= ys < fp.shape[0] and 0 <= xs < fp.shape[1]):
                    continue
                if not fp[ys, max(0, xs - 6):xs + 7].any():
                    continue
                run = 0; yy = ys; col = xs
                while 0 <= yy < fp.shape[0]:
                    row = np.where(fp[yy, max(0, col - 8):col + 9])[0]
                    if len(row) == 0:
                        break
                    col = max(0, col - 8) + int(row.mean()); run += 1; yy += direction
                if run * scale >= PATENCY_MIN_CONTINUATION_PX:
                    patent_elsewhere = oid
                    is_antegrade = is_antegrade and direction != +1
                    is_retrograde = is_retrograde and direction != -1

    retro_delay_frames = None
    retro_without_delay = False
    if is_retrograde and ENABLE_DELAYED_FILLING_CHECK:
        own_first = min(d.frame_idx for d in cluster_detections)
        others_first = [min(o.frame_idx for o in od if min(o.bot, max_bot_px) - max(o.top, min_top_px) > 0)
                        for oid, od in all_clusters_detections.items()
                        if oid != cluster_id and od and any(min(o.bot, max_bot_px) - max(o.top, min_top_px) > 0 for o in od)]
        if others_first:
            retro_delay_frames = int(own_first - min(others_first))
            if retro_delay_frames < RETRO_MIN_DELAY_SEC * VIDEO_FPS:
                is_retrograde = False; retro_without_delay = True
    evaluate_cutoff_occlusion.last_detail = {
        "antegrade": is_antegrade, "retrograde": is_retrograde,
        "retro_delay_frames": retro_delay_frames, "retrograde_pattern_without_delayed_filling": retro_without_delay,
        "patent_elsewhere_via_vessel": patent_elsewhere,
        "min_top_ratio": round(min_top_px / orig_h, 3), "max_bot_ratio": round(max_bot_px / orig_h, 3),
        "other_upstream": other_upstream, "other_downstream": other_downstream,
        "naturally_tapered": is_naturally_tapered, "median_rvd_px": round(med_rvd, 1),
        "pieces_used": len(ext), "pieces_total": len(cluster_detections),
    }
    return bool(is_antegrade or is_retrograde)


def check_unresolvable_lumen_across_frames(candidate_frames: List[CandidateDetection]) -> bool:
    """Checks whether the orthogonal lumen profile is unresolvable (<2.0px) across candidates."""
    if not candidate_frames:
        return False
    unresolvable_count = 0
    for det in candidate_frames:
        if det.MLD_px is not None:
            if det.MLD_px < MIN_RESOLVABLE_LUMEN_PX:
                unresolvable_count += 1
        elif det.lumen_widths:
            if min(det.lumen_widths) < MIN_RESOLVABLE_LUMEN_PX:
                unresolvable_count += 1
        else:
            unresolvable_count += 1
    return bool(unresolvable_count == len(candidate_frames) and len(candidate_frames) >= 3)


# ──────────────────────────────────────────────────────────
#  AGGREGATION & STRICT PRIORITY-ORDERED EVALUATION
# ──────────────────────────────────────────────────────────

def classify_non_occlusion_stenosis(ds_pct: float) -> str:
    """
    Classifies diameter stenosis strictly for Outcome 3.
    NEVER produces 'TOTAL OCCLUSION' or any occlusion-implying label.
    """
    if ds_pct >= DS_SEVERE_THRESHOLD:
        return "SEVERE STENOSIS"
    elif ds_pct >= DS_MODERATE_THRESHOLD:
        return "MODERATE STENOSIS"
    elif ds_pct >= 20.0:
        return "MILD STENOSIS"
    else:
        return "NORMAL"


def build_pooled_centerline(
    dets: List[CandidateDetection],
    orig_w: int,
    orig_h: int,
    step: int = 10,
) -> np.ndarray:
    """
    Constructs a pooled 2D centerline (array of shape [N, 2] with columns [x, y])
    along the principal length of a spatial vessel cluster across its available frames.
    """
    if not dets:
        return np.empty((0, 2), dtype=np.float32)
    min_top = min(d.top for d in dets)
    max_bot = max(d.bot for d in dets)

    pts = []
    for y in range(min_top, max_bot + 1, step):
        xs = []
        for d in dets:
            if d.top <= y <= d.bot:
                if d.mask_region is not None and 0 <= y < d.mask_region.shape[0]:
                    row = d.mask_region[y, d.left:d.right]
                    cols = np.where(row > 0)[0]
                    if len(cols) > 0:
                        xs.append(float(np.mean(cols)) + d.left)
                else:
                    xs.append(float(d.cx))
        if xs:
            pts.append([float(np.median(xs)), float(y)])

    if len(pts) < 2:
        pts = [
            [float(np.median([d.cx for d in dets])), float(min_top)],
            [float(np.median([d.cx for d in dets])), float(max_bot)],
        ]

    arr = np.array(pts, dtype=np.float32)
    if len(arr) >= 3:
        kernel = 3
        pad = kernel // 2
        smoothed_x = np.convolve(arr[:, 0], np.ones(kernel) / kernel, mode="same")
        arr[pad:-pad, 0] = smoothed_x[pad:-pad]
    return arr


def detect_geometric_bifurcations(
    spatial_clusters: Dict[int, List[CandidateDetection]],
    orig_w: int,
    orig_h: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Computes parent-child bifurcation relationships between vessel clusters
    using geometric centerline termination and divergence analysis.
    Computed ONCE per video from pooled evidence across all available frames.

    Two clusters are parent-child ONLY if the child's centerline terminates AT
    a point lying on the parent's continuous centerline path (a genuine geometric junction).
    If no genuine junction is found, both vessels remain independent (is_branch=False, parent_id=None).
    """
    bif_results: Dict[int, Dict[str, Any]] = {}
    centerlines: Dict[int, np.ndarray] = {}

    # 1. Build pooled centerline for each cluster
    for vid, dets in spatial_clusters.items():
        centerlines[vid] = build_pooled_centerline(dets, orig_w, orig_h)
        bif_results[vid] = {
            "is_branch": False,
            "parent_id": None,
            "bifurcation_point": None,
            "bifurcation_angle_deg": None,
            "bifurcation_dist_px": None,
        }

    def _median_width(dets: List[CandidateDetection]) -> float:
        ws = [d.RVD_px for d in dets if d.RVD_px is not None and d.RVD_px > 0]
        return float(np.median(ws)) if ws else float(np.median([d.w for d in dets])) if dets else 10.0

    # 2. Check each pair (parent_cand, child_cand)
    for c_id, c_pts in centerlines.items():
        if len(c_pts) < 3:
            continue
        c_dets = spatial_clusters[c_id]
        c_top = min(d.top for d in c_dets)
        c_bot = max(d.bot for d in c_dets)
        c_h = c_bot - c_top

        # Branch proximal takeoff point
        takeoff_pt = c_pts[0]  # (x, y) at the top of the branch
        takeoff_y = takeoff_pt[1]

        best_parent = None
        best_dist = float("inf")
        best_angle = None
        best_pt = None

        for p_id, p_pts in centerlines.items():
            if p_id == c_id or len(p_pts) < 3:
                continue
            p_dets = spatial_clusters[p_id]
            p_top = min(d.top for d in p_dets)
            p_bot = max(d.bot for d in p_dets)
            p_h = p_bot - p_top

            # Condition 1: Parent must pass through the takeoff point (continuous above and below)
            min_margin = min(40.0, 0.10 * p_h)
            if not (p_top + min_margin <= takeoff_y <= p_bot - min_margin):
                continue

            # Condition 2: Geometric proximity at the takeoff level
            near_p_indices = np.where(np.abs(p_pts[:, 1] - takeoff_y) <= 25.0)[0]
            if len(near_p_indices) == 0:
                continue

            near_pts = p_pts[near_p_indices]
            dists = np.linalg.norm(near_pts - takeoff_pt, axis=1)
            min_idx = np.argmin(dists)
            dist_j = float(dists[min_idx])
            parent_junction_pt = near_pts[min_idx]

            # Takeoff must touch the parent (within ~2 parent widths), not merely run beside it
            parent_w = _median_width(p_dets)
            tol_junction = min(BIFURCATION_MAX_JUNCTION_PX, max(15.0, BIFURCATION_JUNCTION_WIDTH_MULT * parent_w))
            if dist_j > tol_junction:
                continue

            # Condition 3: Divergence downstream
            eval_len = min(80.0, 0.5 * c_h)
            y_eval = takeoff_y + eval_len

            c_down_idx = np.argmin(np.abs(c_pts[:, 1] - y_eval))
            p_down_idx = np.argmin(np.abs(p_pts[:, 1] - y_eval))

            v_child = c_pts[c_down_idx] - takeoff_pt
            v_parent = p_pts[p_down_idx] - parent_junction_pt

            norm_c = np.linalg.norm(v_child)
            norm_p = np.linalg.norm(v_parent)

            if norm_c > 1e-3 and norm_p > 1e-3:
                cos_ang = np.dot(v_child, v_parent) / (norm_c * norm_p)
                angle_deg = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
            else:
                angle_deg = 0.0

            dist_downstream = np.linalg.norm(c_pts[c_down_idx] - p_pts[p_down_idx])

            # Require REAL divergence. The previous test (angle >= 5 deg OR separation >= junction
            # distance) was true for any two parallel vessels at constant spacing, labelling a
            # neighbouring parallel artery as a "branch". Separation must now GROW downstream.
            separation_grows = dist_downstream >= max(BIFURCATION_MIN_SEPARATION_GROWTH * dist_j, dist_j + 10.0)
            if angle_deg >= BIFURCATION_MIN_ANGLE_DEG or separation_grows:
                if dist_j < best_dist:
                    best_dist = dist_j
                    best_parent = p_id
                    best_angle = angle_deg
                    best_pt = (float(parent_junction_pt[0]), float(parent_junction_pt[1]))

        if best_parent is not None:
            bif_results[c_id]["is_branch"] = True
            bif_results[c_id]["parent_id"] = best_parent
            bif_results[c_id]["bifurcation_point"] = best_pt
            bif_results[c_id]["bifurcation_angle_deg"] = round(best_angle, 1)
            bif_results[c_id]["bifurcation_dist_px"] = round(best_dist, 1)
            logger.info(
                f"[Bifurcation] Detected true geometric junction: Vessel #{c_id} is a branch of Parent #{best_parent} "
                f"at ({best_pt[0]:.1f}, {best_pt[1]:.1f}) with angle {best_angle:.1f}° (junction distance: {best_dist:.1f}px)"
            )

    return bif_results


def classify_arterial_anatomy(
    vessel_id: int,
    cluster_detections: List[CandidateDetection],
    all_clusters_detections: Dict[int, List[CandidateDetection]],
    orig_w: int,
    orig_h: int,
    is_occlusion: bool = False,
) -> Dict[str, str]:
    """
    Infers peripheral vascular anatomical artery name, Arabic clinical label, and region
    based strictly on craniocaudal vertical height (y_norm, span, and limb sector):
      - CFA (Common Femoral Artery): Proximal inflow segment in the upper sector (y_norm < 0.28)
      - SFA (Superficial Femoral Artery): Primary continuing trunk through thigh (0.20 <= y_norm < 0.65)
      - Popliteal Artery: Knee joint sector (0.60 <= y_norm < 0.85)
      - Tibial / Distal Runoff (ATA/PTA/Peroneal): Distal calf runoff sector (y_norm >= 0.80)

    NOTE: Never decides whether a vessel is a branch or trunk. Branch relationships are
    determined purely by detect_geometric_bifurcations().
    """
    if not cluster_detections:
        return {
            "anatomical_segment": f"Artery #{vessel_id}",
            "anatomical_segment_ar": f"شريان #{vessel_id}",
            "anatomical_region": "Peripheral Vasculature",
            "short_name": f"Artery #{vessel_id}",
        }

    H = float(orig_h) if orig_h > 0 else 1000.0
    W = float(orig_w) if orig_w > 0 else 1000.0

    min_top = float(min(d.top for d in cluster_detections))
    max_bot = float(max(d.bot for d in cluster_detections))
    med_cy = float(np.median([d.cy for d in cluster_detections]))

    y_norm = med_cy / H
    top_norm = min_top / H
    bot_norm = max_bot / H
    height_span = (max_bot - min_top) / H

    if height_span >= 0.55:
        anat_en = "Femoropopliteal Trunk (SFA / Popliteal)"
        anat_ar = "الجذع الفخذي المأبضي (SFA / Popliteal)"
        region = "Thigh & Knee (الفخذ والركبة)"
        short = "Fem-Pop Trunk"
    elif y_norm < 0.28 and bot_norm < 0.45:
        # High proximal inflow
        anat_en = "CFA (Common Femoral Artery)"
        anat_ar = "شريان الفخذ المشترك (CFA)"
        region = "Groin / Inflow (أعلى الفخذ)"
        short = "CFA"
    elif y_norm >= 0.80 or (bot_norm >= 0.90 and top_norm >= 0.65):
        # Distal calf / runoff
        anat_en = "Tibial / Runoff Vessel (ATA / PTA / Peroneal)"
        anat_ar = "الشرايين القصبية الانتهائية (Runoff - الساق)"
        region = "Calf / Below Knee (الساق)"
        short = "Tibial Runoff"
    elif y_norm >= 0.60:
        # Knee sector / Popliteal
        if is_occlusion:
            anat_en = "Popliteal Artery (Distal Cutoff / Occlusion)"
            anat_ar = "الشريان المأبضي - انسداد تام (Popliteal Occlusion)"
        else:
            anat_en = "Popliteal Artery"
            anat_ar = "الشريان المأبضي (Popliteal Artery)"
        region = "Knee / Popliteal Sector (الركبة)"
        short = "Popliteal"
    else:
        # SFA (Superficial Femoral Artery) in thigh
        if top_norm < 0.25:
            anat_en = "SFA (Superficial Femoral Artery — Proximal/Mid)"
            anat_ar = "شريان الفخذ السطحي (SFA - الفخذ)"
        else:
            anat_en = "SFA (Superficial Femoral Artery)"
            anat_ar = "شريان الفخذ السطحي (SFA)"
        region = "Thigh (الفخذ)"
        short = "SFA"

    return {
        "anatomical_segment": anat_en,
        "anatomical_segment_ar": anat_ar,
        "anatomical_region": region,
        "short_name": short,
    }


def _aggregate_vessel_measurement_core(
    vessel_id: int,
    cluster_detections: List[CandidateDetection],
    all_clusters_detections: Dict[int, List[CandidateDetection]],
    orig_w: int,
    orig_h: int,
    mm_per_px: Optional[float] = None,
    is_branch: bool = False,
    parent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Aggregates measurements for one spatial vessel cluster in strict priority order:
      Priority 1: INSUFFICIENT_DIAGNOSTIC_QUALITY (0 frames pass hard gate)
      Priority 2: LIKELY_OCCLUSION_LUMEN_UNRESOLVED (persistent unresolvable lumen, cutoff, or DS >= 100%)
      Priority 3: Standard measurement (NORMAL, MILD, MODERATE, SEVERE STENOSIS)
    """
    passed_candidates = [d for d in cluster_detections if d.passed_hard_gate]
    num_passed = len(passed_candidates)

    is_calibrated = bool(mm_per_px is not None and mm_per_px > 0)
    calibration_status = "catheter_calibrated" if is_calibrated else "uncalibrated_pixel_units_only"

    # ──────────────────────────────────────────────────────
    #  PRIORITY 1: Outcome 1 — Zero Frames Passed Gate
    # ──────────────────────────────────────────────────────
    if num_passed == 0:
        anat = classify_arterial_anatomy(vessel_id, cluster_detections, all_clusters_detections, orig_w, orig_h, is_occlusion=False)
        return {
            "vessel_id": vessel_id,
            "is_branch": is_branch,
            "parent_id": parent_id,
            "anatomical_segment": anat["anatomical_segment"],
            "anatomical_segment_ar": anat["anatomical_segment_ar"],
            "anatomical_region": anat["anatomical_region"],
            "anatomical_short": anat["short_name"],
            "num_candidate_frames_passed_gate": 0,
            "num_frames_used_in_top_k": 0,
            "best_frame_index": None,
            "DS_pct": None,
            "MLD_mm": None,
            "RVD_mm": None,
            "lesion_length_mm": None,
            "angiographic_severity_index": None,
            "interpretation": "INSUFFICIENT_DIAGNOSTIC_QUALITY",
            "occlusion_flag": False,
            "calibration_status": calibration_status,
            "real_std_dev_pct": None,
            "empirical_range_pct": None,
            "selected_ds_values": [],
            "spread_note": None,
        }

    # Frame Selection from passed candidates
    selected_frames = select_best_frames(passed_candidates, k=K_BEST_FRAMES)
    num_used = len(selected_frames)
    candidate_ds_values = [float(f.DS_pct) for f in selected_frames if f.DS_pct is not None]

    # ──────────────────────────────────────────────────────
    #  PRIORITY 2: Outcome 2 — Likely Occlusion / Unresolvable Lumen / Cutoff
    # ──────────────────────────────────────────────────────
    is_cutoff = evaluate_cutoff_occlusion(
        cluster_detections, all_clusters_detections, vessel_id, orig_w, orig_h,
        has_confirmed_parent=bool(is_branch and parent_id is not None),
    )
    is_unresolvable = check_unresolvable_lumen_across_frames(passed_candidates)

    # Temporary evaluation of worst stenosis from selected clean frames
    candidate_max_ds = max(candidate_ds_values) if candidate_ds_values else 0.0

    # If DS reached total occlusion threshold (>=100%), re-route entirely to Outcome 2
    is_100pct_reroute = bool(candidate_max_ds >= DS_TOTAL_OCCLUSION_THRESHOLD)

    occlusion_rule = [name for name, hit in (("cutoff_antegrade", evaluate_cutoff_occlusion.last_detail.get("antegrade")),
                                             ("cutoff_retrograde", evaluate_cutoff_occlusion.last_detail.get("retrograde")),
                                             ("unresolvable_lumen", is_unresolvable),
                                             ("ds_ge_100", is_100pct_reroute)) if hit]
    cutoff_detail = dict(evaluate_cutoff_occlusion.last_detail)

    if is_cutoff or is_unresolvable or is_100pct_reroute:
        anat = classify_arterial_anatomy(vessel_id, cluster_detections, all_clusters_detections, orig_w, orig_h, is_occlusion=True)
        return {
            "occlusion_rule": occlusion_rule,
            "cutoff_detail": cutoff_detail,
            "vessel_id": vessel_id,
            "is_branch": is_branch,
            "parent_id": parent_id,
            "anatomical_segment": anat["anatomical_segment"],
            "anatomical_segment_ar": anat["anatomical_segment_ar"],
            "anatomical_region": anat["anatomical_region"],
            "anatomical_short": anat["short_name"],
            "num_candidate_frames_passed_gate": num_passed,
            "num_frames_used_in_top_k": num_used,
            "best_frame_index": selected_frames[0].frame_idx if selected_frames else None,
            "DS_pct": None,
            "MLD_mm": None,
            "RVD_mm": None,
            "lesion_length_mm": None,
            "angiographic_severity_index": None,
            "interpretation": "LIKELY_OCCLUSION_LUMEN_UNRESOLVED",
            "occlusion_flag": True,
            "calibration_status": calibration_status,
            "real_std_dev_pct": None,
            "empirical_range_pct": None,
            "selected_ds_values": candidate_ds_values,
            "spread_note": None,
        }

    # ──────────────────────────────────────────────────────
    #  PRIORITY 3: Outcome 3 — Standard Patent / Stenotic Measurement
    # ──────────────────────────────────────────────────────
    # Take the MAX among the k clean candidate frames (not median)
    best_frame = max(selected_frames, key=lambda f: f.DS_pct if f.DS_pct is not None else -1.0)
    DS_pct_final = float(best_frame.DS_pct) if best_frame.DS_pct is not None else 0.0

    # Ensure DS_pct_final < 100.0% in Outcome 3
    DS_pct_final = min(DS_pct_final, 99.9)

    # Real empirical spread computation from the actual DS_pct values of the selected frames
    if len(candidate_ds_values) >= 2:
        real_std_dev = round(float(np.std(candidate_ds_values, ddof=1)), 2)
        emp_range = [round(float(min(candidate_ds_values)), 2), round(float(max(candidate_ds_values)), 2)]
        spread_comment = f"Empirical spread across the {len(candidate_ds_values)} selected diagnostic frames"
    else:
        real_std_dev = None
        emp_range = None
        spread_comment = None

    MLD_px = best_frame.MLD_px
    RVD_px = best_frame.RVD_px
    lesion_length_px = best_frame.stenosis_length_px

    if is_calibrated and mm_per_px is not None:
        MLD_mm = round(MLD_px * mm_per_px, 2) if MLD_px is not None else None
        RVD_mm = round(RVD_px * mm_per_px, 2) if RVD_px is not None else None
        lesion_length_mm = round(lesion_length_px * mm_per_px, 2) if lesion_length_px is not None else None
        length_excess = max(0.0, (lesion_length_mm - NOMINAL_REFERENCE_LENGTH_MM) / NOMINAL_REFERENCE_LENGTH_MM) if lesion_length_mm is not None else 0.0
        length_factor = 1.0 + LENGTH_FACTOR_COEFF * length_excess
    else:
        MLD_mm = None
        RVD_mm = None
        lesion_length_mm = None
        length_factor = 1.0

    DS_fraction = DS_pct_final / 100.0
    ASI = float(np.clip(1.0 - WIBMER_ALPHA * (DS_fraction ** WIBMER_BETA) * length_factor, 0.10, 1.0))

    interpretation = classify_non_occlusion_stenosis(DS_pct_final)
    anat = classify_arterial_anatomy(vessel_id, cluster_detections, all_clusters_detections, orig_w, orig_h, is_occlusion=False)

    return {
        "vessel_id": vessel_id,
        "is_branch": is_branch,
        "parent_id": parent_id,
        "anatomical_segment": anat["anatomical_segment"],
        "anatomical_segment_ar": anat["anatomical_segment_ar"],
        "anatomical_region": anat["anatomical_region"],
        "anatomical_short": anat["short_name"],
        "num_candidate_frames_passed_gate": num_passed,
        "num_frames_used_in_top_k": num_used,
        "best_frame_index": int(best_frame.frame_idx),
        "DS_pct": round(DS_pct_final, 2),
        "MLD_mm": MLD_mm,
        "RVD_mm": RVD_mm,
        "lesion_length_mm": lesion_length_mm,
        "angiographic_severity_index": round(ASI, 4),
        "interpretation": interpretation,
        "occlusion_flag": False,
        "calibration_status": calibration_status,
        "real_std_dev_pct": real_std_dev,
        "empirical_range_pct": emp_range,
        "selected_ds_values": candidate_ds_values,
        "spread_note": spread_comment,
    }


_GAP_CLASS_SEVERITY = {"NO_FLOW": 3, "FAINT_FLOW": 2, "SEGMENTATION_GAP": 1}


def assess_segment_interruption(
    cluster_detections: List[CandidateDetection],
    orig_h: int,
) -> Dict[str, Any]:
    """
    Decides ONCE per vessel, from pooled per-frame gap evidence, whether the vessel is
    interrupted in the middle (proximal stump + distal stump with a gap between them).

    Persistence is required: the gap must appear in >= INTERRUPTION_MIN_FRAMES frames AND in
    >= INTERRUPTION_MIN_PERSISTENCE of the frames where the vessel is visible on BOTH sides of the
    gap. A gap that closes in many such frames is bolus timing, not an occlusion.

    status:
      "NONE"              no usable gap evidence
      "OCCLUSION"         persistent gap with little/no dye inside -> Outcome 2
      "SEGMENTATION_GAP"  persistent gap WITH dye inside -> lumen patent but NOT measured inside the gap
      "TRANSIENT_GAP"     gap seen, but not persistent enough to call either way
    """
    result: Dict[str, Any] = {
        "interruption_status": "NONE",
        "occlusion_subtype": None,
        "interruption_gap_px": None,
        "interruption_frames": 0,
        "interruption_persistence": None,
        "interruption_votes": {},
    }

    # Worst usable gap per frame
    per_frame: Dict[int, Dict[str, Any]] = {}
    for d in cluster_detections:
        for rec in d.gap_records:
            cls = rec.get("classification")
            if cls not in _GAP_CLASS_SEVERITY:
                continue  # STUMPS_NOT_OPACIFIED / UNKNOWN_NO_IMAGE are not evidence
            prev = per_frame.get(rec["frame_idx"])
            if prev is None or _GAP_CLASS_SEVERITY[cls] > _GAP_CLASS_SEVERITY[prev["classification"]]:
                per_frame[rec["frame_idx"]] = rec
    if not per_frame:
        return result

    gap_top = float(np.median([r["gap_top_y"] for r in per_frame.values()]))
    gap_bot = float(np.median([r["gap_bottom_y"] for r in per_frame.values()]))

    # Frames where the vessel is visible on both sides of the gap region
    frames_dets: Dict[int, List[CandidateDetection]] = {}
    for d in cluster_detections:
        frames_dets.setdefault(d.frame_idx, []).append(d)
    both_sides = 0
    for f_idx, dets in frames_dets.items():
        above = any(d.top < gap_top for d in dets)
        below = any(d.bot > gap_bot for d in dets)
        if above and below:
            both_sides += 1
    both_sides = max(both_sides, len(per_frame))
    persistence = len(per_frame) / float(both_sides)

    votes = {"NO_FLOW": 0, "FAINT_FLOW": 0, "SEGMENTATION_GAP": 0}
    for r in per_frame.values():
        votes[r["classification"]] += 1

    result.update({
        "interruption_gap_px": round(float(np.median([r["gap_px"] for r in per_frame.values()])), 1),
        "interruption_frames": len(per_frame),
        "interruption_persistence": round(persistence, 2),
        "interruption_votes": votes,
    })

    if len(per_frame) < INTERRUPTION_MIN_FRAMES or persistence < INTERRUPTION_MIN_PERSISTENCE:
        result["interruption_status"] = "TRANSIENT_GAP"
        return result

    occlusion_votes = votes["NO_FLOW"] + votes["FAINT_FLOW"]
    if occlusion_votes > votes["SEGMENTATION_GAP"]:
        result["interruption_status"] = "OCCLUSION"
        result["occlusion_subtype"] = "TOTAL_NO_FLOW" if votes["NO_FLOW"] >= votes["FAINT_FLOW"] else "SUBTOTAL_FAINT_FLOW"
    else:
        result["interruption_status"] = "SEGMENTATION_GAP"
    return result


def aggregate_vessel_measurement(
    vessel_id: int,
    cluster_detections: List[CandidateDetection],
    all_clusters_detections: Dict[int, List[CandidateDetection]],
    orig_w: int,
    orig_h: int,
    mm_per_px: Optional[float] = None,
    is_branch: bool = False,
    parent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Priority-ordered outcome evaluation (see _aggregate_vessel_measurement_core), plus the
    mid-vessel interruption check:

      - A persistent interruption with little/no dye in the gap is an OCCLUSION finding and
        overrides Outcomes 1 and 3. It is evidence from two well-opacified stumps, so it does not
        depend on a single segment passing the height/area hard gate (stumps are shorter by nature).
      - A persistent gap WITH dye inside is NOT called an occlusion, but the lumen inside the gap
        was never measured, so the vessel is flagged requires_visual_review=True instead of being
        silently reported as normal.
    """
    evaluate_cutoff_occlusion.last_detail = None
    res = _aggregate_vessel_measurement_core(
        vessel_id, cluster_detections, all_clusters_detections,
        orig_w, orig_h, mm_per_px=mm_per_px, is_branch=is_branch, parent_id=parent_id,
    )
    if "cutoff_detail" not in res:
        res["cutoff_detail"] = evaluate_cutoff_occlusion.last_detail
    cd = res.get("cutoff_detail") or {}
    if cd.get("patent_elsewhere_via_vessel") is not None and not res.get("occlusion_flag"):
        note = ("Ends abruptly, but another vessel is seen passing through the same point and continuing beyond it "
                "at another time in the run, so it is NOT reported as an occlusion. Visual review required.")
        res["requires_visual_review"] = True
        res["review_reason"] = note if not res.get("review_reason") else res["review_reason"] + " | " + note
    if cd.get("retrograde_pattern_without_delayed_filling") and not res.get("occlusion_flag"):
        note = (f"Vessel starts abruptly mid-image but filled only {cd.get('retro_delay_frames')} frame(s) after the "
                f"other vessels at its level (no delayed filling), so it is NOT reported as an occlusion: its connection "
                f"to a parent artery was probably not segmented. Visual review required.")
        res["requires_visual_review"] = True
        res["review_reason"] = note if not res.get("review_reason") else res["review_reason"] + " | " + note
    intr = assess_segment_interruption(cluster_detections, orig_h)
    res.update(intr)
    res["requires_visual_review"] = False
    res["review_reason"] = None

    if intr["interruption_status"] == "OCCLUSION":
        anat = classify_arterial_anatomy(vessel_id, cluster_detections, all_clusters_detections, orig_w, orig_h, is_occlusion=True)
        res.update({
            "anatomical_segment": anat["anatomical_segment"],
            "anatomical_segment_ar": anat["anatomical_segment_ar"],
            "anatomical_region": anat["anatomical_region"],
            "anatomical_short": anat["short_name"],
            "DS_pct": None, "MLD_mm": None, "RVD_mm": None, "lesion_length_mm": None,
            "angiographic_severity_index": None,
            "interpretation": "LIKELY_OCCLUSION_LUMEN_UNRESOLVED",
            "occlusion_flag": True,
            "real_std_dev_pct": None, "empirical_range_pct": None,
            "selected_ds_values": [], "spread_note": None,
            "occlusion_rule": ["mid_vessel_interruption"],
        })
        logger.info(
            f"[Interruption] Vessel #{vessel_id}: mid-vessel gap ~{intr['interruption_gap_px']}px in "
            f"{intr['interruption_frames']} frames (persistence {intr['interruption_persistence']}), "
            f"votes={intr['interruption_votes']} -> {intr['occlusion_subtype']}"
        )
    elif res.get("occlusion_flag"):
        res["occlusion_subtype"] = "CUTOFF_OR_UNRESOLVED_LUMEN"
    elif intr["interruption_status"] == "TRANSIENT_GAP" and not res.get("occlusion_flag"):
        good_gap_frames = sorted({d.frame_idx for d in cluster_detections if d.passed_hard_gate and d.gap_records})
        if len(good_gap_frames) >= GAP_REVIEW_MIN_GOOD_FRAMES:
            res["requires_visual_review"] = True
            res["review_reason"] = (
                f"The segmentation broke along this vessel (gap ~{intr['interruption_gap_px']}px) in "
                f"{len(good_gap_frames)} good-quality frame(s). A tight lesion can make the model lose the lumen "
                f"exactly there, and the lumen inside the break is NOT measured: the reported DS% may miss it. "
                f"Visual review of the break is required."
            )
    elif intr["interruption_status"] == "SEGMENTATION_GAP":
        res["requires_visual_review"] = True
        res["review_reason"] = (
            f"Segmentation gap of ~{intr['interruption_gap_px']}px with contrast present inside it in "
            f"{intr['interruption_frames']} frames: lumen inside the gap was NOT measured, so the reported "
            f"DS% covers only the segmented parts. Visual review of the gap is required."
        )
    return res


def orig_w_hint(clusters: Dict[int, List[CandidateDetection]]) -> int:
    for dets in clusters.values():
        for d in dets:
            if d.mask_region is not None:
                return int(d.mask_region.shape[1])
    return 1280


def select_main_arteries(
    results: List[Dict[str, Any]],
    clusters: Dict[int, List[CandidateDetection]],
    evidence_info: Dict[int, Dict[str, Any]],
    orig_h: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[CandidateDetection]], Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Keeps only the main arteries (at most MAX_MAIN_ARTERIES), decided AFTER the occlusion decision:
      - a vessel reported as an occlusion is ALWAYS kept (an occluded artery is often short, and hiding it is the
        worst possible error); this does not detect occlusions, it only stops this filter from hiding one;
      - other vessels must be reliable (not low-evidence), have a caliber >= MAIN_MIN_REL_CALIBER of the thickest
        vessel in the same video, and span >= MAIN_MIN_SPAN of the image height; the largest (caliber x span) win.
    Kept vessels are renumbered 1..n (original order); a branch whose parent was dropped loses its parent link.
    Dropped vessels are returned as suppressed records (reason SUPPRESSED_MINOR_VESSEL) so nothing vanishes silently.
    """
    if not ENABLE_MAIN_ARTERY_FILTER or not results:
        return results, clusters, evidence_info, []
    info = {}
    for r in results:
        vid = r["vessel_id"]; dets = _extent_pieces(clusters[vid])
        rvd = [d.RVD_px for d in dets if d.RVD_px]
        caliber = float(np.median(rvd)) if rvd else float(np.median([d.w for d in dets]))
        span = (max(d.bot for d in dets) - min(d.top for d in dets)) / float(orig_h)
        info[vid] = {"caliber": caliber, "span": span, "occl": bool(r.get("occlusion_flag")), "low": bool(r.get("low_evidence"))}
    reliable_cal = [v["caliber"] for v in info.values() if not v["low"]] or [v["caliber"] for v in info.values()]
    cmax = max(reliable_cal) if reliable_cal else 1.0
    by_id_all = {r["vessel_id"]: r for r in results}
    for vid, v in info.items():
        rim = (evidence_info.get(vid) or {}).get("edge_rim_ratio")
        v["bone_edge"] = rim is not None and rim >= BONE_EDGE_HIDE_RIM
        # Same artery seen again: its cutoff was withdrawn because another vessel passes through the same point, and
        # its whole path lies inside that vessel (123: the upper half of the main artery after patient motion).
        v["same_as"] = None
        other = ((by_id_all[vid].get("cutoff_detail") or {}).get("patent_elsewhere_via_vessel"))
        if other is not None and other in clusters and other != vid:
            share = _centerline_share(_cluster_footprint(_extent_pieces(clusters[vid]), orig_w_hint(clusters), orig_h),
                                      _cluster_footprint(_extent_pieces(clusters[other]), orig_w_hint(clusters), orig_h))
            if share >= TRACK_MIN_PATH_SHARE:
                v["same_as"] = other
        v["rel"] = v["caliber"] / cmax if cmax > 0 else 0.0
        v["qualifies"] = ((not v["low"]) and (not v["bone_edge"]) and v["same_as"] is None
                          and v["rel"] >= MAIN_MIN_REL_CALIBER and v["span"] >= MAIN_MIN_SPAN)
        v["via_branch"] = False
    # A reliable branch geometrically linked to a qualifying main artery counts as main even if it is shorter or thinner
    # (the main arteries of the thigh and leg branch off each other). Low-evidence pieces never qualify this way.
    if MAIN_KEEP_LINKED_BRANCHES:
        by_id = {r["vessel_id"]: r for r in results}
        for vid, v in info.items():
            r = by_id[vid]; parent = r.get("parent_id")
            if (not v["qualifies"] and not v["low"] and not v["bone_edge"] and v["same_as"] is None and r.get("is_branch") and parent in info
                    and (info[parent]["qualifies"] or info[parent]["occl"])):
                v["qualifies"] = True; v["via_branch"] = True
    occluded = [vid for vid, v in info.items() if v["occl"] and not v["bone_edge"]]   # a bone edge is not an artery
    others = sorted([vid for vid, v in info.items() if v["qualifies"] and not v["occl"]],
                    key=lambda vid: -info[vid]["caliber"] * info[vid]["span"])
    keep = set(occluded) | set(others[:max(0, MAX_MAIN_ARTERIES - len(occluded))])
    if not keep:   # never return an empty report: keep the single strongest candidate
        keep = {max(info, key=lambda vid: (not info[vid]["low"], info[vid]["caliber"] * info[vid]["span"]))}
    dropped = []
    for r in results:
        vid = r["vessel_id"]
        if vid in keep:
            continue
        v = info[vid]
        why = ("not an artery: dark line with a stronger bright rim (subtraction edge of bone/tissue, e.g. patient motion)"
               if v["bone_edge"] else f"same artery as vessel {v['same_as']}, seen again in another phase (e.g. after patient motion)"
               if v["same_as"] is not None else "low evidence" if v["low"] else "more than 3 main-artery candidates" if v["qualifies"] else f"caliber {v['rel']:.2f} of the main artery" if v["rel"] < MAIN_MIN_REL_CALIBER
               else f"visible over only {v['span']:.2f} of the image" if v["span"] < MAIN_MIN_SPAN else "more than 3 main-artery candidates")
        dropped.append({"raw_cluster_id": vid, "evidence_status": "SUPPRESSED_MINOR_VESSEL", "reason": why,
                        "frames_passed_gate": evidence_info.get(vid, {}).get("frames_passed_gate"),
                        "caliber_px": round(v["caliber"], 1), "rel_caliber": round(v["rel"], 2), "span": round(v["span"], 2),
                        "detections": _describe_detections(clusters[vid])})
        logger.info(f"[MainArtery] vessel {vid} not reported: {why}")
    for r in results:
        vid = r["vessel_id"]
        if vid not in keep:
            continue
        rim = (evidence_info.get(vid) or {}).get("edge_rim_ratio")
        note = None
        if info[vid]["bone_edge"]:          # only possible for an occlusion (never hidden): say so loudly
            note = "Shows a subtraction-edge signature (bright rim stronger than the dark line): may be a bone edge, not an artery."
        if note:
            r["possible_edge_artifact"] = True; r["requires_visual_review"] = True
            r["review_reason"] = note if not r.get("review_reason") else r["review_reason"] + " | " + note
    kept_ids = [r["vessel_id"] for r in results if r["vessel_id"] in keep]
    remap = {old: new for new, old in enumerate(kept_ids, start=1)}
    new_results, new_clusters, new_info = [], {}, {}
    for r in results:
        old = r["vessel_id"]
        if old not in keep:
            continue
        r = dict(r); r["vessel_id"] = remap[old]
        if r.get("parent_id") is not None:
            r["parent_id"] = remap.get(r["parent_id"])
            if r["parent_id"] is None:
                r["is_branch"] = False
        new_results.append(r); new_clusters[remap[old]] = clusters[old]
        if old in evidence_info:
            new_info[remap[old]] = evidence_info[old]
    return new_results, new_clusters, new_info, dropped


def evidence_threshold(total_frames: int) -> int:
    """Minimum number of gate-passing frames for a vessel to count as reliably detected."""
    frac = int(np.ceil(MIN_EVIDENCE_FRACTION * max(1, total_frames)))
    return int(max(MIN_EVIDENCE_FRAMES_FLOOR, min(MIN_EVIDENCE_FRAMES_MAX, frac)))


def _frames_passed(dets: List[CandidateDetection]) -> int:
    return len({d.frame_idx for d in dets if d.passed_hard_gate})


def _edge_artifact_score(dets: List[CandidateDetection], gray_frames: Optional[Dict[int, np.ndarray]],
                         orig_w: int, orig_h: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Subtraction-edge signature. A contrast-filled vessel is a DARK column only. A bone/soft-tissue edge
    revealed by patient motion in a subtracted image appears as a dark line with a BRIGHT line next to it.
    Returns (rim_ratio = bright rim / dark core, rim_vs_noise = bright rim / far-background noise).
    """
    if not gray_frames:
        return None, None
    ratios, rel = [], []
    for d in sorted(dets, key=lambda x: x.area, reverse=True)[:5]:
        g = gray_frames.get(d.frame_idx)
        if g is None:
            continue
        m = _ensure_full_mask(d, orig_w, orig_h).astype(np.uint8)
        y0, y1 = max(0, d.top), min(orig_h, d.bot)
        x0, x1 = max(0, d.left - 60), min(orig_w, d.right + 60)
        gg = g[y0:y1, x0:x1].astype(np.float32)
        dark = cv2.GaussianBlur(gg, (0, 0), 25) - gg                      # + = darker than surroundings
        mm = m[y0:y1, x0:x1]
        core = mm > 0
        rim = (cv2.dilate(mm, np.ones((1, 25), np.uint8)) > 0) & ~(cv2.dilate(mm, np.ones((1, 5), np.uint8)) > 0)
        far = (cv2.dilate(mm, np.ones((1, 101), np.uint8)) > 0) & ~(cv2.dilate(mm, np.ones((1, 61), np.uint8)) > 0)
        rows = [r for r in range(core.shape[0]) if core[r].any() and rim[r].any() and far[r].any()]
        if len(rows) < 10:
            continue
        dk = np.median([dark[r][core[r]].max() for r in rows])
        br = np.median([(-dark[r][rim[r]]).max() for r in rows])
        nz = np.median([np.abs(dark[r][far[r]]).max() for r in rows])
        if dk > 0:
            ratios.append(br / dk)
            rel.append(br / max(nz, 1e-3))
    if not ratios:
        return None, None
    return float(np.median(ratios)), float(np.median(rel))


def _diverges_from(weak_mask: np.ndarray, trunk_mask: np.ndarray) -> bool:
    """
    True if the weak piece moves away from the trunk along its length (a branch), False if it stays at a
    roughly constant distance (a parallel copy / double contour of the same artery).
    Distance is measured row by row from the weak piece's centre to the nearest trunk pixel.
    """
    rows = np.where(weak_mask.any(axis=1) & trunk_mask.any(axis=1))[0]
    if len(rows) < 10:
        return False
    dist, widths = [], []
    for r in rows:
        xw = np.where(weak_mask[r])[0]; xr = np.where(trunk_mask[r])[0]
        c = xw.mean(); widths.append(len(xw))
        dist.append(float(np.min(np.abs(xr - c))))
    spread = float(np.percentile(dist, 90) - np.percentile(dist, 10))
    return spread >= max(BRANCH_MIN_DIVERGENCE_PX, BRANCH_MIN_DIVERGENCE_WIDTH_MULT * float(np.median(widths)))


def _side_by_side_fraction(
    weak_dets: List[CandidateDetection],
    reliable_by_frame: Dict[int, List[CandidateDetection]],
    orig_w: int, orig_h: int,
) -> Tuple[Optional[float], int]:
    """
    How often a weak piece runs BESIDE a reliable vessel in the same (or adjacent) frame.
    A real branch has its own path next to the trunk in the same frame (high vertical sharing, little
    pixel overlap). A duplicate lies ON the trunk, or continues it end-to-end (little vertical sharing).
    Returns (fraction of comparable frames that look side-by-side, number of comparable frames);
    fraction is None when no reliable vessel exists near any of the weak piece's frames.
    """
    comparable = side = 0
    for d in weak_dets:
        near = [r for f in range(d.frame_idx - SIDE_BY_SIDE_FRAME_WINDOW, d.frame_idx + SIDE_BY_SIDE_FRAME_WINDOW + 1)
                for r in reliable_by_frame.get(f, [])]
        near = [r for r in near if min(d.bot, r.bot) - max(d.top, r.top) > 0]      # shares some height
        md = _ensure_full_mask(d, orig_w, orig_h) > 0
        k = 2 * SIDE_BY_SIDE_MAX_GAP_PX + 1
        near = [r for r in near                                                     # and is actually close to it
                if (md & (cv2.dilate((_ensure_full_mask(r, orig_w, orig_h) > 0).astype(np.uint8), np.ones((k, k), np.uint8)) > 0)).any()]
        if not near:
            continue
        comparable += 1
        n_d = max(1, int(md.sum()))
        beside = False
        for r in near:
            v_share = (min(d.bot, r.bot) - max(d.top, r.top)) / max(1.0, float(d.bot - d.top))
            mr = cv2.dilate((_ensure_full_mask(r, orig_w, orig_h) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            on_top = float((md & mr).sum()) / n_d
            if on_top > SIDE_BY_SIDE_MAX_OVERLAP:
                beside = False
                break                                   # lies on a reliable vessel in this frame -> not a branch here
            if v_share >= SIDE_BY_SIDE_MIN_VERTICAL and (not ENABLE_BRANCH_DIVERGENCE
                                                         or _diverges_from(md, _ensure_full_mask(r, orig_w, orig_h) > 0)):
                beside = True
        side += int(beside)
    return (side / comparable if comparable else None), comparable


def _describe_detections(dets: List[CandidateDetection]) -> List[Dict[str, Any]]:
    """Compact per-frame description of a cluster's pieces (for debug_clusters.json)."""
    out = []
    for d in sorted(dets, key=lambda x: (x.frame_idx, x.top)):
        out.append({
            "frame": int(d.frame_idx), "box": [int(d.left), int(d.top), int(d.right), int(d.bot)],
            "area": int(d.area), "passed_gate": bool(d.passed_hard_gate),
            "gate_rejection": getattr(d, "gate_rejection_reason", None),
            "DS_pct": None if d.DS_pct is None else round(float(d.DS_pct), 1),
            "MLD_px": None if d.MLD_px is None else round(float(d.MLD_px), 1),
            "RVD_px": None if d.RVD_px is None else round(float(d.RVD_px), 1),
            "stenosis_xy": None if d.sten_x is None else [round(float(d.sten_x)), round(float(d.sten_y))],
            "contrast": round(float(d.contrast_score_raw), 1),
            "gaps": [{"gap_px": g.get("gap_px"), "class": g.get("classification"), "ratio": g.get("contrast_ratio")}
                     for g in d.gap_records],
        })
    return out


def triage_vessel_evidence(
    clusters: Dict[int, List[CandidateDetection]],
    gray_frames: Optional[Dict[int, np.ndarray]],
    total_frames: int,
    orig_w: int,
    orig_h: int,
) -> Tuple[Dict[int, List[CandidateDetection]], Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Sorts evaluated clusters by how well they are supported (run AFTER evaluate_and_score_candidates).

      RELIABLE      : >= evidence_threshold() gate-passing frames -> reported normally.
      weak clusters (fewer frames) are then checked:
        SUPPRESSED_DUPLICATE     : lies mostly inside a reliable vessel's footprint (same artery counted twice,
                                   or late-washout pieces of the same trunk) -> removed from the report.
        LOW_EVIDENCE             : anything else (e.g. a genuine small branch) -> kept, but it may never be
                                   called an occlusion and is flagged for visual review. If it also shows the
                                   subtraction-edge signature (dark line with a bright rim, typical of a bone edge
                                   revealed by patient motion) it is flagged possible_edge_artifact — flagged, not
                                   removed, because that signal was not separable enough on real data to delete on.

    Returns (kept clusters RENUMBERED 1..n in original order, per-kept-cluster evidence info, suppressed list).
    """
    thr = evidence_threshold(total_frames)
    passed = {cid: _frames_passed(d) for cid, d in clusters.items()}
    reliable = [cid for cid in clusters if passed[cid] >= thr]

    reliable_by_frame: Dict[int, List[CandidateDetection]] = {}
    for cid in reliable:
        for d in clusters[cid]:
            reliable_by_frame.setdefault(d.frame_idx, []).append(d)

    footprint = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for cid in reliable:
        for d in clusters[cid]:
            footprint |= (_ensure_full_mask(d, orig_w, orig_h) > 0).astype(np.uint8)
    if footprint.any():
        k = 2 * DUPLICATE_FOOTPRINT_DILATE_PX + 1
        footprint = cv2.dilate(footprint, np.ones((k, k), np.uint8))

    kept_order: List[int] = []
    info: Dict[int, Dict[str, Any]] = {}
    suppressed: List[Dict[str, Any]] = []
    for cid, dets in clusters.items():
        rec: Dict[str, Any] = {"raw_cluster_id": cid, "frames_passed_gate": passed[cid], "evidence_threshold": thr,
                               "footprint_overlap": None, "edge_rim_ratio": None, "edge_rim_vs_noise": None}
        if passed[cid] >= thr:
            rec["evidence_status"] = "RELIABLE"
            rim, rim_rel = _edge_artifact_score(dets, gray_frames, orig_w, orig_h)
            rec["edge_rim_ratio"] = None if rim is None else round(rim, 2)
            rec["edge_rim_vs_noise"] = None if rim_rel is None else round(rim_rel, 2)
        else:
            tot = inside = 0
            for d in dets:
                m = _ensure_full_mask(d, orig_w, orig_h) > 0
                tot += int(m.sum()); inside += int((m & (footprint > 0)).sum())
            overlap = inside / tot if tot else 0.0
            rim, rim_rel = _edge_artifact_score(dets, gray_frames, orig_w, orig_h)
            sbs, n_cmp = _side_by_side_fraction(dets, reliable_by_frame, orig_w, orig_h)
            rec["side_by_side_fraction"] = None if sbs is None else round(sbs, 2)
            rec["side_by_side_frames"] = n_cmp
            rec.update({"footprint_overlap": round(overlap, 2),
                        "edge_rim_ratio": None if rim is None else round(rim, 2),
                        "edge_rim_vs_noise": None if rim_rel is None else round(rim_rel, 2)})
            branch_like = sbs is not None and sbs >= SIDE_BY_SIDE_MIN_FRACTION
            if reliable and overlap >= DUPLICATE_FOOTPRINT_OVERLAP and not branch_like:
                rec["evidence_status"] = "SUPPRESSED_DUPLICATE"
            else:
                rec["evidence_status"] = "LOW_EVIDENCE"
                # A piece running right beside a dark trunk shows a fake "bright rim" (the trunk darkens the local
                # background), so the edge signature is not trusted for branch-like pieces.
                rec["possible_edge_artifact"] = bool(rim is not None and rim >= EDGE_ARTIFACT_RIM_RATIO and not branch_like)
                rec["branch_like"] = bool(branch_like)
        if rec["evidence_status"].startswith("SUPPRESSED"):
            rec["detections"] = _describe_detections(dets)
            suppressed.append(rec)
            logger.info(f"[Evidence] raw cluster {cid} removed: {rec['evidence_status']} "
                        f"(frames={passed[cid]}<{thr}, overlap={rec['footprint_overlap']}, rim={rec['edge_rim_ratio']})")
        else:
            kept_order.append(cid)
            info[cid] = rec

    renumbered: Dict[int, List[CandidateDetection]] = {}
    new_info: Dict[int, Dict[str, Any]] = {}
    for new_id, cid in enumerate(kept_order, start=1):
        renumbered[new_id] = clusters[cid]
        new_info[new_id] = info[cid]
    return renumbered, new_info, suppressed


def apply_evidence_status(res: Dict[str, Any], ev: Dict[str, Any], anatomy_fn=None) -> Dict[str, Any]:
    """Adds evidence fields to an aggregated vessel result; a LOW_EVIDENCE vessel can never be an occlusion.
    anatomy_fn: optional callable returning classify_arterial_anatomy(..., is_occlusion=False); used to
    replace an anatomy name that carries occlusion wording (e.g. "Popliteal Artery (Distal Cutoff / Occlusion)")
    when the occlusion call is withdrawn — otherwise the word "Occlusion" would still reach the report."""
    res["evidence_status"] = ev["evidence_status"]
    res["low_evidence"] = ev["evidence_status"] == "LOW_EVIDENCE"
    res["evidence_frames_passed"] = ev["frames_passed_gate"]
    res["evidence_threshold"] = ev["evidence_threshold"]
    if not res["low_evidence"]:
        return res
    note = (f"Seen in only {ev['frames_passed_gate']} diagnostic-quality frame(s) (reliable needs >= "
            f"{ev['evidence_threshold']}): possible small vessel; treat its numbers as low confidence.")
    if ev.get("branch_like"):
        note += " It runs right beside another vessel, so it is likely a small branch."
    res["branch_like"] = bool(ev.get("branch_like", False))
    if res.get("occlusion_flag"):
        res.update({"occlusion_flag": False, "occlusion_subtype": None,
                    "interpretation": "LOW_EVIDENCE_NOT_ASSESSED"})
        if anatomy_fn is not None:
            anat = anatomy_fn()
            res.update({"anatomical_segment": anat["anatomical_segment"],
                        "anatomical_segment_ar": anat["anatomical_segment_ar"],
                        "anatomical_region": anat["anatomical_region"],
                        "anatomical_short": anat["short_name"]})
        note += " An occlusion pattern was seen but is NOT reported on this little evidence."
    if ev.get("possible_edge_artifact"):
        note += (" It also looks like a subtraction edge (dark line with a bright rim), e.g. a bone edge "
                 "revealed by patient motion: may not be a vessel at all.")
    res["possible_edge_artifact"] = bool(ev.get("possible_edge_artifact", False))
    res["requires_visual_review"] = True
    res["review_reason"] = note if not res.get("review_reason") else res["review_reason"] + " | " + note
    return res


# ──────────────────────────────────────────────────────────
#  PIPELINE ORCHESTRATION
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
#  CSV DETECTION LOADER & PIPELINE ORCHESTRATION
# ──────────────────────────────────────────────────────────

def load_detections_from_csv(
    csv_path_or_df: Union[str, pd.DataFrame],
    default_w: int = 1280,
    default_h: int = 1280,
) -> Tuple[List[List[CandidateDetection]], int, int]:
    """
    Loads candidate detections directly from a pre-segmented CSV or DataFrame.
    Enables instant testing, validation, and benchmarking across pre-computed cine detections.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df, encoding="utf-8", encoding_errors="replace")
    else:
        df = csv_path_or_df.copy()

    df_det = df[df["vessel_id"].notna() & df["area"].notna() & (df["area"] > 0)].copy()
    if df_det.empty:
        return [], default_w, default_h

    max_x = df_det["cx"].max() if "cx" in df_det.columns else 1000
    max_y = df_det["cy"].max() if "cy" in df_det.columns else 1000
    eff_w = default_w if (max_x > 600 or max_y > 600) else 512
    eff_h = default_h if (max_x > 600 or max_y > 600) else 512

    detections_by_frame_dict: Dict[int, List[CandidateDetection]] = {}

    for _, row in df_det.iterrows():
        f_idx = int(row["frame"]) if "frame" in row and pd.notna(row["frame"]) else 0
        h_val = int(row["height_px"]) if "height_px" in row and pd.notna(row["height_px"]) else 50
        w_val = int(row.get("w", max(10, h_val // 10))) if "w" in row and pd.notna(row.get("w")) else max(10, h_val // 10)
        cx_val = int(row["cx"]) if "cx" in row and pd.notna(row["cx"]) else 0
        cy_val = int(row["cy"]) if "cy" in row and pd.notna(row["cy"]) else 0
        top_val = int(cy_val - h_val // 2)
        bot_val = int(cy_val + h_val // 2)
        left_val = int(cx_val - w_val // 2)
        mask_reg = np.zeros((eff_h, eff_w), dtype=np.uint8)
        mask_reg[max(0, top_val):min(eff_h, bot_val), max(0, left_val):min(eff_w, right_val)] = 1

        det = CandidateDetection(
            frame_idx=f_idx,
            mask_region=mask_reg,
            top=top_val,
            bot=bot_val,
            left=left_val,
            right=right_val,
            w=w_val,
            h=h_val,
            cx=cx_val,
            cy=cy_val,
            area=float(row["area"]),
            continuity=float(row.get("continuity", 1.0)),
            DS_pct=float(row["DS_pct"]) if "DS_pct" in row and pd.notna(row["DS_pct"]) else None,
            MLD_px=float(row["MLD_px"]) if "MLD_px" in row and pd.notna(row["MLD_px"]) else None,
            RVD_px=float(row["RVD_px"]) if "RVD_px" in row and pd.notna(row["RVD_px"]) else None,
            stenosis_length_px=float(row["stenosis_length_px"]) if "stenosis_length_px" in row and pd.notna(row["stenosis_length_px"]) else 0.0,
            sten_x=float(row["stenosis_x"]) if "stenosis_x" in row and pd.notna(row["stenosis_x"]) else float(cx_val),
            sten_y=float(row["stenosis_y"]) if "stenosis_y" in row and pd.notna(row["stenosis_y"]) else float(cy_val),
            contrast_score_raw=float(row["contrast_score"]) if "contrast_score" in row and pd.notna(row["contrast_score"]) else 20.0,
        )
        if f_idx not in detections_by_frame_dict:
            detections_by_frame_dict[f_idx] = []
        detections_by_frame_dict[f_idx].append(det)

    frames_sorted = sorted(detections_by_frame_dict.keys())
    detections_by_frame = [detections_by_frame_dict[f] for f in frames_sorted]
    return detections_by_frame, eff_w, eff_h


def run_frameselect_pipeline(
    input_source: Union[str, pd.DataFrame],
    output_csv_path: Optional[str] = None,
    mm_per_px: Optional[float] = None,
    model: Optional[Any] = None,
    device: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Executes the frame-selection vessel measurement pipeline.

    Args:
        input_source: Video file path (.mp4) or detection CSV file path (.csv) or DataFrame
        output_csv_path: Optional path to save summary CSV
        mm_per_px: Optional calibration factor in mm per pixel
        model: Optional pre-loaded PyTorch model
        device: Optional PyTorch device

    Returns:
        DataFrame with one row per detected vessel
    """
    gray_frames: Dict[int, np.ndarray] = {}
    baseline_gray: Optional[np.ndarray] = None
    is_csv_input = False

    if isinstance(input_source, pd.DataFrame) or (isinstance(input_source, str) and input_source.lower().endswith(".csv")):
        is_csv_input = True
        logger.info(f"[FrameSelect] Loading pre-segmented detections from CSV: {input_source if isinstance(input_source, str) else 'DataFrame'}...")
        detections_by_frame, orig_w, orig_h = load_detections_from_csv(input_source)
        video_id = os.path.basename(str(input_source)) if isinstance(input_source, str) else None
        if mm_per_px is None and video_id:
            mm_per_px = get_calibration_for_video(video_id)
    else:
        video_path = str(input_source)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if mm_per_px is None:
            mm_per_px = get_calibration_for_video(video_path)

        if model is None:
            model = get_model()
        if device is None:
            device = _get_device()
        model.eval()

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"[FrameSelect] Processing {video_path} ({orig_w}x{orig_h}, {total_frames} frames)...")

        raw_frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()

        if not raw_frames:
            logger.warning("[FrameSelect] Video contains no readable frames.")
            empty_df = pd.DataFrame(columns=[
                "vessel_id", "num_candidate_frames_passed_gate", "num_frames_used_in_top_k",
                "best_frame_index", "DS_pct", "MLD_mm", "RVD_mm", "lesion_length_mm",
                "angiographic_severity_index", "interpretation", "occlusion_flag", "calibration_status"
            ])
            if output_csv_path:
                empty_df.to_csv(output_csv_path, index=False)
            return empty_df

        gray_frames = {i: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for i, f in enumerate(raw_frames)}
        baseline_gray = gray_frames[0] if 0 in gray_frames else None

        detections_by_frame = []
        for i in range(0, len(raw_frames), BATCH_SIZE):
            batch_frames = raw_frames[i:i + BATCH_SIZE]
            tensors = [preprocess(f) for f in batch_frames]
            batch_tensor = torch.cat(tensors, dim=0).to(device)

            with torch.no_grad():
                preds = model(batch_tensor)
                probs = torch.sigmoid(preds).cpu().numpy()

            for b_idx in range(len(batch_frames)):
                f_idx = i + b_idx
                prob_map = probs[b_idx, 0]
                bin_mask = postprocess(prob_map, orig_h, orig_w, threshold=0.50)

                gf = gray_frames.get(f_idx)
                qualified = qualify_vessels(bin_mask, orig_h, orig_w, gray_frame=gf, baseline_frame=baseline_gray)

                frame_dets: List[CandidateDetection] = []
                for v in qualified:
                    c_val = compute_contrast_component(gf, v["mask_region"], baseline_gray=baseline_gray) if gf is not None else 20.0
                    frame_dets.append(CandidateDetection(
                        frame_idx=f_idx,
                        mask_region=v["mask_region"],
                        top=v["top"],
                        bot=v["bot"],
                        left=v["left"],
                        right=v["right"],
                        w=v["w"],
                        h=v["h"],
                        cx=v["cx"],
                        cy=v["cy"],
                        area=float(v["area"]),
                        continuity=float(v["continuity"]),
                        contrast_score_raw=c_val,
                    ))
                detections_by_frame.append(frame_dets)

            if (min(i + BATCH_SIZE, len(raw_frames))) % 20 == 0 or (i + BATCH_SIZE) >= len(raw_frames):
                logger.info(f"[FrameSelect] Segmented {min(i + BATCH_SIZE, len(raw_frames))}/{len(raw_frames)} frames...")

    # Spatial clustering of candidate detections across frames
    raw_clusters = group_vessels_spatially(detections_by_frame, orig_w, orig_h, gray_frames=gray_frames)
    spatial_clusters = filter_significant_vessels(raw_clusters, orig_w, orig_h)
    logger.info(f"[FrameSelect] Identified {len(spatial_clusters)} reportable spatial vessel region(s) (filtered from {len(raw_clusters)} raw cluster(s)).")

    # Evaluate each spatial cluster (gate + scoring)
    evaluated_clusters: Dict[int, List[CandidateDetection]] = {}
    for c_id, dets in spatial_clusters.items():
        evaluated_dets, peak_h, peak_a = evaluate_and_score_candidates(
            dets, orig_w, orig_h, gray_frames, baseline_gray=baseline_gray
        )
        evaluated_clusters[c_id] = evaluated_dets

    # Evidence triage BEFORE topology: drop duplicates / edge artifacts, flag weak vessels, renumber 1..n
    evaluated_clusters, evidence_info, suppressed_detections = triage_vessel_evidence(
        evaluated_clusters, gray_frames or None, len(detections_by_frame), orig_w, orig_h
    )

    # Detect parent-child bifurcation topology once per video from pooled evidence (kept vessels only)
    bifurcation_map = detect_geometric_bifurcations(evaluated_clusters, orig_w, orig_h)

    # Aggregate measurement per vessel cluster
    results: List[Dict[str, Any]] = []
    for c_id, dets in evaluated_clusters.items():
        bif_info = bifurcation_map.get(c_id, {"is_branch": False, "parent_id": None})
        summary_row = aggregate_vessel_measurement(
            vessel_id=c_id,
            cluster_detections=dets,
            all_clusters_detections=evaluated_clusters,
            orig_w=orig_w,
            orig_h=orig_h,
            mm_per_px=mm_per_px,
            is_branch=bif_info["is_branch"],
            parent_id=bif_info["parent_id"],
        )
        results.append(apply_evidence_status(
            summary_row, evidence_info[c_id],
            anatomy_fn=lambda c_id=c_id, dets=dets: classify_arterial_anatomy(
                c_id, dets, evaluated_clusters, orig_w, orig_h, is_occlusion=False)))

    # Main arteries only (after the occlusion decision; occlusions are never hidden)
    results, evaluated_clusters, evidence_info, _minor = select_main_arteries(results, evaluated_clusters, evidence_info, orig_h)

    df_out = pd.DataFrame(results)

    # Order columns explicitly matching requested output format
    column_order = [
        "vessel_id",
        "is_branch",
        "parent_id",
        "anatomical_segment",
        "anatomical_region",
        "num_candidate_frames_passed_gate",
        "num_frames_used_in_top_k",
        "best_frame_index",
        "DS_pct",
        "MLD_mm",
        "RVD_mm",
        "lesion_length_mm",
        "angiographic_severity_index",
        "interpretation",
        "occlusion_flag",
        "occlusion_subtype",
        "interruption_status",
        "interruption_gap_px",
        "interruption_frames",
        "requires_visual_review",
        "review_reason",
        "evidence_status",
        "evidence_frames_passed",
        "calibration_status",
    ]
    for col in column_order:
        if col not in df_out.columns:
            df_out[col] = None
    df_out = df_out[column_order]

    if output_csv_path:
        df_out.to_csv(output_csv_path, index=False)
        logger.info(f"[FrameSelect] Output exported to: {output_csv_path}")

    return df_out


# ──────────────────────────────────────────────────────────
#  API SERVER HIGH-LEVEL INTEGRATION
# ──────────────────────────────────────────────────────────

def format_summary_frameselect(summary: Dict[str, Any]) -> str:
    """Formats clinical summary text for the frame-selection pipeline (severity reported as ASI only)."""
    per_vessel = summary.get("per_vessel", {})
    lines = []
    lines.append("=" * 65)
    lines.append(f"  PAD FRAME-SELECTION RESULTS — {len(per_vessel)} Vessel Region(s)")
    lines.append("  Stenosis Method : QCA Diameter Stenosis (DS%) from Top Diagnostic Frames")
    lines.append("  Severity Metric : Angiographic Severity Index (ASI)")
    lines.append(f"  Calibration     : {summary.get('calibration_note', 'uncalibrated')}")
    lines.append("=" * 65)

    for vid, v in per_vessel.items():
        anat_label = v.get("anatomical_segment") or f"Vessel #{v['vessel_id']}"
        anat_ar = f" ({v.get('anatomical_segment_ar')})" if v.get("anatomical_segment_ar") else ""
        region_str = f" [{v.get('anatomical_region')}]" if v.get("anatomical_region") else ""
        lines.append(f"\n  Vessel #{v['vessel_id']} : {anat_label}{anat_ar}{region_str}")
        if v.get("is_branch") and v.get("parent_id") is not None:
            lines.append(f"    Vessel Topology    : Branch of Vessel #{v['parent_id']}")
        else:
            lines.append(f"    Vessel Topology    : Independent Arterial Conduit")
        lines.append(f"    Diagnostic Quality : {v.get('frame_count', 0)} frames passed gate")
        interp = v.get("interpretation", "NORMAL")
        lines.append(f"    Clinical Status    : {interp}")
        if v.get("requires_visual_review"):
            lines.append(f"    ⚠ VISUAL REVIEW    : {v.get('review_reason')}")
        if v.get("occlusion_flag") or interp == "LIKELY_OCCLUSION_LUMEN_UNRESOLVED":
            sub = v.get("occlusion_subtype")
            if sub in ("TOTAL_NO_FLOW", "SUBTOTAL_FAINT_FLOW"):
                lines.append(f"    Occlusion       : Flagged — mid-vessel interruption ~{v.get('interruption_gap_px')}px "
                             f"({'no contrast in gap' if sub == 'TOTAL_NO_FLOW' else 'faint contrast in gap'}, "
                             f"{v.get('interruption_frames')} frames)")
            else:
                lines.append("    Occlusion       : Flagged (Lumen unresolvable / distal cutoff)")
            lines.append("    Stenosis DS%    : Unresolved (Protected against false quantification)")
            lines.append("    Severity Index  : Unresolved")
        else:
            ds = v.get("max_DS_pct")
            ds_str = f"{ds:.1f}%" if ds is not None else "N/A"
            lines.append(f"    Stenosis DS%    : {ds_str}")
            asi = v.get("hemodynamic_risk_estimate")
            asi_str = f"{asi:.4f}" if asi is not None else "N/A"
            lines.append(f"    Severity Index  : {asi_str}")
            if v.get("MLD_mm") is not None and v.get("RVD_mm") is not None:
                lines.append(f"    MLD / RVD       : {v['MLD_mm']:.2f} mm / {v['RVD_mm']:.2f} mm (calibrated)")
            elif v.get("MLD_px") is not None and v.get("RVD_px") is not None:
                lines.append(f"    MLD / RVD       : {v['MLD_px']:.1f} px / {v['RVD_px']:.1f} px [Uncalibrated]")
            unc = v.get("asi_ds_spread", {}) or {}
            if unc.get("real_std_dev_pct") is not None and unc.get("empirical_range_pct") is not None:
                lines.append(f"    Empirical Spread: Range [{unc['empirical_range_pct'][0]:.1f}%, {unc['empirical_range_pct'][1]:.1f}%], SD: ±{unc['real_std_dev_pct']:.1f}% ({unc.get('n_frames_analyzed', 0)} frames)")
        lines.append(f"    Action          : {v.get('recommendation', '')}")

    removed = summary.get("suppressed_detections") or []
    if removed:
        lines.append("\n  Detections removed as non-vessels (not reported above):")
        for r in removed:
            why = {"SUPPRESSED_DUPLICATE": "duplicate of a reported vessel / washout fragment",
                   "SUPPRESSED_MINOR_VESSEL": f"minor vessel, not one of the main arteries ({r.get('reason')})"}.get(r["evidence_status"], r["evidence_status"])
            lines.append(f"    - raw cluster {r['raw_cluster_id']}: {why} ({r['frames_passed_gate']} frames)")
    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


def run_inference_frameselect(
    input_video_path: str,
    threshold: float = 0.50,
    work_dir: Optional[str] = None,
    pixel_spacing_mm: Optional[float] = None,
    catheter_french_size: Optional[float] = None,
    catheter_diameter_px: Optional[float] = None,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    High-level API server entrypoint using the frame-selection pipeline.
    Produces overlay video, mask video, stats CSV, and structured summary dict
    matching api_server.py / AnalysisResponse requirements.
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="frameselect_")
    os.makedirs(work_dir, exist_ok=True)

    overlay_out = os.path.join(work_dir, "output_OVERLAY.mp4")
    mask_out = os.path.join(work_dir, "output_MASK.mp4")
    csv_out = os.path.join(work_dir, "output_stats.csv")

    # 1. Calibration computation
    scale_mm_per_px = None
    calibration_status = "uncalibrated_pixel_units_only"
    calibration_note = "Relative pixel measurements only (no millimeter reference provided)"

    if pixel_spacing_mm and pixel_spacing_mm > 0:
        scale_mm_per_px = float(pixel_spacing_mm)
        calibration_status = "catheter_calibrated"
        calibration_note = f"Calibrated via DICOM PixelSpacing: {scale_mm_per_px:.4f} mm/px"
    elif catheter_french_size and catheter_diameter_px and catheter_diameter_px > 0:
        catheter_mm = float(catheter_french_size) / 3.0
        scale_mm_per_px = catheter_mm / float(catheter_diameter_px)
        calibration_status = "catheter_calibrated"
        calibration_note = f"Calibrated via {catheter_french_size:.1f} Fr Catheter: {scale_mm_per_px:.4f} mm/px"
    else:
        auto_cal = get_calibration_for_video(input_video_path)
        if auto_cal is not None and auto_cal > 0:
            scale_mm_per_px = auto_cal
            calibration_status = "catheter_calibrated"
            calibration_note = f"Calibrated via known benchmark reference: {scale_mm_per_px:.4f} mm/px"

    # 2. Open video
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    globals()["VIDEO_FPS"] = float(fps) if fps and fps > 1 else 15.0   # delayed-filling check uses real seconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    # 3. Model inference and video writing
    device = _get_device()
    model = get_model()
    model.to(device)
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_overlay = cv2.VideoWriter(overlay_out, fourcc, fps, (orig_w, orig_h), isColor=True)
    w_mask = cv2.VideoWriter(mask_out, fourcc, fps, (orig_w, orig_h), isColor=False)

    detections_by_frame: List[List[CandidateDetection]] = []
    all_masks: List[np.ndarray] = []
    gray_frames = {i: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for i, f in enumerate(raw_frames)}
    baseline_gray = gray_frames[0] if 0 in gray_frames else None

    for i in range(0, len(raw_frames), BATCH_SIZE):
        batch_frames = raw_frames[i:i + BATCH_SIZE]
        tensors = [preprocess(f) for f in batch_frames]
        batch_tensor = torch.cat(tensors, dim=0).to(device)

        with torch.no_grad():
            preds = model(batch_tensor)
            probs = torch.sigmoid(preds).cpu().numpy()

        for b_idx in range(len(batch_frames)):
            f_idx = i + b_idx
            prob_map = probs[b_idx, 0]
            bin_mask = postprocess(prob_map, orig_h, orig_w, threshold=threshold)
            all_masks.append(bin_mask)
            w_mask.write(bin_mask)

            gf = gray_frames.get(f_idx)
            qualified = qualify_vessels(bin_mask, orig_h, orig_w, gray_frame=gf, baseline_frame=baseline_gray)
            frame_dets: List[CandidateDetection] = []
            for v in qualified:
                c_val = compute_contrast_component(gf, v["mask_region"], baseline_gray=baseline_gray) if gf is not None else 20.0
                frame_dets.append(CandidateDetection(
                    frame_idx=f_idx,
                    mask_region=v["mask_region"],
                    top=v["top"],
                    bot=v["bot"],
                    left=v["left"],
                    right=v["right"],
                    w=v["w"],
                    h=v["h"],
                    cx=v["cx"],
                    cy=v["cy"],
                    area=v["area"],
                    continuity=v["continuity"],
                    contrast_score_raw=c_val,
                ))
            detections_by_frame.append(frame_dets)

    # 4. Spatial clustering & vessel significance gating
    raw_clusters = group_vessels_spatially(detections_by_frame, orig_w, orig_h, gray_frames=gray_frames)
    clusters = filter_significant_vessels(raw_clusters, orig_w, orig_h)
    logger.info(f"[FrameSelect] Identified {len(clusters)} reportable vessel cluster(s) from {len(raw_clusters)} raw spatial regions.")

    evaluated_clusters: Dict[int, List[CandidateDetection]] = {}
    for cid, dets in clusters.items():
        evaluated_dets, _, _ = evaluate_and_score_candidates(
            dets, orig_w, orig_h, gray_frames, baseline_gray=baseline_gray
        )
        evaluated_clusters[cid] = evaluated_dets

    # Evidence triage BEFORE topology: drop duplicates / edge artifacts, flag weak vessels, renumber 1..n
    clusters, evidence_info, suppressed_detections = triage_vessel_evidence(
        evaluated_clusters, gray_frames, total_frames, orig_w, orig_h
    )
    logger.info(f"[FrameSelect] Evidence triage: {len(clusters)} vessel(s) kept, {len(suppressed_detections)} removed.")

    # Detect true geometric bifurcation topology once across pooled evidence (kept vessels only)
    bifurcation_map = detect_geometric_bifurcations(clusters, orig_w, orig_h)

    per_vessel_results: List[Dict[str, Any]] = []
    diagnostic_frame_indices: Set[int] = set()

    for cid, dets in clusters.items():
        bif_info = bifurcation_map.get(cid, {"is_branch": False, "parent_id": None})
        res = aggregate_vessel_measurement(
            vessel_id=cid,
            cluster_detections=dets,
            all_clusters_detections=clusters,
            orig_w=orig_w,
            orig_h=orig_h,
            mm_per_px=scale_mm_per_px,
            is_branch=bif_info["is_branch"],
            parent_id=bif_info["parent_id"],
        )
        res = apply_evidence_status(
            res, evidence_info[cid],
            anatomy_fn=lambda cid=cid, dets=dets: classify_arterial_anatomy(
                cid, dets, clusters, orig_w, orig_h, is_occlusion=False))
        per_vessel_results.append(res)
        if res.get("best_frame_index") is not None:
            diagnostic_frame_indices.add(res["best_frame_index"])
        passed = [d for d in dets if d.passed_hard_gate]
        top_k = select_best_frames(passed, k=K_BEST_FRAMES)
        for d in top_k:
            diagnostic_frame_indices.add(d.frame_idx)

    # 4a. Main arteries only (decided after the occlusion decision; occlusions are never hidden)
    per_vessel_results, clusters, evidence_info, minor_vessels = select_main_arteries(
        per_vessel_results, clusters, evidence_info, orig_h)
    if minor_vessels:
        suppressed_detections = list(suppressed_detections) + minor_vessels
        diagnostic_frame_indices = set()
        for r in per_vessel_results:
            if r.get("best_frame_index") is not None:
                diagnostic_frame_indices.add(r["best_frame_index"])
            for d in select_best_frames([d for d in clusters[r["vessel_id"]] if d.passed_hard_gate], k=K_BEST_FRAMES):
                diagnostic_frame_indices.add(d.frame_idx)
    # 4b. Save high-resolution diagnostic frame images (Raw, Mask, QCA Overlay)
    frames_dir = os.path.join(work_dir, "diagnostic_frames")
    os.makedirs(frames_dir, exist_ok=True)
    diagnostic_frames_list: List[Dict[str, Any]] = []

    for cid, dets in clusters.items():
        res = next((r for r in per_vessel_results if r["vessel_id"] == cid), None)
        if not res:
            continue
        best_fidx = res.get("best_frame_index")
        passed = [d for d in dets if d.passed_hard_gate]
        top_k = select_best_frames(passed, k=K_BEST_FRAMES)
        
        for d in top_k:
            f_idx = d.frame_idx
            is_best = (f_idx == best_fidx)
            
            # Save raw frame
            raw_img = raw_frames[f_idx]
            raw_path = os.path.join(frames_dir, f"raw_{f_idx}.jpg")
            if not os.path.exists(raw_path):
                cv2.imwrite(raw_path, raw_img)
                
            # Save binary mask
            mask_img = all_masks[f_idx] if f_idx < len(all_masks) else np.zeros((orig_h, orig_w), dtype=np.uint8)
            mask_path = os.path.join(frames_dir, f"mask_{f_idx}.png")
            if not os.path.exists(mask_path):
                cv2.imwrite(mask_path, mask_img)
                
            # Build QCA diagnostic overlay
            qca_overlay = raw_img.copy()
            c_mask = np.zeros_like(qca_overlay)
            c_mask[mask_img > 0] = (0, 240, 100) # Bright green
            cv2.addWeighted(c_mask, 0.40, qca_overlay, 0.60, 0, qca_overlay)
            
            # Vessel box
            cv2.rectangle(qca_overlay, (d.left, d.top), (d.right, d.bot), (0, 255, 0), 3)
            
            # Stenosis MLD mark
            if d.sten_x is not None and d.sten_y is not None:
                sx, sy = int(d.sten_x), int(d.sten_y)
                cv2.circle(qca_overlay, (sx, sy), 7, (0, 0, 255), -1)
                mld_val = d.MLD_px or 0.0
                cv2.line(qca_overlay, (int(sx - mld_val / 2), sy), (int(sx + mld_val / 2), sy), (0, 0, 255), 3)
                cv2.putText(qca_overlay, f"MLD: {mld_val:.1f}px", (sx + 12, sy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Top Banner
            cv2.rectangle(qca_overlay, (0, 0), (orig_w, 70), (20, 20, 20), -1)
            anat_short = res.get("anatomical_short", f"V#{cid}")
            low_ev = bool(res.get("low_evidence", False))
            role_text = f"Vessel #{cid} [{anat_short}] — {'BEST DIAGNOSTIC FRAME' if is_best else 'Top Frame'} — Frame #{f_idx}"
            if low_ev:
                role_text += "  [LOW EVIDENCE - REVIEW]"
            cv2.putText(qca_overlay, role_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255) if low_ev else (0, 230, 255), 2)
            if low_ev:
                sub_text = (f"Seen in only {res.get('evidence_frames_passed')} good frame(s): not a reliable vessel"
                            + (" - may be a bone/tissue edge" if res.get("possible_edge_artifact") else "")
                            + (" | stenosis NOT assessed" if res.get("DS_pct") is None else f" | DS {res['DS_pct']:.1f}% (low confidence)"))
                sub_color = (0, 165, 255)
            else:
                ds_str = f"DS: {d.DS_pct:.1f}%" if d.DS_pct is not None else "Cutoff / Occlusion"
                sub_text = f"Frame quality: {d.total_score:.3f} | {ds_str} | Status: {res['interpretation']} | Artery: {res.get('anatomical_segment', '')}"
                sub_color = (0, 255, 0) if not res['occlusion_flag'] else (0, 165, 255)
            cv2.putText(qca_overlay, sub_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sub_color, 2)
            
            overlay_path = os.path.join(frames_dir, f"overlay_{f_idx}_v{cid}.jpg")
            cv2.imwrite(overlay_path, qca_overlay)
            
            diagnostic_frames_list.append({
                "frame_idx": int(f_idx),
                "vessel_id": int(cid),
                "is_branch": bool(res.get("is_branch", False)),
                "parent_id": res.get("parent_id"),
                "anatomical_segment": res.get("anatomical_segment"),
                "anatomical_segment_ar": res.get("anatomical_segment_ar"),
                "anatomical_short": anat_short,
                "is_best": bool(is_best),
                "role": f"Vessel #{cid} [{anat_short}] ({'Best' if is_best else 'Top-k'})",
                "score": float(round(d.total_score, 3)),
                "ds_pct": (None if (res.get("low_evidence") and res.get("DS_pct") is None)
                           else float(round(d.DS_pct, 1)) if d.DS_pct is not None else None),
                "low_evidence": bool(res.get("low_evidence", False)),
                "possible_edge_artifact": bool(res.get("possible_edge_artifact", False)),
                "review_reason": res.get("review_reason"),
                "mld_px": float(round(d.MLD_px, 1)) if d.MLD_px is not None else None,
                "rvd_px": float(round(d.RVD_px, 1)) if d.RVD_px is not None else None,
                "interpretation": str(res["interpretation"]),
                "occlusion_flag": bool(res["occlusion_flag"]),
                "raw_file": f"raw_{f_idx}.jpg",
                "mask_file": f"mask_{f_idx}.png",
                "overlay_file": f"overlay_{f_idx}_v{cid}.jpg",
            })

    # 5. Render overlay video
    for f_idx, frame_bgr in enumerate(raw_frames):
        overlay = frame_bgr.copy()
        mask = all_masks[f_idx] if f_idx < len(all_masks) else np.zeros((orig_h, orig_w), dtype=np.uint8)

        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = (0, 230, 100)
        cv2.addWeighted(colored_mask, 0.45, overlay, 0.55, 0, overlay)

        if f_idx in diagnostic_frame_indices:
            cv2.putText(
                overlay,
                "[DIAGNOSTIC FRAME - TOP SELECTION]",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for res in per_vessel_results:
            vid = res["vessel_id"]
            interp = res["interpretation"]
            occ = res["occlusion_flag"]
            ds = res["DS_pct"]
            anat_short = res.get("anatomical_short") or res.get("anatomical_segment", "").split("(")[0].strip() or f"V#{vid}"
            low_ev = bool(res.get("low_evidence", False))
            if low_ev:
                status_txt = (f"Vessel #{vid} [{anat_short}]: LOW EVIDENCE - needs review"
                              + (" (possible bone/tissue edge)" if res.get("possible_edge_artifact") else ""))
            else:
                status_txt = f"Vessel #{vid} [{anat_short}]: {interp}"
                if occ:
                    status_txt += " (OCCLUSION)"
                elif ds is not None:
                    status_txt += f" ({ds:.1f}% DS)"
            cv2.putText(
                overlay,
                status_txt,
                (20, 75 + (vid - 1) * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255) if low_ev else ((0, 0, 255) if occ else (0, 255, 0)),
                2,
                cv2.LINE_AA,
            )

        w_overlay.write(overlay)

    w_overlay.release()
    w_mask.release()

    # 6. Export stats CSV
    df_out = pd.DataFrame(per_vessel_results)
    df_out.to_csv(csv_out, index=False)

    # 7. Build structured summary matching api_server schema
    per_vessel_data = {}
    for res in per_vessel_results:
        vid_str = str(res["vessel_id"])
        interp = res["interpretation"]
        occ = bool(res["occlusion_flag"])
        ds_val = float(res["DS_pct"]) if res["DS_pct"] is not None and not np.isnan(res["DS_pct"]) else None
        asi_val = float(res["angiographic_severity_index"]) if res["angiographic_severity_index"] is not None and not np.isnan(res["angiographic_severity_index"]) else None
        mld_mm = float(res["MLD_mm"]) if res["MLD_mm"] is not None and not np.isnan(res["MLD_mm"]) else None
        rvd_mm = float(res["RVD_mm"]) if res["RVD_mm"] is not None and not np.isnan(res["RVD_mm"]) else None
        slen_mm = float(res["lesion_length_mm"]) if res["lesion_length_mm"] is not None and not np.isnan(res["lesion_length_mm"]) else None

        if occ:
            rec = "Likely total or subtotal arterial occlusion — urgent revascularization indicated"
            risk_label = "Unresolved (Total Occlusion)"
        elif ds_val is not None:
            if ds_val >= 70.0:
                rec = "Severe stenosis (≥70%) — consider revascularization"
            elif ds_val >= 50.0:
                rec = "Moderate stenosis (50-69%) — clinical context & Duplex ultrasound"
            elif ds_val >= 20.0:
                rec = "Mild stenosis (20-49%) — conservative medical therapy"
            else:
                rec = "No significant stenosis (<20%) — no intervention needed"
            risk_label = f"Angiographic Severity Index: {asi_val:.2f}" if asi_val is not None else "Standard Risk"
        else:
            rec = "Insufficient diagnostic quality frames for definitive assessment"
            risk_label = "Indeterminate"

        is_br = bool(res.get("is_branch", False))
        p_id = res.get("parent_id", None)

        cand_ds_list = res.get("selected_ds_values", [])
        if len(cand_ds_list) >= 2 and not occ and ds_val is not None:
            unc_obj = {
                "real_std_dev_pct": res.get("real_std_dev_pct"),
                "empirical_range_pct": res.get("empirical_range_pct"),
                "n_frames_analyzed": len(cand_ds_list),
                "note": f"Empirical spread across the {len(cand_ds_list)} selected diagnostic frames",
            }
        else:
            unc_obj = {
                "real_std_dev_pct": None,
                "empirical_range_pct": None,
                "n_frames_analyzed": len(cand_ds_list) if cand_ds_list else (1 if not occ and ds_val is not None else 0),
                "note": None,
            }

        per_vessel_data[vid_str] = {
            "vessel_id": res["vessel_id"],
            "anatomical_segment": res.get("anatomical_segment"),
            "anatomical_segment_ar": res.get("anatomical_segment_ar"),
            "anatomical_region": res.get("anatomical_region"),
            "anatomical_short": res.get("anatomical_short"),
            "is_branch": is_br,
            "parent_id": p_id,
            "frame_count": res["num_candidate_frames_passed_gate"],
            "mean_asi_pct": round(asi_val * 100.0, 1) if asi_val is not None else None,
            "min_asi_pct": round(asi_val * 100.0, 1) if asi_val is not None else None,
            "hemodynamic_risk_estimate": asi_val,
            "hemodynamic_risk_label": risk_label,
            "asi_ds_spread": unc_obj,
            "mean_DS_pct": ds_val,
            "max_DS_pct": ds_val,
            "max_DS_frame": res["best_frame_index"],
            "MLD_px": None,
            "RVD_px": None,
            "stenosis_length_px": None,
            "MLD_mm": mld_mm,
            "RVD_mm": rvd_mm,
            "stenosis_length_mm": slen_mm,
            "is_calibrated": bool(scale_mm_per_px is not None),
            "calibration_tag": calibration_status,
            "interpretation": interp,
            "recommendation": rec,
            "ds_classification": interp,
            "stenosis_status": interp,
            "anatomical_functional_disagreement": False,
            "low_contrast_only": False,
            "appearance_ratio": round(res["num_candidate_frames_passed_gate"] / max(1, total_frames), 3),
            "tracking_coverage": round(res["num_candidate_frames_passed_gate"] / max(1, total_frames), 3),
            "quality_gate_passed": bool(res["num_candidate_frames_passed_gate"] > 0),
            "evidence_level": "MANY_DIAGNOSTIC_FRAMES" if res["num_candidate_frames_passed_gate"] >= 10 else "FEW_DIAGNOSTIC_FRAMES",
            "evidence_reason": f"{res['num_candidate_frames_passed_gate']} frames passed the quality gate (top {res['num_frames_used_in_top_k']} used). Frame count only, not a statistical confidence.",
            "requires_visual_review": bool(res.get("requires_visual_review", False)),
            "review_reason": res.get("review_reason"),
            "occlusion_subtype": res.get("occlusion_subtype"),
            "interruption_status": res.get("interruption_status"),
            "interruption_gap_px": res.get("interruption_gap_px"),
            "interruption_frames": res.get("interruption_frames"),
            "evidence_status": res.get("evidence_status"),
            "low_evidence": bool(res.get("low_evidence", False)),
            "evidence_frames_passed": res.get("evidence_frames_passed"),
            "evidence_threshold": res.get("evidence_threshold"),
            "possible_edge_artifact": bool(res.get("possible_edge_artifact", False)),
            "branch_like": bool(res.get("branch_like", False)),
            "is_borderline": bool(ds_val is not None and 48.0 <= ds_val <= 52.0),
            "is_late_washout_bone": False,
            "is_transient_flicker": False,
            "transient_low_confidence": False,
            "branch_classification": "confirmed_branch" if is_br else None,
            "branch_stability_ratio": 1.0 if is_br else None,
            "parent_frame_count_during_overlap": None,
            "cto_suspected": occ,
            "cto_trigger_frame": res["best_frame_index"],
            "cto_narrowest_pt": None,
            "is_total_occlusion": occ,
            "occlusion_flag": occ,
            "best_measurement_frame": res["best_frame_index"],
            "best_measurement_frame_score_breakdown": None,
        }

    summary = {
        "success": True,
        "total_frames": total_frames,
        "analyzed_frames": len(raw_frames),
        "n_confirmed_vessels": len(per_vessel_data),
        "per_vessel": per_vessel_data,
        "suppressed_detections": [{k: v for k, v in r.items() if k != "detections"} for r in suppressed_detections],
        "bifurcation_events_detected_during_tracking": sum(1 for r in per_vessel_results if r.get("is_branch")),
        "possible_preexisting_bifurcations": sum(1 for r in per_vessel_results if r.get("is_branch")),
        "loops_excluded": 0,
        "stenosis_method": "Frame-Selection QCA (Top-k diagnostic frames)",
        "severity_formula": "Angiographic Severity Index (ASI) = geometric function of DS% and lesion length; NOT a physiological pressure measurement",
        "severity_estimation_method": "Top-k diagnostic frame selection (no continuous tracking)",
        "physiological_limitations_note": "Evaluated on peak-contrast diagnostic frames with hard quality gates to exclude bolus wavefront artifacts.",
        "calibration_status": calibration_status,
        "calibration_note": calibration_note,
        "scale_mm_per_px": scale_mm_per_px,
        "video_info": {
            "width": orig_w,
            "height": orig_h,
            "fps": fps,
        },
        "diagnostic_frames": diagnostic_frames_list,
    }

    try:
        debug = {
            "video": os.path.basename(str(input_video_path)),
            "total_frames": total_frames,
            "evidence_threshold": evidence_threshold(total_frames),
            "vessels": [
                {
                    "vessel_id": r["vessel_id"],
                    "interpretation": r.get("interpretation"),
                    "occlusion_rule": r.get("occlusion_rule"),
                    "cutoff_detail": r.get("cutoff_detail"),
                    "interruption": {k: r.get(k) for k in ("interruption_status", "interruption_gap_px", "interruption_frames",
                                                           "interruption_persistence", "interruption_votes")},
                    "evidence": evidence_info.get(r["vessel_id"]),
                    "is_branch": r.get("is_branch"), "parent_id": r.get("parent_id"),
                    "detections": _describe_detections(clusters[r["vessel_id"]]),
                }
                for r in per_vessel_results
            ],
            "suppressed": suppressed_detections,
        }
        with open(os.path.join(work_dir, "debug_clusters.json"), "w", encoding="utf-8") as fh:
            json.dump(debug, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    except Exception as exc:                      # debug output must never break the analysis itself
        logger.warning(f"[Debug] could not write debug_clusters.json: {exc}")
    return overlay_out, mask_out, csv_out, summary


# ──────────────────────────────────────────────────────────
#  CLI ENTRYPOINT
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Frame-Selection PAD Angiography Vessel Measurement Pipeline"
    )
    parser.add_argument("video_path", type=str, help="Path to input angiogram video (.mp4)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path")
    parser.add_argument("--mm-per-px", type=float, default=None, help="Catheter calibration factor (mm/px)")
    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(args.video_path)
        args.output = f"{base}_frameselect.csv"

    df_res = run_frameselect_pipeline(
        input_source=args.video_path,
        output_csv_path=args.output,
        mm_per_px=args.mm_per_px,
    )
    print("\n--- Pipeline Measurement Results ---")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
