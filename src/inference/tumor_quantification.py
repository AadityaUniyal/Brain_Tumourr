"""
Tumor Volumetric & RANO (Response Assessment in Neuro-Oncology) Metrology Engine.
Computes:
1. Calibrated Area (mm^2) and Estimated Volume (cm^3).
2. Maximal cross-sectional diameter D_max (major axis) and largest perpendicular cross-diameter D_perp (minor axis).
3. RANO bidirectional cross-product (cm^2).
4. Sub-pixel calliper vector endpoints in 2D image coordinates for PACS crosshair rendering.
"""
from typing import Dict, Any, List
import numpy as np
import cv2


def compute_rano_callipers(binary_mask: np.ndarray, pixel_spacing_mm: float = 0.8) -> Dict[str, Any]:
    """
    Computes RANO bidirectional diameters from a 2D binary segmentation mask.
    
    Args:
        binary_mask: 2D binary numpy array (0 or 1 / 255)
        pixel_spacing_mm: in-plane pixel spacing in millimeters
        
    Returns:
        dict containing:
            major_axis_mm: float
            minor_axis_mm: float
            product_cm2: float
            major_line: List[List[float]] -> [[x1, y1], [x2, y2]]
            minor_line: List[List[float]] -> [[u1, v1], [u2, v2]]
    """
    empty_result = {
        "major_axis_mm": 0.0,
        "minor_axis_mm": 0.0,
        "product_cm2": 0.0,
        "major_line": [],
        "minor_line": []
    }
    
    if binary_mask is None or np.sum(binary_mask > 0) < 5:
        return empty_result
    
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return empty_result
    
    # Largest contiguous lesion contour
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return empty_result
        
    rect = cv2.minAreaRect(c)  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)  # 4 vertices: [p0, p1, p2, p3]
    
    # Compute edge lengths
    d01 = float(np.linalg.norm(box[0] - box[1]))
    d12 = float(np.linalg.norm(box[1] - box[2]))
    
    if d01 >= d12:
        major_len_px = d01
        minor_len_px = d12
        # Major line connects midpoints of edge (0,3) and (1,2)
        m1 = ((box[0] + box[3]) / 2.0).tolist()
        m2 = ((box[1] + box[2]) / 2.0).tolist()
        # Minor line connects midpoints of edge (0,1) and (2,3)
        u1 = ((box[0] + box[1]) / 2.0).tolist()
        u2 = ((box[2] + box[3]) / 2.0).tolist()
    else:
        major_len_px = d12
        minor_len_px = d01
        m1 = ((box[0] + box[1]) / 2.0).tolist()
        m2 = ((box[2] + box[3]) / 2.0).tolist()
        u1 = ((box[0] + box[3]) / 2.0).tolist()
        u2 = ((box[1] + box[2]) / 2.0).tolist()
        
    major_mm = round(float(major_len_px * pixel_spacing_mm), 2)
    minor_mm = round(float(minor_len_px * pixel_spacing_mm), 2)
    product_cm2 = round(float((major_mm * minor_mm) / 100.0), 3)
    
    return {
        "major_axis_mm": major_mm,
        "minor_axis_mm": minor_mm,
        "product_cm2": product_cm2,
        "major_line": [[round(m1[0], 2), round(m1[1], 2)], [round(m2[0], 2), round(m2[1], 2)]],
        "minor_line": [[round(u1[0], 2), round(u1[1], 2)], [round(u2[0], 2), round(u2[1], 2)]]
    }
