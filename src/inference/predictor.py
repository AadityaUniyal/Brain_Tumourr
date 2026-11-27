"""
Unified Clinical Predictor:
Coordinates preprocessor, model inference, energy OOD gate,
conformal uncertainty, tumor volume quantification, and Grad-CAM generation.
"""
import torch
import numpy as np
import cv2

from src.data.preprocessor import MedicalImagePreprocessor
from src.models.dual_head_net import DualHeadBrainTumorNet
from src.inference.explainability import GradCAM
from src.inference.ood_detector import EnergyOODDetector
from src.inference.conformal import ConformalPredictor
from src.inference.selective_classifier import SelectiveClassifier

CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]


class ClinicalPredictor:
    def __init__(
        self,
        model=None,
        checkpoint_path=None,
        device=None,
        ood_threshold=-6.5,
        conformal_alpha=0.05,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = MedicalImagePreprocessor(target_size=(256, 256))

        if model is None:
            self.model = DualHeadBrainTumorNet(num_classes=len(CLASS_NAMES), pretrained=True)
            if checkpoint_path is not None:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
        else:
            self.model = model

        self.model.to(self.device).eval()
        self.grad_cam = GradCAM(self.model)
        self.ood_detector = EnergyOODDetector(energy_threshold=ood_threshold)
        self.conformal = ConformalPredictor(alpha=conformal_alpha, class_names=CLASS_NAMES)
        self.selective_classifier = SelectiveClassifier(confidence_threshold=0.80, energy_threshold=ood_threshold)

    def predict(self, file_source, filename="scan.jpg", slice_thickness_mm=3.0, pixel_spacing_mm=0.8):
        """
        Complete end-to-end diagnostic inference:
        1. Parse medical format (DICOM / NIfTI / MAT / standard)
        2. Normalize & resize
        3. Model forward pass (Segmentation + Classification)
        4. OOD safety check
        5. Calibrated probabilities & Conformal prediction set
        6. Tumor area & volume estimation
        7. Grad-CAM saliency extraction
        """
        # Preprocess
        preprocessed_2d = self.preprocessor.preprocess(file_source, filename=filename)

        # Tensor conversion [1, 3, 256, 256]
        img_3c = np.repeat(np.expand_dims(preprocessed_2d, axis=0), 3, axis=0)
        tensor = torch.tensor(img_3c, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.set_grad_enabled(True):
            outputs = self.model(tensor)
            logits = outputs["logits"]
            seg_logits = outputs["seg_mask"]

        # OOD Safety Check
        is_ood, energy_score, ood_msg = self.ood_detector.check(logits)

        # Probabilities
        probs = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        # Conformal Prediction Set
        conformal_res = self.conformal.predict_set(probs)

        # Segmentation Mask & Volume estimation
        seg_mask = torch.sigmoid(seg_logits).squeeze().detach().cpu().numpy()
        binary_mask = (seg_mask > 0.5).astype(np.uint8)

        # Area & Volume Calculation
        pixel_count = int(np.sum(binary_mask))
        pixel_area_mm2 = pixel_spacing_mm * pixel_spacing_mm
        tumor_area_mm2 = pixel_count * pixel_area_mm2
        # Estimated slice volume in cm^3: Area (mm^2) * thickness (mm) / 1000
        tumor_volume_cm3 = (tumor_area_mm2 * slice_thickness_mm) / 1000.0

        # Grad-CAM Heatmap
        heatmap, _ = self.grad_cam.generate_heatmap(tensor, target_class=pred_idx)
        cam_overlay = self.grad_cam.overlay_on_image(preprocessed_2d, heatmap, alpha=0.45)

        class_distribution = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

        # Evaluate Conformal Risk-Controlled Selective Classification & Triage
        selective_res = self.selective_classifier.evaluate_decision(
            predicted_class=pred_label,
            class_probabilities=class_distribution,
            conformal_set=conformal_res["prediction_set"],
            energy_score=energy_score,
            is_ood=is_ood,
            tumor_volume_cm3=tumor_volume_cm3,
        )

        return {
            "status": "success",
            "filename": filename,
            "is_ood": is_ood,
            "ood_message": ood_msg,
            "energy_score": energy_score,
            "diagnostic": {
                "predicted_tumor_type": pred_label,
                "confidence": confidence,
                "class_probabilities": class_distribution,
                "conformal_prediction_set": conformal_res["prediction_set"],
                "is_clinically_uncertain": conformal_res["is_uncertain"],
                "coverage_guarantee": conformal_res["confidence_level"],
            },
            "quantification": {
                "tumor_pixel_count": pixel_count,
                "tumor_area_mm2": round(tumor_area_mm2, 2),
                "estimated_volume_cm3": round(tumor_volume_cm3, 3),
            },
            "selective_triage": selective_res,
            "visualizations": {
                "raw_slice": preprocessed_2d,
                "seg_mask": seg_mask,
                "binary_mask": binary_mask,
                "gradcam_heatmap": heatmap,
                "gradcam_overlay": cam_overlay,
            },
        }
