import os
import io
import sys
import logging
from typing import Optional, Tuple, Dict, Any

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pickle

try:
    import joblib
except Exception:
    joblib = None  

try:
    from flask_cors import CORS
except Exception:  
    CORS = None  

try:
    import torch  
    import torch.nn.functional as F  
except Exception:  
    torch = None  
    F = None  

try:
    from PIL import Image
except Exception as pil_exc:  
    raise RuntimeError("Pillow (PIL) is required. Please install with: pip install pillow") from pil_exc

import numpy as np


# ---------------------------
# Application setup
# ---------------------------
app = Flask(__name__)
if CORS is not None:
    CORS(app)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("flood-backend")


# ---------------------------
# Model wrapper
# ---------------------------
class FloodModel:
    """Wrapper that loads a Torch model if available; otherwise uses a heuristic fallback.

    The expected Torch model should accept a 4D tensor (N, C, H, W) in RGB order,
    with float32 values in [0, 1], and output either:
      - a (N, 1) flood probability, or
      - a (N, 1, H, W) per-pixel mask from which we compute a probability.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.device = self._select_device(device)
        self.model = None
        self.sklearn_model = None
        self.model_output_is_mask = False

        if model_path is None:
            model_path = os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "models", "resnet_metrics.pkl"))

        self.model_path = model_path
        self._load_model_if_available()

    def _select_device(self, requested: Optional[str]) -> str:
        if torch is None:
            return "cpu"
        if requested:
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model_if_available(self) -> None:
        if torch is None:
            logger.info("PyTorch not installed or unavailable. Will try non-torch models or heuristic.")

        if not os.path.isfile(self.model_path):
            logger.warning("Model file not found at %s. Using heuristic fallback.", self.model_path)
            return

        
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == ".pkl":
            try:
                candidate = None
                if joblib is not None:
                    candidate = joblib.load(self.model_path)
                else:
                    with open(self.model_path, "rb") as f:
                        candidate = pickle.load(f)
                
                if any(hasattr(candidate, attr) for attr in ("predict_proba", "predict", "decision_function")):
                    self.sklearn_model = candidate
                    logger.info("Loaded sklearn/joblib model from %s", self.model_path)
                    return
                else:
                    logger.warning("Loaded object from %s is not a model (no predict methods). Ignoring.", self.model_path)
                    self.sklearn_model = None
            except Exception as exc:
                logger.exception("Failed to load .pkl model: %s", exc)
                self.sklearn_model = None

        
        if torch is not None:
            try:
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                self.model.eval()
                logger.info("Loaded TorchScript model from %s on %s", self.model_path, self.device)
                return
            except Exception:
                pass

            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if hasattr(checkpoint, "eval"):
                    self.model = checkpoint
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    raise RuntimeError(
                        "Found a checkpoint dict without model class. Please export TorchScript or provide model code."
                    )
                else:
                    raise RuntimeError("Unsupported model file format.")
                self.model.eval()
                logger.info("Loaded Torch model from %s on %s", self.model_path, self.device)
                return
            except Exception as exc:
                self.model = None
                logger.exception("Failed to load torch model at %s: %s.", self.model_path, exc)

        logger.warning("No compatible model loaded. Using heuristic fallback.")

    @staticmethod
    def _preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (384, 384)) -> np.ndarray:
        image = image.convert("RGB")
        if target_size is not None:
            image = image.resize(target_size)
        array = np.asarray(image).astype(np.float32) / 255.0  
        return array

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> float:
        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
        if hist.sum() == 0:
            return 0.5
        hist = hist / hist.sum()
        omega = np.cumsum(hist)
        mu = np.cumsum(hist * np.arange(256))
        mu_t = mu[-1]
        denom = omega * (1 - omega)
        denom[denom == 0] = 1e-9
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        idx = int(np.nanargmax(sigma_b2))
        return idx / 255.0

    def _predict_with_torch(self, image_array: np.ndarray) -> float:
        assert torch is not None and self.model is not None
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)

        # Handle various output shapes
        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.dim() == 4:  # (N, 1, H, W) mask
            self.model_output_is_mask = True
            if F is not None:
                output = torch.sigmoid(output)
            mask_prob = output.mean().item()
            return float(mask_prob)

        if output.dim() == 2:  # (N, 1) probability
            prob = torch.sigmoid(output).mean().item() if output.size(1) == 1 else torch.softmax(output, dim=1)[:, 1].mean().item()
            return float(prob)

        if output.dim() == 1:  # (N,) probability/logit
            if output.numel() == 1:
                value = output.item()
                try:
                    # treat as logit
                    return float(1.0 / (1.0 + np.exp(-float(value))))
                except Exception:
                    return float(value)
            # multi-class vector: assume index 1 is flood class
            return float(torch.softmax(output, dim=0)[1].item())

        # Fallback: mean after sigmoid
        return float(torch.sigmoid(output).mean().item())

    @staticmethod
    def _compute_features(image_array: np.ndarray) -> np.ndarray:
        red = image_array[:, :, 0]
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]
        gray = image_array.mean(axis=2)

        is_grayscale = bool(np.mean(np.abs(red - green)) < 0.005 and np.mean(np.abs(green - blue)) < 0.005)

        if is_grayscale:
            mean_g = float(gray.mean())
            std_g = float(gray.std())
            p25 = float(np.quantile(gray, 0.25))
            p50 = float(np.quantile(gray, 0.50))
            p75 = float(np.quantile(gray, 0.75))
            # Otsu threshold and simple gradient smoothness
            gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
            hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
            hist = hist / (hist.sum() if hist.sum() else 1.0)
            omega = np.cumsum(hist)
            mu = np.cumsum(hist * np.arange(256))
            mu_t = mu[-1]
            denom = omega * (1 - omega)
            denom[denom == 0] = 1e-9
            sigma_b2 = (mu_t * omega - mu) ** 2 / denom
            t_idx = int(np.nanargmax(sigma_b2))
            t_val = t_idx / 255.0
            dark = gray < max(0.15, min(0.6, t_val))
            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)
            gx[:, 1:] = np.diff(gray, axis=1)
            gy[1:, :] = np.diff(gray, axis=0)
            grad = np.sqrt(gx * gx + gy * gy)
            thr = float(np.quantile(grad, 0.2))
            smooth = grad < thr
            water_like = dark & smooth
            features = np.array([
                mean_g, std_g, p25, p50, p75,
                float(dark.mean()), float(smooth.mean()), float(water_like.mean()),
            ], dtype=np.float32)
        else:
            mean_rgb = image_array.mean(axis=(0, 1))
            std_rgb = image_array.std(axis=(0, 1))
            pil_img = Image.fromarray(np.clip(image_array * 255.0, 0, 255).astype(np.uint8), mode="RGB").convert("HSV")
            hsv = np.asarray(pil_img).astype(np.float32) / 255.0
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            blue_dom = ((blue - red) > 0.04) & ((blue - green) > 0.02)
            vegetation = (green > red + 0.06) & (green > blue + 0.05)
            water_like = blue_dom & (sat < 0.4) & (val > 0.18) & (val < 0.9) & (~vegetation)
            features = np.array([
                mean_rgb[0], mean_rgb[1], mean_rgb[2],
                std_rgb[0], std_rgb[1], std_rgb[2],
                float(blue_dom.mean()),
                float(vegetation.mean()),
                float(water_like.mean()),
                float(sat.mean()), float(val.mean()),
            ], dtype=np.float32)
        return features.reshape(1, -1)

    def _predict_with_sklearn(self, image_array: np.ndarray) -> float:
        assert self.sklearn_model is not None
        features = self._compute_features(image_array)
        model = self.sklearn_model
        try:
            # Prefer predict_proba
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)
                # Determine positive class column
                pos_idx = None
                if hasattr(model, "classes_"):
                    classes = getattr(model, "classes_")
                    try:
                        # Prefer class 1 if present
                        if 1 in classes:
                            pos_idx = int(list(classes).index(1))
                        elif "flood" in classes:
                            pos_idx = int(list(classes).index("flood"))
                    except Exception:
                        pos_idx = None
                if pos_idx is None:
                    pos_idx = proba.shape[1] - 1 if proba.ndim == 2 and proba.shape[1] > 1 else 0
                value = float(proba[0, pos_idx]) if proba.ndim == 2 else float(proba.ravel()[0])
                # Guard: if outside [0,1], squash via sigmoid
                if not np.isfinite(value) or value < 0.0 or value > 1.0:
                    value = float(1.0 / (1.0 + np.exp(-value))) if np.isfinite(value) else 0.0
                # If degenerate 0/1, blend with heuristic
                if value <= 0.0 or value >= 1.0:
                    value = self._heuristic_flood_probability(image_array)
                return float(np.clip(value, 0.0, 1.0))
            # Next try decision_function
            if hasattr(model, "decision_function"):
                score = model.decision_function(features)
                score_value = float(score.ravel()[0])
                value = float(1.0 / (1.0 + np.exp(-score_value)))
                if value <= 0.0 or value >= 1.0:
                    value = self._heuristic_flood_probability(image_array)
                return float(np.clip(value, 0.0, 1.0))
            pred = model.predict(features)
            value = float(pred.ravel()[0])
            # Map predicted label to probability proxy
            if value not in (0.0, 1.0):
                # If regression-like, normalize via sigmoid
                value = float(1.0 / (1.0 + np.exp(-value)))
            if value <= 0.0 or value >= 1.0:
                value = self._heuristic_flood_probability(image_array)
            return float(np.clip(value, 0.0, 1.0))
        except Exception as exc:
            logger.exception("Sklearn inference failed: %s", exc)
            # Fallback to heuristic if model fails
            return self._heuristic_flood_probability(image_array)

    def _heuristic_flood_probability(self, image_array: np.ndarray) -> float:
        """Heuristic tailored for RGB and grayscale imagery.

        - Grayscale: water is dark and smooth; use Otsu dark region ∩ low gradients.
        - RGB: blue dominance with HSV constraints.
        """
        red = image_array[:, :, 0]
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]
        gray = image_array.mean(axis=2)

        is_grayscale = bool(np.mean(np.abs(red - green)) < 0.005 and np.mean(np.abs(green - blue)) < 0.005)

        if is_grayscale:
            t = self._otsu_threshold(gray)
            dark = gray < max(0.15, min(0.6, t))
            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)
            gx[:, 1:] = np.diff(gray, axis=1)
            gy[1:, :] = np.diff(gray, axis=0)
            grad = np.sqrt(gx * gx + gy * gy)
            thr = float(np.quantile(grad, 0.2))
            smooth = grad < thr
            mask = dark & smooth
            return float(np.clip(mask.mean(), 0.0, 1.0))

        pil_img = Image.fromarray(np.clip(image_array * 255.0, 0, 255).astype(np.uint8), mode="RGB").convert("HSV")
        hsv = np.asarray(pil_img).astype(np.float32) / 255.0
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        blue_over_red = blue - red
        blue_over_green = blue - green
        blue_dominant = (blue_over_red > 0.04) & (blue_over_green > 0.02)
        sat_ok = sat < 0.4
        val_ok = (val > 0.18) & (val < 0.9)
        vegetation_like = (green > red + 0.06) & (green > blue + 0.05)
        high_sat_nonblue = (sat > 0.55) & (~((hue > 0.5) & (hue < 0.75)))
        mask = blue_dominant & sat_ok & val_ok & (~vegetation_like) & (~high_sat_nonblue)
        return float(np.clip(mask.mean(), 0.0, 1.0))

    def predict_probability(self, image: Image.Image) -> Dict[str, Any]:
        array = self._preprocess_image(image)
        if self.sklearn_model is not None:
            prob = self._predict_with_sklearn(array)
            source = "sklearn_model"
        elif self.model is not None and torch is not None:
            try:
                prob = self._predict_with_torch(array)
                source = "torch_model"
            except Exception as exc:
                logger.exception("Torch inference failed: %s. Falling back to heuristic.", exc)
                prob = self._heuristic_flood_probability(array)
                source = "heuristic_fallback_after_error"
        else:
            prob = self._heuristic_flood_probability(array)
            source = "heuristic_no_model"

        return {
            "flood_probability": float(np.clip(prob, 0.0, 1.0)),
            "inference_source": source,
            "model_path": self.model_path,
            "device": self.device,
        }


# Instantiate global model
MODEL: FloodModel = FloodModel()


# ---------------------------
# Helpers
# ---------------------------
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def read_image_from_request(file_storage) -> Image.Image:
    data = file_storage.read()
    if len(data) == 0:
        raise ValueError("Empty file.")
    stream = io.BytesIO(data)
    image = Image.open(stream)
    return image


# ---------------------------
# Routes
# ---------------------------
@app.route("/health", methods=["GET"])
def health() -> Tuple[str, int]:
    return "ok", 200


@app.route("/model_info", methods=["GET"])
def model_info() -> Tuple[Any, int]:
    info: Dict[str, Any] = {
        "model_path": MODEL.model_path,
        "device": MODEL.device,
        "loaded_sklearn": bool(MODEL.sklearn_model is not None),
        "loaded_torch": bool(MODEL.model is not None),
    }
    if MODEL.sklearn_model is not None:
        skl = MODEL.sklearn_model
        info.update({
            "has_predict_proba": hasattr(skl, "predict_proba"),
            "has_decision_function": hasattr(skl, "decision_function"),
            "has_predict": hasattr(skl, "predict"),
        })
        if hasattr(skl, "classes_"):
            try:
                info["classes_"] = list(getattr(skl, "classes_"))
            except Exception:
                info["classes_"] = "<unreadable>"
    return jsonify(info), 200


@app.route("/", methods=["GET"])
def index():
    return render_template("flood.html")


@app.route("/predict", methods=["POST"])
def predict() -> Tuple[Any, int]:
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image' in the request."}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename or "")
    if filename == "" or not allowed_file(filename):
        return jsonify({"error": "Invalid or missing image file. Allowed: " + ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))}), 400

    try:
        image = read_image_from_request(file)
        result = MODEL.predict_probability(image)
        response = {
            "success": True,
            "flood_probability": result["flood_probability"],
            "inference_source": result["inference_source"],
            "device": result["device"],
            "model_path": result["model_path"],
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)


