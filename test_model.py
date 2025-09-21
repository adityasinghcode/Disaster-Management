import os
import argparse
import json
from typing import Optional

import numpy as np
from PIL import Image


from app import FloodModel


def run_single_check(model_path: str, image_path: Optional[str]) -> None:
    model = FloodModel(model_path=model_path)

    info = {
        "model_path": model.model_path,
        "device": model.device,
        "loaded_sklearn": bool(model.sklearn_model is not None),
        "loaded_torch": bool(model.model is not None),
    }

    if model.sklearn_model is not None:
        skl = model.sklearn_model
        info["sklearn_has_predict_proba"] = hasattr(skl, "predict_proba")
        info["sklearn_has_decision_function"] = hasattr(skl, "decision_function")
        info["sklearn_has_predict"] = hasattr(skl, "predict")
        if hasattr(skl, "classes_"):
            try:
                info["classes_"] = list(getattr(skl, "classes_"))
            except Exception:
                info["classes_"] = "<unreadable>"

    print("Model load info:\n" + json.dumps(info, indent=2))

    # Choose input image
    image: Image.Image
    if image_path and os.path.isfile(image_path):
        image = Image.open(image_path)
        print(f"Using provided image: {image_path}")
    else:
        
        print("No valid image provided. Generating synthetic samples...")

        h, w = 384, 384
        left = np.clip(np.random.normal(loc=0.18, scale=0.02, size=(h, w // 2)), 0, 1)
        right = np.clip(np.random.normal(loc=0.65, scale=0.20, size=(h, w - w // 2)), 0, 1)
        gray = np.concatenate([left, right], axis=1).astype(np.float32)
        image = Image.fromarray(np.uint8(gray * 255), mode="L").convert("RGB")


    result = model.predict_probability(image)
    print("\nPrediction result:")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test resnet_metrics.pkl model pipeline")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "models", "resnet_metrics.pkl")), help="Path to .pkl model file")
    parser.add_argument("--image", default=None, help="Optional path to an image to test")
    args = parser.parse_args()

    run_single_check(args.model, args.image)


if __name__ == "__main__":
    main()


