# ai_adapter.py
import numpy as np
import importlib
import traceback

# Try to import user's inference code (assumes inference.py exposes infer(image) or Classifier)
try:
    inference = importlib.import_module('inference')  # your inference.py
    # Prefer an `infer(image)` function if present
    if hasattr(inference, 'infer'):
        def predict(frame: np.ndarray):
            # user should return (label_str, confidence_float)
            return inference.infer(frame)
    elif hasattr(inference, 'Classifier'):  # or a Classifier class with predict()
        clf = inference.Classifier()
        def predict(frame: np.ndarray):
            res = clf.predict(frame)
            return res  # assume (label, confidence)
    else:
        raise ImportError("No compatible infer() or Classifier found in inference.py")
except Exception:
    # Fallback dummy predictor
    import random
    def predict(frame):
        labels = ['flood', 'fire', 'landslide', 'normal']
        label = random.choice(labels)
        confidence = round(random.uniform(0.6, 0.98), 2)
        return label, confidence

    # Helpful debug print to terminal
    print("ai_adapter: Could not import inference.py — using dummy predictor.")
    traceback.print_exc()
