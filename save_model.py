import joblib
import os
import subprocess
from transform import preprocessor, model

# Assume these are loaded from your previous session or re-trained
# preprocessor = ...
# model = ...
# optimal_threshold = ... (from your PR curve analysis, e.g., 0.45)

ARTIFACT_PATH = 'artifacts/pipeline_v1.joblib'
os.makedirs('artifacts', exist_ok=True)

# Bundle everything
artifact = {
    'preprocessor': preprocessor,
    'model': model,
    #'threshold': optimal_threshold, # Use the value from your PR curve analysis
    'feature_names': preprocessor.get_feature_names_out()
}

joblib.dump(artifact, ARTIFACT_PATH)
print(f"✅ Artifact saved to {ARTIFACT_PATH}")