import cv2
import numpy as np
import xgboost as xgb
import os

def get_lab_stats(image_path):
    """Extracts LAB mean and std for each channel from the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    img = cv2.resize(img, (50, 50))  # Optional resize for consistency
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    return {
        "L_mean": np.mean(l), "A_mean": np.mean(a), "B_mean": np.mean(b),
        "L_std": np.std(l), "A_std": np.std(a), "B_std": np.std(b)
    }

def extract_features(c_path, t_path):
    """Computes final 15-dim feature vector based on LAB color stats."""
    c_feat = get_lab_stats(c_path)
    t_feat = get_lab_stats(t_path)

    # Ratios
    L_ratio = t_feat["L_mean"] / (c_feat["L_mean"] + 1e-6)
    A_ratio = t_feat["A_mean"] / (c_feat["A_mean"] + 1e-6)
    B_ratio = t_feat["B_mean"] / (c_feat["B_mean"] + 1e-6)

    # Differences
    L_diff = t_feat["L_mean"] - c_feat["L_mean"]
    A_diff = t_feat["A_mean"] - c_feat["A_mean"]
    B_diff = t_feat["B_mean"] - c_feat["B_mean"]

    # Std Ratios
    L_std_ratio = t_feat["L_std"] / (c_feat["L_std"] + 1e-6)
    A_std_ratio = t_feat["A_std"] / (c_feat["A_std"] + 1e-6)
    B_std_ratio = t_feat["B_std"] / (c_feat["B_std"] + 1e-6)

    # Include raw stds
    feature_vector = [
        L_ratio, A_ratio, B_ratio,
        L_diff, A_diff, B_diff,
        L_std_ratio, A_std_ratio, B_std_ratio,
        c_feat["L_std"], c_feat["A_std"], c_feat["B_std"],
        t_feat["L_std"], t_feat["A_std"], t_feat["B_std"]
    ]
    
    return np.array(feature_vector)

def quantitative_analysis(c_path, t_path, model_path="xgb_model.json"):
    """Predicts hCG concentration from C and T line image paths."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Load trained model
    booster = xgb.Booster()
    booster.load_model(model_path)

    # Extract features
    features = extract_features(c_path, t_path)
    dmatrix = xgb.DMatrix(features.reshape(1, -1))

    # Make prediction
    prediction = booster.predict(dmatrix)[0]
    return round(prediction, 2)

# 🔍 Example usage
# pred = quantitative_analysis("path_to_c_image.jpg", "path_to_t_image.jpg")
# print(f"Predicted hCG Concentration: {pred} mIU/mL")


