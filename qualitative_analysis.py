import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image

def qualitative_analysis(image_path):
    """Performs qualitative analysis by detecting lines on hCG test strips."""
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    # Find the dominant angle of the strip
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Convert radians to degrees
            angles.append(angle)

    median_angle = np.median(angles) if angles else 0

    # Rotate the image to align the strip vertically
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Convert to LAB and extract Lightness (L) channel
    lab_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2LAB)
    l_channel = lab_image[:, :, 0]

    # Determine strip orientation
    if h > w:
        intensity_profile = np.mean(l_channel, axis=1)  # Vertical Strip
        axis_label = "Vertical Position"
    else:
        intensity_profile = np.mean(l_channel, axis=0)  # Horizontal Strip
        axis_label = "Horizontal Position"

    # Compute adaptive prominence for peak detection
    max_intensity_drop = np.max(intensity_profile) - np.min(intensity_profile)
    adaptive_prominence = 0.2 * max_intensity_drop  # Adjust factor as needed

    # Detect dips (dark lines)
    peaks, _ = find_peaks(-intensity_profile, distance=20, prominence=adaptive_prominence)

    # Draw detected lines on the image
    for peak in peaks:
        if h > w:
            cv2.line(aligned_image, (0, peak), (w, peak), (0, 0, 255), 2)  # Vertical image
        else:
            cv2.line(aligned_image, (peak, 0), (peak, h), (0, 0, 255), 2)  # Horizontal image

    # Convert OpenCV image to PIL for Streamlit display
    result_pil = Image.fromarray(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))

    # Generate Intensity Profile Plot
    plt.figure(figsize=(6, 4))
    plt.plot(intensity_profile, label="LAB Lightness Intensity", color='blue')
    plt.scatter(peaks, intensity_profile[peaks], color='red', label="Detected Lines")
    plt.xlabel(axis_label)
    plt.ylabel("Intensity")
    plt.title("LAB L Channel Intensity Profile with Detected Lines")
    plt.legend()

    return result_pil, plt
