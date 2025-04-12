# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# from PIL import Image
# from qualitative_analysis import qualitative_analysis
# from quantitative_analysis import quantitative_analysis

# # ====== Load YOLO Model ======
# MODEL_PATH = r"D:\MUSIC N JOY\zIIT DELHI\Mtech Project\GUI\Trained_model\weights\best.pt"
# try:
#     model = YOLO(MODEL_PATH)
# except Exception as e:
#     st.error(f"❌ Error Loading Model: {e}")
#     st.stop()

# # ====== Streamlit UI ======
# st.title("🔬 hCG Test Detection")
# st.write("📤 Upload an image here.")
# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# # ====== Resize Helper Function ======
# def resize_image(image, base_width=200):
#     w_percent = base_width / float(image.size[0])
#     new_height = int(float(image.size[1]) * w_percent)
#     return image.resize((base_width, new_height), Image.LANCZOS)

# # ====== Helper: Get Latest Prediction Folder ======
# def get_latest_prediction_folder():
#     detect_path = "runs/detect"
#     subdirs = [os.path.join(detect_path, d) for d in os.listdir(detect_path) if os.path.isdir(os.path.join(detect_path, d))]
#     if subdirs:
#         return max(subdirs, key=os.path.getmtime)
#     return None

# # ====== Main Logic ======
# if uploaded_file:
#     # Convert Uploaded File to OpenCV Format
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Convert BGR to RGB for correct color display
#     image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#     uploaded_pil = Image.fromarray(image_rgb)

#     # ====== Run YOLO Detection ======
#     results = model.predict(image_np, save=True, save_crop=True)
#     detected = any(len(result.boxes) > 0 for result in results)

#     if detected:
#         img_with_boxes = results[0].plot()
#         img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
#         detected_pil = Image.fromarray(img_with_boxes_rgb)

#         detected_save_path = "runs/detect/latest_prediction.jpg"
#         detected_pil.save(detected_save_path)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(resize_image(uploaded_pil), caption="📷 Uploaded Image", use_container_width=False)
#         with col2:
#             st.image(resize_image(detected_pil), caption="✅ Detection Result", use_container_width=False)

#         # ====== Locate Latest Cropped C & T Line Paths ======
#         latest_pred_folder = get_latest_prediction_folder()
#         if latest_pred_folder:
#             roi_path = os.path.join(latest_pred_folder, "crops", "ROI", "image0.jpg")
#             c_line_path = os.path.join(latest_pred_folder, "crops", "C", "image0.jpg")
#             t_line_path = os.path.join(latest_pred_folder, "crops", "T", "image0.jpg")
#         else:
#             roi_path = c_line_path = t_line_path = None

#         # ====== Display ROI, C-Line, and T-Line ======
#         st.subheader("🔍 Cropped Regions (ROI, C-Line, T-Line)")
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             if roi_path and os.path.exists(roi_path):
#                 roi_img = Image.open(roi_path).convert("RGB")
#                 st.image(resize_image(roi_img), caption="📌 ROI")
#         with col2:
#             if c_line_path and os.path.exists(c_line_path):
#                 c_img = Image.open(c_line_path).convert("RGB")
#                 st.image(resize_image(c_img), caption="📌 Control Line (C)")
#         with col3:
#             if t_line_path and os.path.exists(t_line_path):
#                 t_img = Image.open(t_line_path).convert("RGB")
#                 st.image(resize_image(t_img), caption="📌 Test Line (T)")

#         # ====== Analysis Buttons ======
#         st.subheader("🧪 Choose Analysis Type")
#         col1, col2 = st.columns(2)

#         with col1:
#             if st.button("📊 Qualitative Analysis"):
#                 try:
#                     result_img, intensity_plot = qualitative_analysis(roi_path)
#                     st.image(resize_image(result_img), caption="🔍 Detected Lines on Strip")
#                     st.pyplot(intensity_plot)
#                 except Exception as e:
#                     st.error(f"⚠️ Error: {e}")

#         with col2:
#             if st.button("📈 Quantitative Analysis"):
#                 try:
#                     if c_line_path and t_line_path:
#                         predicted_conc = quantitative_analysis(c_line_path, t_line_path)
#                         st.success(f"🧪 Predicted hCG Concentration: **{predicted_conc} mIU/mL**")
#                     else:
#                         st.error("❌ C or T line image not found. Please upload a valid strip image.")
#                 except Exception as e:
#                     st.error(f"⚠️ Error in Quantitative Analysis: {e}")

#         # ====== Determine Result Based on C & T Visibility ======
#         st.subheader("🔎 Test Result")
#         if c_line_path and os.path.exists(c_line_path):
#             if t_line_path and os.path.exists(t_line_path):
#                 st.success("✅ **Positive Result** - hCG detected.")
#             else:
#                 st.warning("❌ **Negative Result** - No hCG detected.")
#         else:
#             st.error("⚠️ **Invalid Test** - Control Line (C) not detected. Please retry!")

#     else:
#         st.warning("⚠️ Invalid strip image! Please try reuploading a valid test strip.")




import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
from qualitative_analysis import qualitative_analysis
from quantitative_analysis import quantitative_analysis

# ====== Load YOLO Model ======
MODEL_PATH = "Trained_model/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")
    st.stop()

# ====== Streamlit UI ======
st.title("🔬 hCG Test Detection")
st.write("📤 Upload an image here.")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ====== Resize Helper Function ======
def resize_image(image, base_width=200):
    w_percent = base_width / float(image.size[0])
    new_height = int(float(image.size[1]) * w_percent)
    return image.resize((base_width, new_height), Image.LANCZOS)

# ====== Helper: Get Latest Prediction Folder ======
def get_latest_prediction_folder():
    detect_path = "runs/detect"
    subdirs = [os.path.join(detect_path, d) for d in os.listdir(detect_path) if os.path.isdir(os.path.join(detect_path, d))]
    if subdirs:
        return max(subdirs, key=os.path.getmtime)
    return None

# ====== Main Logic ======
if uploaded_file:
    # Convert Uploaded File to OpenCV Format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    uploaded_pil = Image.fromarray(image_rgb)

    # ====== Run YOLO Detection ======
    results = model.predict(image_np, save=True, save_crop=True)
    detected = any(len(result.boxes) > 0 for result in results)

    if detected:
        img_with_boxes = results[0].plot()
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        detected_pil = Image.fromarray(img_with_boxes_rgb)

        detected_save_path = "runs/detect/latest_prediction.jpg"
        detected_pil.save(detected_save_path)

        col1, col2 = st.columns(2)
        with col1:
            st.image(resize_image(uploaded_pil), caption="📷 Uploaded Image", use_container_width=False)
        with col2:
            st.image(resize_image(detected_pil), caption="✅ Detection Result", use_container_width=False)

        # ====== Locate Latest Cropped C & T Line Paths ======
        latest_pred_folder = get_latest_prediction_folder()
        if latest_pred_folder:
            roi_path = os.path.join(latest_pred_folder, "crops", "ROI", "image0.jpg")
            c_line_path = os.path.join(latest_pred_folder, "crops", "C", "image0.jpg")
            t_line_path = os.path.join(latest_pred_folder, "crops", "T", "image0.jpg")
        else:
            roi_path = c_line_path = t_line_path = None

        # ====== Display ROI, C-Line, and T-Line ======
        st.subheader("🔍 Cropped Regions (ROI, C-Line, T-Line)")
        col1, col2, col3 = st.columns(3)

        with col1:
            if roi_path and os.path.exists(roi_path):
                roi_img = Image.open(roi_path).convert("RGB")
                st.image(resize_image(roi_img), caption="📌 ROI")
        with col2:
            if c_line_path and os.path.exists(c_line_path):
                c_img = Image.open(c_line_path).convert("RGB")
                st.image(resize_image(c_img), caption="📌 Control Line (C)")
        with col3:
            if t_line_path and os.path.exists(t_line_path):
                t_img = Image.open(t_line_path).convert("RGB")
                st.image(resize_image(t_img), caption="📌 Test Line (T)")

        # ====== Analysis Buttons ======
        st.subheader("🧪 Choose Analysis Type")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Qualitative Analysis"):
                try:
                    result_img, intensity_plot = qualitative_analysis(roi_path)
                    st.image(resize_image(result_img), caption="🔍 Detected Lines on Strip")
                    st.pyplot(intensity_plot)

                    # ====== Show Test Result after Qualitative Analysis ======
                    st.subheader("🔎 Test Result")
                    if c_line_path and os.path.exists(c_line_path):
                        if t_line_path and os.path.exists(t_line_path):
                            st.success("✅ **Positive Result** - hCG detected.")
                        else:
                            st.warning("❌ **Negative Result** - No hCG detected.")
                    else:
                        st.error("⚠️ **Invalid Test** - Control Line (C) not detected. Please retry!")

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

        with col2:
            if st.button("📈 Quantitative Analysis"):
                try:
                    if c_line_path and t_line_path:
                        predicted_conc = quantitative_analysis(c_line_path, t_line_path)
                        st.success(f"🧪 Predicted hCG Concentration: **{predicted_conc} mIU/mL**")
                    else:
                        st.error("❌ C or T line image not found. Please upload a valid strip image.")
                except Exception as e:
                    st.error(f"⚠️ Error in Quantitative Analysis: {e}")

    else:
        st.warning("⚠️ Invalid strip image! Please try reuploading a valid test strip.")

