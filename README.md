# 📊 hCG Test Strip Detection and Analysis using YOLOv8 and XGBoost

This project provides a comprehensive solution for **automated pregnancy test strip analysis** using **YOLOv8** for object detection and **XGBoost** for hCG concentration prediction via image-based colorimetry.

Developed as part of an M.Tech thesis at **IIT Delhi**, the system performs both **qualitative** and **quantitative** analysis of hCG test strips captured via a smartphone or imaging system.


## 🚀 Features

- 🔍 **YOLOv8 Object Detection**: Detects Region of Interest (ROI), Control Line (C), and Test Line (T)
- 🧪 **Qualitative Analysis**: Generates intensity profiles from test strip lines
- 📈 **Quantitative Analysis**: Predicts hCG concentration in mIU/mL using trained XGBoost model
- 🧠 **AI-Powered**: Utilizes deep learning and machine learning for fast, accurate analysis
- 💻 **Streamlit GUI**: User-friendly interface for real-time image upload and testing


## 🗂️ Project Structure

![image](https://github.com/user-attachments/assets/06776ded-d697-415c-bf39-183d171ff097)

## 🔧 Installation Instructions

### 1. Clone the Repository

git clone https://github.com/Akashhverma/hcg_detection_app.git
cd hcg_detection_app
### 2. (Optional) Create Virtual Environment

python -m venv venv
source venv/bin/activate        # On macOS/Linux
venv\Scripts\activate           # On Windows

### 3. Install Dependencies
pip install -r requirements.txt
▶️ Running the App
Launch the Streamlit app:
streamlit run test.py
Visit http://localhost:8501 in your browser to use the GUI.

## 🧠 Model Summary

### YOLOv8 (Ultralytics)
Detects:
ROI (Test strip area)
Control Line (C)
Test Line (T)

### XGBoost Regressor
Input: Extracted color space features (RGB, HSV, LAB) from cropped C and T lines
Output: hCG concentration in mIU/mL

Trained on labeled dataset using manual annotations and lab-calibrated values

### 📊 Modes of Operation
#### ✅ Qualitative Analysis
Extracts and plots intensity values across C and T lines.
Used for visual validation and line strength analysis.

#### 📈 Quantitative Analysis
Predicts actual hCG concentration using a trained machine learning model.
Interprets result:
Positive if concentration > threshold
Negative if below threshold
Invalid if C-line is missing

#### 🖼️ Sample Output
![Screenshot 2025-03-05 152532](https://github.com/user-attachments/assets/8974cf53-5a0e-4692-9110-41a4fab85513)
![Screenshot 2025-04-16 170102](https://github.com/user-attachments/assets/93f3f498-5757-476e-99b2-d86527640620)
![Screenshot 2025-03-05 152551](https://github.com/user-attachments/assets/3100bad7-06c1-45e9-9305-edae326ccfdc)

### Streamlit link for app:
https://hcgdetectionapp-jjss2yoggzzurpwnbuoh9w.streamlit.app/



🧑‍💻 Author
## Akash Verma

### M.Tech, Instrument Technology (SeNSE), IIT Delhi

🔗 https://www.linkedin.com/in/akash-verma-525a88145/
💻 https://github.com/Akashhverma

📄 License
This project is licensed under the MIT License.
