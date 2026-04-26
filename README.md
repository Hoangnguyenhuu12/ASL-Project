# American Sign Language Alphabet Recognition

A real-time ASL alphabet recognition system using MediaPipe hand landmark extraction and a Random Forest classifier. Built for the DAP391m course at HCM FPT University.

## Results
- **Random Forest:** 99.82% test accuracy, 99.78% OOB score, 5-fold CV: 99.68%
- **XGBoost:** 99.59% test accuracy (comparison baseline)
- Recognizes all 26 static ASL alphabet letters (A–Z) in real-time via webcam

## Documentation

📄 **[ASL_Report.pdf](./ASL_Report.pdf)** — Full technical report covering methodology, feature engineering, SMOTE balancing, and model evaluation.

🎯 **[ASL_Presentation.pdf](./ASL_Presentation.pdf)** — Presentation slides covering problem statement, system architecture, and results.

## Project Structure
```
├── code/
│   ├── ASL_final_code.py             # Main real-time ASL recognition application
│   ├── cutvideo.py                   # Step 1: Extract frames from recorded videos
│   ├── croppicture.py                # Step 2: Crop & normalize hand images (256x256)
│   ├── label.py                      # Step 3: Extract landmarks & label data (Google Colab)
│   └── train.ipynb                   # Step 4: Train Random Forest and XGBoost models
├── models/
│   ├── mo_hinh_randomforest_cu_chi.pkl   # Trained Random Forest model
│   └── bo_giai_ma_nhan_randomforest.pkl  # Label encoder for the RF model
├── data/
│   └── hand_landmarks_final.csv      # Processed dataset (21 landmarks × 3D coords per sample)
├── docs/
│   ├── ASL_Report.pdf
│   └── ASL_Presentation.pdf
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Download the MediaPipe Hand Landmarker model
```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
Place `hand_landmarker.task` in the root project folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the real-time recognizer
```bash
python ASL_final_code.py
```
Press `q` to quit.

## Data Pipeline (for retraining)

The raw video dataset is hosted on Google Drive: (https://drive.google.com/drive/folders/1MdvkPtljR4XTG63Gc2YMiJOJRlQlHQMq)

If you want to rebuild the dataset from scratch, follow these steps in order:

**Step 1** — Extract frames from recorded `.mov` videos:
```bash
# Edit the folder paths at the top of cutvideo.py first
python cutvideo.py
```

**Step 2** — Crop and normalize hand images to 256×256:
```bash
# Edit MODEL_PATH, INPUT_DIR, OUTPUT_DIR at the top of croppicture.py first
python croppicture.py
```

**Step 3** — Extract hand landmarks and generate the CSV dataset (run in Google Colab):
```python
# Upload your normalized images as a ZIP to Google Drive
# Edit the paths at the bottom of label.py, then run in Colab
```

**Step 4** — Train the model:
```bash
jupyter notebook train.ipynb
```

## System Architecture
```
Webcam → MediaPipe (Hand Detection) → Crop to 256×256
→ MediaPipe (Re-detect on crop) → Normalize Landmarks
→ Random Forest → Predicted Letter + Confidence %
```

## Key Features
- Dual-detection pipeline: detects hand on full frame, crops, then re-detects for precision
- Landmark normalization: wrist-centered + palm-size scaled + rotation-aligned
- Adaptive Gaussian Thresholding for robustness under varying lighting
- Confidence display: green (>50%), orange (<50%), "?" (<15%)

## Technologies
- **Python** — Core language
- **OpenCV (cv2)** — Video capture and image processing
- **MediaPipe** — 21-point 3D hand landmark extraction
- **scikit-learn** — Random Forest classifier
- **XGBoost** — Comparison model
- **imbalanced-learn** — SMOTE for class balancing
- **Pillow & NumPy** — Image manipulation

## Authors
**Group 6** — AI1904, HCM FPT University
- Nguyen Huu Hoang (leader)
- Hoang Phuc Binh
- Tran Khoi Nguyen
- Mai Tran Hao
