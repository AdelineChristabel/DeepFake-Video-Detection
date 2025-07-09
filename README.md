#DeepFake Video Detection using EfficientNetB2 + Transformer

This project implements a DeepFake video classification pipeline using EfficientNetB2 for spatial (frame-level) feature extraction and a Transformer encoder for modeling temporal relationships across frames. It addresses class imbalance using upsampling and focal loss, and includes tools for training, evaluation, and inference.



#📁 Dataset Structure
split_dataset/
├── train/
│ ├── real/
│ └── fake/
└── test/
  ├── real/
  └── fake/

Each subfolder must contain '.mp4' or video files for processing.

# Features:
- Extracts 20 evenly spaced frames per video
- Uses EfficientNetB2 (ImageNet-pretrained) as feature extractor
- Temporal modeling via Transformer encoder blocks
- Handles class imbalance via upsampling and class weighting
- Focal loss to address imbalance in training
- Confusion matrix and accuracy/loss visualizations
- Inference script to test individual videos


# Requirements:

Install dependencies with:
                          pip install -r requirements.txt

* Key Libraries:
TensorFlow 2.x
OpenCV
NumPy
scikit-learn
matplotlib
tqdm

 ** Model Architecture **
EfficientNetB2: Feature extractor (frame-wise)
Transformer Encoder: Captures temporal relations across frames
GlobalAveragePooling + Dense: Final classifier
The model is trained using binary focal loss and optimized with Adam.

Training
Make sure your dataset path is set correctly in the script or notebook. Then:
                                           python file_name.py
                                           
This will:
- Extract features from frames
- Balance dataset with upsampling
- Train the model

Save it as deepfake_efficientnetb2_transformer_balanced_focal2.keras

📈 Evaluation
After training, the script displays:
✅ Confusion Matrix (Real vs Fake)
📊 Accuracy and Loss graphs per epoch
📝 Optional: classification_report for precision, recall, F1-score

🔍 Inference
To test a single video:
                      python file_name.py path/to/video.mp4
                      

Plots for training accuracy, validation accuracy, and loss
Confusion matrix on test set






                          


