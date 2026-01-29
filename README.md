# Traffic Sign Recognition (TSR) using PyTorch

An end-to-end Deep Learning project that classifies 43 different types of traffic signs using a custom **Residual Convolutional Neural Network (ResNet)**. The project includes a training pipeline with evaluation metrics and a real-time prediction script for static images and webcam feeds.

## 🚀 Features

* **Custom ResNet Architecture**: Uses skip-connections to mitigate vanishing gradients and achieve high accuracy.
* **Data Augmentation**: Robust preprocessing and normalization tailored to the GTSRB dataset.
* **Real-time Inference**: OpenCV-integrated webcam support with a **Stability Voting Filter** for smooth predictions.
* **Comprehensive Evaluation**: Generates Loss/Accuracy curves, Confusion Matrices, and Classification Reports.

## 📁 Project Structure

```text
Project_Root/
├── sign_recognition/
│   ├── Train/         # Folder containing 43 subfolders (0-42)
│   ├── Test/          # Folder containing test images
│   ├── Train.csv      # Training metadata
│   ├── Test.csv       # Test metadata
│   └── Meta.csv       # Sign property metadata
├── models/
│   └── traffic_sign_model.pth  # Saved weights after training
├── train.py           # Training & Evaluation script
├── predict.py         # Real-time and static prediction script
└── README.md          # Project documentation

```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd sign_recognition

```


2. **Install Dependencies**:
```bash
pip install torch torchvision pandas pillow scikit-learn matplotlib seaborn opencv-python

```



## 📈 Model Architecture

The model is based on a **Residual Neural Network**. It consists of an initial convolutional layer followed by three stages of Residual Blocks that increase feature depth from 64 to 256.

* **Activation**: ReLU
* **Pooling**: Adaptive Average Pooling
* **Loss Function**: Cross-Entropy Loss
* **Optimizer**: Adam ()

## 🚦 Usage

### 1. Training and Evaluation

Run the training script to train the model for 15 epochs and generate performance graphs:

```bash
python train.py

```

After training, the script will automatically display:

* **Training History**: Loss and Accuracy plots.
* **Confusion Matrix**: Visual representation of class-wise errors.

### 2. Prediction (Inference)

Use the prediction script to test the model:

```bash
python predict.py

```

Choose between:

* **Mode 1**: Enter a path to an image (e.g., `dataset/Test/00001.png`).
* **Mode 2**: Open your webcam for real-time detection.

## 📊 Performance

The model typically achieves:

* **Training Accuracy**: >99%
* **Test Accuracy**: ~97-98% (depending on data quality)

## 📝 License

This project is licensed under the MIT License.

---

### How to use this file:

1. Open a text editor (like Notepad or VS Code).
2. Paste the content above.
3. Save the file as `README.md` in your project folder.
