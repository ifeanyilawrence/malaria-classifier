# Malaria Cell Classifier

A deep learning project that uses Convolutional Neural Networks (CNN) to classify blood cell images as either parasitized (infected with malaria) or uninfected.

## 🩸 Overview

This project implements a binary classification model to detect malaria parasites in blood cell images using TensorFlow/Keras. The model can assist in automated malaria diagnosis by analyzing microscopic images of blood cells.

## 🎯 Features

- **Custom CNN Architecture**: 3-layer convolutional neural network with batch normalization and L2 regularization
- **Data Augmentation**: Advanced image preprocessing with rotation, shifting, shearing, and flipping
- **Real-time Prediction**: Single image classification with confidence scores
- **Comprehensive Evaluation**: Confusion matrix, classification reports, and training visualizations
- **Model Persistence**: Automatic saving of best performing models

## 📊 Dataset

The model is trained on microscopic blood cell images categorized into:
- **Parasitized**: Cells infected with malaria parasites
- **Uninfected**: Healthy blood cells

Images are preprocessed to 130x130x3 pixels for optimal model performance.

## 🏗️ Model Architecture

```
Sequential CNN Model:
├── Conv2D (32 filters, 3x3) + BatchNorm + MaxPool2D
├── Conv2D (64 filters, 3x3) + BatchNorm + MaxPool2D  
├── Conv2D (64 filters, 3x3) + BatchNorm + MaxPool2D
├── Flatten
├── Dense (128 units) + Dropout (0.5)
└── Dense (1 unit, sigmoid) - Binary Classification
```

**Key Features:**
- L2 regularization to prevent overfitting
- Batch normalization for training stability
- Dropout for improved generalization
- Adam optimizer with learning rate scheduling

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib seaborn pillow scikit-learn pathlib
```

### Running the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ifeanyilawrence/malaria-classifier.git
   cd malaria-classifier
   ```

2. **Prepare your dataset:**
   - Place cell images in `cell_images/Parasitized/` and `cell_images/Uninfected/` directories

3. **Train the model:**
   - Open and run `main.ipynb` in Jupyter Notebook
   - The notebook includes data loading, preprocessing, training, and evaluation

4. **Make predictions:**
   ```python
   predict_image('path_to_your_image.png')
   ```

## 📈 Model Performance

The model includes comprehensive evaluation metrics:
- **Training/Validation Accuracy & Loss Curves**
- **Confusion Matrix Visualization**
- **Classification Report** (Precision, Recall, F1-Score)
- **Test Set Evaluation**

Training features:
- Early stopping to prevent overfitting
- Model checkpointing to save best weights
- Learning rate reduction on plateau

## 📁 Project Structure

```
malaria-classifier/
├── main.ipynb                          # Main training notebook
├── cell_images/                        # Dataset directory
│   ├── Parasitized/                   # Infected cell images
│   └── Uninfected/                    # Healthy cell images
├── test_for_parasitized_cell.png      # Sample infected cell
├── test_for_uninfected_cell.png       # Sample healthy cell
├── .gitignore                         # Git ignore rules
└── README.md                          # Project documentation
```

## 🔧 Training Configuration

- **Image Size**: 130x130x3
- **Batch Size**: 32
- **Train/Validation Split**: 70/30
- **Data Augmentation**: Rotation, shifting, shearing, zoom, horizontal flip
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: Binary crossentropy
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## 🧪 Usage Example

```python
# Load and preprocess image
img = load_img('cell_image.png', target_size=(130, 130))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)[0][0]
predicted_class = 'Parasitized' if prediction > 0.5 else 'Uninfected'
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {predicted_class} (Confidence: {confidence:.4f})")
```

## 📚 Dependencies

- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Pillow (PIL)
- scikit-learn
- pathlib

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ifeanyi Lawrence**
- GitHub: [@ifeanyilawrence](https://github.com/ifeanyilawrence)

## 🙏 Acknowledgments

- Dataset source and medical imaging community
- TensorFlow/Keras documentation and tutorials
- Research papers on malaria detection using deep learning

---

⚠️ **Disclaimer**: This model is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.
