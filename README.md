# Skin Disease Prediction

This project leverages deep learning to classify skin diseases from images using a Convolutional Neural Network (CNN) built upon the EfficientNetB0 architecture. The model is integrated into an interactive web application using Streamlit, allowing users to upload skin images and receive real-time predictions. ([varshitha-g/Skin-Disease-Prediction - GitHub](https://github.com/varshitha-g/Skin-Disease-Prediction?utm_source=chatgpt.com))

## 🔍 Overview

Skin diseases are prevalent worldwide, and early detection is crucial for effective treatment. This project aims to assist in the preliminary diagnosis of skin conditions by providing a tool that can classify images into specific disease categories, thereby supporting healthcare professionals and raising awareness among the general public.

## 🧠 Model Architecture

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Dense Layer (128 units) with ReLU activation
  - Batch Normalization
  - Dropout (0.5)
  - Dense Layer (32 units) with ReLU activation
  - Batch Normalization
  - Dropout (0.5)
  - Output Layer (9 units) with Softmax activation

The model is compiled using the Adam optimizer and trained with the sparse categorical cross-entropy loss function.

## 🖼️ Dataset

- **Source**: [Specify the dataset source, e.g., ISIC Archive, DermNet, or a custom dataset]
- **Classes**: 9 distinct skin disease categories
- **Preprocessing**:
  - Resizing images to 256x256 pixels
  - Normalization of pixel values
  - Data augmentation techniques applied to enhance model robustness ([GitHub - FridahKimathi/Skin-Disease-Image-Classifier-for-Accurate ...](https://github.com/FridahKimathi/Skin-Disease-Image-Classifier-for-Accurate-and-Accessible-Diagnosis?utm_source=chatgpt.com), [varshitha-g/Skin-Disease-Prediction - GitHub](https://github.com/varshitha-g/Skin-Disease-Prediction?utm_source=chatgpt.com))

## 🚀 Deployment

The application is deployed using Streamlit, providing a user-friendly interface for image upload and prediction display.

## 📁 Project Structure

```
skin_disease_prediction/
├── model/
│   ├── model_weights.h5
├── app.py
├── model_training.ipynb
├── requirements.txt
├── training_log.csv
├── README.md
```

- `model/`: Contains the trained model weights.
- `app.py`: Streamlit application script.
- `model_training.ipynb`: Jupyter Notebook for model training and evaluation.
- `requirements.txt`: List of dependencies.
- `training_log.csv`: Logs of training metrics per epoch. ([Skin Disease Prediction using CNN Model - GitHub](https://github.com/Karthikeyan-S-04/Derm_AI?utm_source=chatgpt.com), [Bhuvan588/Skin-Disease-Prediction - GitHub](https://github.com/Bhuvan588/Skin-Disease-Prediction?utm_source=chatgpt.com), [Deeply Supervised Skin Lesions Diagnosis with Stage and Branch Attention](https://arxiv.org/abs/2205.04326?utm_source=chatgpt.com), [[PDF] Post-process correction improves the accuracy of satellite PM2.5 ...](https://amt.copernicus.org/articles/17/5747/2024/amt-17-5747-2024.pdf?utm_source=chatgpt.com))

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohitkumhar/skin_disease_prediction.git
   cd skin_disease_prediction
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

5. **Interact with the app**:
   - Upload a skin image in JPG, JPEG, or PNG format.
   - View the predicted skin disease category.

## 📊 Results

- **Training Accuracy**: [Insert training accuracy, e.g., 80%]
- **Validation Accuracy**: [Insert validation accuracy, e.g., 76%]
- **Test Accuracy**: [Insert test accuracy, e.g., 70%]

*Note: These metrics are placeholders. Please update them with actual results from your training logs.*

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - TensorFlow & Keras: For building and training the CNN model
  - Streamlit: For deploying the web application
  - NumPy & Pandas: For data manipulation
  - Matplotlib & Seaborn: For data visualization ([MahimaKhatri/Skin-Disease-Detection: ML and signal ... - GitHub](https://github.com/MahimaKhatri/Skin-Disease-Detection?utm_source=chatgpt.com), [varshitha-g/Skin-Disease-Prediction - GitHub](https://github.com/varshitha-g/Skin-Disease-Prediction?utm_source=chatgpt.com), [DermAI-Skin-Disease-Diagnosis/README.md at main - GitHub](https://github.com/sakshiselmokar/DermAI-Skin-Disease-Diagnosis/blob/main/README.md?utm_source=chatgpt.com))

## 📌 Future Enhancements

- Incorporate more diverse and larger datasets to improve model generalization.
- Implement real-time data augmentation during training.
- Enhance the user interface for better user experience.
- Deploy the application on cloud platforms for broader accessibility. ([varshitha-g/Skin-Disease-Prediction - GitHub](https://github.com/varshitha-g/Skin-Disease-Prediction?utm_source=chatgpt.com))

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to contribute to the project, please fork the repository and submit a pull request.


---

*Disclaimer: This tool is intended for educational and research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
