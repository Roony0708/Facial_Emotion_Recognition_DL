# 🎭 Facial Emotion Recognition using Deep Learning

## 🚀 Project Overview  
This project utilizes **Deep Learning** to classify human facial expressions into various emotions. A **Convolutional Neural Network (CNN)** is trained on facial images and deployed for **real-time emotion detection** using a webcam. It has practical applications in **healthcare, security, AI-driven customer support, and human-computer interaction.**

---

## 📂 Dataset
The model is trained on a publicly available facial emotion dataset, with preprocessing and augmentation techniques to enhance accuracy.

### 🔹 **Emotions Detected:**
✅ **Happy** 😊  
✅ **Sad** 😔  
✅ **Angry** 😡  
✅ **Neutral** 😐  
✅ **Surprised** 😲  
✅ **Fearful** 😨  

### 🔹 **Key Features:**
✔️ **Deep Learning-based Emotion Recognition** using **CNN & Transfer Learning** (VGG16, ResNet50V2)  
✔️ **Real-time Emotion Detection App** using **OpenCV** & **Flask**  
✔️ **Preprocessing & Augmentation** for improved model accuracy  
✔️ **Performance Evaluation** using Accuracy, Precision, Recall & Confusion Matrix  

---

## ⚙️ Project Workflow
### 1️⃣ Importing Required Libraries
📌 **TensorFlow & Keras** – For deep learning model training.  
📌 **OpenCV** – Real-time face detection.  
📌 **Matplotlib & Seaborn** – Data visualization.  
📌 **Sklearn** – Data preprocessing, evaluation, and dataset splitting.

### 2️⃣ Loading and Preprocessing the Dataset
📌 Convert images to **grayscale** for reduced complexity.  
📌 Resize images to **48x48 pixels** for uniformity.  
📌 Apply **data augmentation** (rotation, flipping, zooming) to enhance model generalization.

### 3️⃣ Splitting Data into Training & Testing Sets
📌 **80% Training, 20% Testing** split.  
📌 **One-hot encoding** applied to labels for multi-class classification.

### 4️⃣ Building the Deep Learning Model
🛠️ **Convolutional Layers** – Extract facial features (eyes, mouth, expressions).  
🛠️ **MaxPooling Layers** – Reduce spatial dimensions while retaining key features.  
🛠️ **Fully Connected Layers** – Perform final emotion classification.  
🛠️ **Activation Functions** – **ReLU** (non-linearity) & **Softmax** (classification).  
🛠️ **Batch Normalization & Dropout** – Prevent overfitting.

### 5️⃣ Training the Model
📌 **Loss Function:** Categorical Crossentropy (for multi-class classification).  
📌 **Optimizer:** Adam (efficient learning rate adaptation).  
📌 **Evaluation Metric:** Accuracy.  
📌 **Training:** Dataset used for training with validation on test data.  
📌 **Performance Visualization:** Accuracy & loss curves plotted.

### 6️⃣ Model Evaluation
📌 Model tested on unseen images for accuracy.  
📌 **Confusion Matrix** plotted for misclassification analysis.  
📌 **Performance Metrics:** Precision, Recall, F1-score computed.

### 7️⃣ Real-time Facial Emotion Recognition App
📌 **Streamlit-based Web App** for real-time emotion detection.  
📌 **OpenCV** captures webcam frames.  
📌 Model **predicts emotions** and overlays results on the screen.  
📌 Accessible via **web browser** for easy testing.

---

## 🎯 Usage
🔹 Upload images or use a webcam for emotion detection.  
🔹 Run inference using the trained model.  
🔹 Visualize predictions using **plots & confusion matrices**.  

---

## 🔮 Future Improvements
🚀 **Enhance accuracy** with a larger, diverse dataset.  
🚀 **Deploy on edge devices** for real-time, low-latency applications.  
🚀 **Integrate with AI assistants** for emotion-aware interactions.  

---

## 📜 Conclusion
This project showcases the power of **deep learning & computer vision** in real-time emotion detection. It has promising applications in **healthcare, security, AI, and human-computer interaction**. 

---

## 👨‍💻 Author
**Abhishek Yadav**  
🚀 Passionate about Machine Learning & AI  
📩 Contact: [Your Email]  
