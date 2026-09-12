# payment-fraud-detection-using-ANN
A robust deep learning framework utilizing Artificial Neural Networks (ANN) to analyze and detect fraudulent financial transactions in real-time.
# Fraud Detection Neural Network 

A complete deep learning solution for detecting fraudulent transactions, featuring a trained TensorFlow/Keras neural network and an interactive web dashboard built with Streamlit. 

## 🧠 Key Insights & Problem Solving: Handling Imbalanced Data

During the development and training of the neural network, the most significant hurdle was the heavily skewed nature of the dataset. Legitimate transactions vastly outnumbered fraudulent ones. 

* **The Problem:** Because the data was highly imbalanced, the initial model gave disproportionate importance to the frequently occurring terms (the majority "Not Fraud" class). The network realized it could achieve a high baseline accuracy simply by predicting "Not Fraud" every time, effectively failing to learn the underlying patterns of actual fraud. 
* **The Impact:** While overall raw accuracy seemed artificially high, the model's actual performance (Recall/Sensitivity) on the minority class was poor. 
* **The Solution:** The issue was resolved by implementing **Class Weights** during the training phase. By calculating and assigning weights that penalized the model more heavily for misclassifying the minority class, the network was forced to treat both classes with equal importance. This successfully mitigated the bias toward frequent occurrences, resulting in a robust model that accurately identifies fraudulent patterns rather than just guessing the majority class.

## ⚙️ Tech Stack

* **Machine Learning:** TensorFlow, Keras, Scikit-Learn (preprocessing/scaling)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib (for `.pkl` scalers), Keras (for `.keras` models)

## 🚀 Features

* **Real-Time Prediction:** Users can input transaction parameters into the web interface and receive an instant fraud probability score.
* **Deep Learning Architecture:** Utilizes a feed-forward neural network to capture complex, non-linear relationships in transaction data.
* **Standardized Inputs:** Integrates a pre-fitted Scikit-Learn scaler to ensure user input data perfectly matches the distribution of the training data.
* **Interactive UI:** Provides immediate, color-coded visual feedback based on the risk threshold.

## 🛠️ Local Installation & Setup

If you are running this project locally on a Windows machine, follow these steps to ensure all dependencies and security policies are handled correctly.

**1. Clone the repository and navigate to the directory**
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name

**Create and activate a virtual environment**
python -m venv venv
# Activate the environment in PowerShell
.\venv\Scripts\Activate.ps1

**Install dependencies**
pip install tensorflow streamlit streamlit-lottie scikit-learn joblib pandas numpy

💻 Usage
Ensure your virtual environment is active, then launch the Streamlit dashboard by running the script directly through your environment's Python engine:

PowerShell
.\venv\Scripts\python.exe -m streamlit run pya
