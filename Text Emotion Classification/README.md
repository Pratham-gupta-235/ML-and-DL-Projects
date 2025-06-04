# Text Emotion Classification using Deep Learning 🎭

This project uses a deep learning model built with **TensorFlow/Keras** to classify text into emotional categories. It processes a dataset of text-emotion pairs and builds an NLP pipeline using tokenization, padding, label encoding, and a feedforward neural network.

---

## 📌 Project Overview

The aim is to classify given text messages into emotions like `anger`, `joy`, `sadness`, `fear`, etc. This project uses deep learning methods instead of traditional machine learning algorithms.

---

## 📁 Dataset

* **Source**: `Dataset/train.txt`
* Format: `Text;Emotion` (semicolon-separated)
* Example:

  | Text                      | Emotion |
  | ------------------------- | ------- |
  | i feel happy this morning | joy     |
  | i am not looking forward  | sadness |
  | he is my best friend      | love    |

---

## 🧠 Tech Stack

* Python 🐍
* TensorFlow / Keras
* Pandas / NumPy
* Scikit-learn (for label encoding, train-test split)

---

## 🧪 Model Architecture

* **Embedding Layer** – Learns word representations
* **Flatten Layer** – Flattens embeddings
* **Dense Layers** – Fully connected layers for classification
* **Loss** – `categorical_crossentropy`
* **Optimizer** – `adam`
* **Metric** – `accuracy`

---

## 🔄 Workflow

1. **Load & Preprocess Data**:

   * Read semicolon-separated file
   * Rename columns to `Text`, `Emotions`

2. **Text Tokenization**:

   * Convert text to sequences using Keras `Tokenizer`
   * Pad sequences to uniform length

3. **Label Encoding**:

   * Convert string labels into numeric using `LabelEncoder`
   * One-hot encode using `to_categorical`

4. **Model Building**:

   ```python
   model = Sequential()
   model.add(Embedding(vocab_size, 100, input_length=max_len))
   model.add(Flatten())
   model.add(Dense(32, activation='relu'))
   model.add(Dense(len(label_classes), activation='softmax'))
   ```

5. **Training**:

   * Train with `epochs=10`, `batch_size=32`

6. **Evaluation**:

   * Training and validation accuracy printed
   * Optionally test on new data

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Pratham-gupta-235/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Text Emotion Classification
```

### 2. Install Requirements

```bash
pip install tensorflow pandas scikit-learn
```

### 3. Launch Notebook

```bash
jupyter notebook classification.ipynb
```

---

## 📊 Results

* The model learns to distinguish between emotion classes using embeddings and simple dense layers.
* Can be extended to use LSTM, GRU, or Transformer-based models for better accuracy.

---

## 🔧 Future Improvements

* Replace Dense model with LSTM/GRU/Transformer
* Add more robust preprocessing (stopword removal, lemmatization)
