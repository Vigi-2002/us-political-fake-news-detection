# U.S. Political Fake News Detection

A deep learning–based fake news detection system focused on U.S. political news articles. The model uses NLP preprocessing and a BiLSTM architecture to classify news as real or fake, deployed using Streamlit for interactive prediction.

This system analyzes **linguistic and stylistic writing patterns** learned from historical datasets — it does **not verify factual correctness**

---

## 🔍 Project Overview

Fake news has significantly impacted political communication, particularly during elections and policy debates.

This project aims to:

- Analyze US political news articles
- Learn language patterns associated with fake and real news
- Estimate the **probability of an article being fake**
- Deploy the trained model using a Streamlit web application

---

## 📌 Domain Scope

✅ US political news articles  
❌ International news  
❌ Sports, entertainment, or general content  

> Predictions are reliable **only within the US political domain**.

This limitation reflects real-world machine learning practices and helps avoid misleading results caused by domain shift.

---

## 🧠 Model Architecture

- Text preprocessing and normalization  
- Tokenization and sequence padding  
- Embedding layer (128 dimensions)  
- **Bidirectional LSTM (Bi-LSTM)**  
- Global Max Pooling  
- Fully connected dense layers  
- Sigmoid output for probability estimation  

---

## ⚙️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **NLTK**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Streamlit**

---

## 📊 Model Evaluation

To ensure realistic performance:

- Stratified train–test split
- Duplicate text overlap removed
- Data leakage checks performed
- Evaluation metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 🔎 Prediction Interpretation

Instead of binary classification, predictions are interpreted using probability ranges:

| Fake Probability | Interpretation |
|------------------|----------------|
| < 0.30 | Likely Real |
| 0.30 – 0.60 | Uncertain / Mixed |
| > 0.60 | Likely Fake |

This approach avoids overconfident predictions and better reflects real-world ML deployment.

---

## 🌐 Streamlit Web Application

The interactive web app allows users to:

- Paste a US political news article
- Load random example articles
- View fake news probability
- Receive confidence-based interpretation
- Understand model limitations through disclaimers

---

## ⚠️ Disclaimer

This model does **not determine factual truth**.

It evaluates **linguistic and stylistic patterns** learned from historical US political news datasets.

Predictions should be used strictly for **educational and experimental purposes**.

---

## 📁 Project Structure

Fake-News-Detection/
│
├── app.py
├── fake_news_model.h5
├── tokenizer.pkl
├── fake_news_training.ipynb
├── README.md

---

## 🎯 Key Learning Outcomes

- End-to-end NLP pipeline design  
- Deep learning for text classification  
- Handling data leakage in NLP projects  
- Understanding domain shift  
- Model deployment using Streamlit  
- Ethical communication of ML limitations  

---

## 👨‍💻 Author

**Vighnesh**  
Aspiring Data Scientist  

---

⭐ If you found this project useful, feel free to star the repository!
