# 🎭 Emotion Detection Model

A Jupyter notebook project for building an emotion detection model using Machine Learning and Natural Language Processing (NLP).

## 📘 Overview
This project analyzes text data to detect human emotions such as happiness, sadness, anger, fear, surprise, and neutrality.  
It uses data preprocessing, TF-IDF feature extraction, and machine-learning classifiers.

## ⚙️ Technologies Used
- Python 🐍  
- scikit-learn  
- Pandas, NumPy  
- NLTK / spaCy  
- Jupyter Notebook  

## 🧠 Model Details
- **Algorithm:** Logistic Regression / Naive Bayes  
- **Dataset:** Emotion labeled text dataset  
- **Feature Extraction:** TF-IDF Vectorizer  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score  

## 📊 Results
Dataset source: dair-ai/emotion

Number of samples: 20000

Number of classes: 6

The least frequent class is surprise with 3.6% of the data.

Best classical model: LinearSVC

Validation Accuracy: 0.8840

Validation Macro-F1: 0.8433

Test Accuracy: 0.8860

Test Macro-F1: 0.8490

Lowest per-class F1 is for surprise: 0.7576.

Interpretability examples: anger: irritable, dangerous, petty; fear: shaken, terrified, vulnerable

Ablation study: Variant A Macro-F1 0.8470, Variant B Macro-F1 0.8433. Unigrams sufficed and bigrams did not help.

Error analysis: The model tends to confuse classes when the text contains mixed or sarcastic expressions, or when utterances are short and ambiguous.

Future work: Investigate class weighting or oversampling to address imbalance, handle emojis and hashtags, incorporate domain-specific vocabulary, fine-tune a transformer on longer sequences, and calibrate probability estimates.

## 📁 Repository Structure
