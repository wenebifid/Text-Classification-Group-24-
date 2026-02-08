# Hate Speech Detection using LSTM

## Project Overview
This project implements a deep learning-based system for detecting hate speech in tweets, classifying them into three categories: Hate Speech, Offensive Language, and Neither. The system leverages various word embedding techniques (TF-IDF, Word2Vec, GloVe) with LSTM networks for text classification.

## Key Features
- Text preprocessing (cleaning, tokenization, stopword removal, lemmatization)
- Multiple embedding techniques (TF-IDF, Word2Vec, GloVe)
- Bidirectional LSTM architecture with attention mechanisms
- Class imbalance handling using SMOTE
- Comprehensive model evaluation and visualization
- Model persistence for production deployment

## Dataset
The model is trained on the Davidson et al. (2017) hate speech dataset, which contains labeled tweets with the following distribution:
- Hate Speech
- Offensive Language
- Neither

## Model Performance
The best performing model achieved the following metrics:

| Model | Accuracy | Val_accuracy | Val_loss | Learning_rate |
|------|----------|-----------|--------|----------|
| GloVe + LSTM | 0.87 | 0.87 | 0.33 | 0.001 |
| TF-IDF + LSTM | 0.61 | 0.606 | 4.2 | 2.5000e-04 |
| Word2Vec Skip-gram + LSTM | 0.83 | 0.84 | 0.39 | 0.001 |

## Best Model Architecture (Word2Vec + LSTM)
1. Embedding Layer: Pre-trained Word2Vec embeddings (300 dimensions)
2. Bidirectional LSTM: 128 units with return sequences  
   - Dropout: 0.3  
   - Recurrent Dropout: 0.2
3. Bidirectional LSTM: 64 units  
   - Dropout: 0.3  
   - Recurrent Dropout: 0.2
4. Dense Layers:
   - 64 units with ReLU activation
   - Batch Normalization
   - Dropout: 0.3
   - 32 units with ReLU activation
   - Dropout: 0.2
5. Output Layer: 3 units with Softmax activation

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- NLTK
- Gensim
- imbalanced-learn

### Installation
Clone the repository:
git clone https://github.com/wenebifid/Text-Classification-Group-24-.git

Navigate into the project directory:
cd Text-Classification-Group-24-

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate
(On Windows use: venv\\Scripts\\activate)

## Usage

### Data Preparation
- Place your dataset in the data/ directory
- The dataset should be a CSV file with text and label columns

### Training the Model
Run:
python train.py

### Making Predictions
Example usage in Python:

from tensorflow.keras.models import load_model  
import pickle  
import numpy as np  
from tensorflow.keras.preprocessing.sequence import pad_sequences  

Load the model and tokenizer:
model = load_model('models/best_hate_speech_model.keras')  
with open('models/tokenizer.pkl', 'rb') as f:  
    tokenizer = pickle.load(f)

Prediction function:
def predict(text):  
    sequence = tokenizer.texts_to_sequences([text])  
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH)  
    prediction = model.predict(padded)  
    return ['Hate Speech', 'Offensive', 'Neither'][prediction.argmax()]

## Acknowledgements
- Davidson et al. for the hate speech dataset
- TensorFlow and Keras for the deep learning framework
- Gensim for word embeddings
