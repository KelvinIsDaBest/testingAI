import streamlit as st
import joblib
from joblib import load
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

HF_MODEL_REPO = "kelvindabest/sentiment-model"

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

download_nltk_data()


def calculate_average_metrics(comparison_df):
    if comparison_df is None or comparison_df.empty:
        return None
    try:
        df_copy = comparison_df.copy()
        df_copy['Approach'] = df_copy['Model'].apply(
            lambda x: 'Standard TF-IDF' if 'Standard TF-IDF' in x else (
                'POS-Driven' if 'POS-Driven' in x else 'Transformer'
            )
        )
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        average_metrics = df_copy.groupby('Approach')[metrics].mean()
        return average_metrics
    except Exception as e:
        st.error(f"Error calculating average metrics: {e}")
        return None


def count_meaningful_words(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in tokens if word not in stop_words and len(word) > 2 and word.isalpha()]
    return len(meaningful_words)


def preprocess_review_standard_improved(review_text):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded = contractions.fix(text_without_br)
    lowercased = text_expanded.lower()
    tokens = word_tokenize(lowercased)
    cleaned_tokens = [re.sub(r'[^\w\s]', '', w) for w in tokens if w and w.strip()]
    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords
    final_tokens = [w for w in cleaned_tokens if w not in filtered_stopwords]
    return ' '.join(final_tokens)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_review_pos_driven_improved(review_text, compound_list):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded = contractions.fix(text_without_br)
    lowercased = text_expanded.lower()
    tokens = word_tokenize(lowercased)
    cleaned_tokens = [re.sub(r'[^\w\s]', '', w) for w in tokens if w and w.strip()]

    pos_tagged = pos_tag(cleaned_tokens)
    lemmatizer = WordNetLemmatizer()
    sentiment_preserve = {'best', 'worst', 'better', 'worse', 'amazing', 'terrible', 'awful', 'excellent',
                          'good', 'decent', 'okay', 'acceptable', 'satisfying', 'engaging', 'strong', 'succeed'}
    processed_tokens = []
    for word, tag in pos_tagged:
        if word.lower() in sentiment_preserve or tag.startswith('JJ') or tag.startswith('RB'):
            processed_tokens.append(word.lower())
        else:
            processed_tokens.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords
    final_tokens = [w for w in processed_tokens if w not in filtered_stopwords]
    return ' '.join(final_tokens)


@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}

    # Hugging Face transformer model
    try:
        models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)
        models['sentiment_pipeline'] = pipeline("sentiment-analysis", model=models['transformer_model'], tokenizer=models['transformer_tokenizer'])
    except Exception as e:
        st.error(f"Error loading Transformer model from Hugging Face: {e}")
        models['sentiment_pipeline'] = None

    # Load local models
    try:
        models['lr_std_tfidf'] = load('logistic_regression_model_for_std_tfidf_baseline.joblib')
        models['nb_std_tfidf'] = load('naive_bayes_model_for_std_tfidf_baseline.joblib')
        models['svm_std_tfidf'] = load('svm_model_for_std_tfidf_baseline.joblib')
        models['tfidf_vectorizer_std'] = load('tfidf_vectorizer_for_std_tfidf_baseline.joblib')

        models['lr_pos_driven'] = load('logistic_regression_model_for_pos_driven.joblib')
        models['nb_pos_driven'] = load('naive_bayes_model_for_pos_driven.joblib')
        models['svm_pos_driven'] = load('svm_model_for_pos_driven.joblib')
        models['tfidf_vectorizer_pos'] = load('tfidf_vectorizer_for_pos_driven.joblib')

        models['compound_list'] = load('compound_list.joblib')
        data['comparison_df'] = pd.read_pickle('comparison_df.pkl')
    except Exception as e:
        st.warning(f"Some local models or data could not be loaded: {e}")

    return models, data


models, data = load_models_and_data()

if models:
    st.set_page_config(layout="wide")
    st.title("Movie Reviews Sentiment Analysis (TF-IDF + POS + Transformer via Hugging Face)")

    # ----------------- Prediction UI -----------------
    st.subheader("Predict Movie Review")
    user_input = st.text_area("Enter your movie review here:", height=200)

    if st.button("Analyze Sentiment"):
        if user_input:
            word_count = count_meaningful_words(user_input.strip())
            if word_count >= 5:
                st.success(f"🟢 High input quality ({word_count} meaningful words)")
            elif word_count >= 3:
                st.info(f"🟡 Medium input quality ({word_count} meaningful words)")
            else:
                st.warning(f"🔴 Low input quality ({word_count} meaningful words)")

            results = {}

            # Transformer
            if 'sentiment_pipeline' in models and models['sentiment_pipeline']:
                try:
                    transformer_result = models['sentiment_pipeline'](user_input)[0]
                    results['Transformer (Hugging Face)'] = {
                        'prediction': transformer_result['label'],
                        'confidence': transformer_result['score']
                    }
                except Exception as e:
                    st.error(f"Transformer prediction error: {e}")

            # Show results
            st.write("---")
            st.subheader("Results:")
            for model_name, result in results.items():
                st.write(f"**{model_name}** → {result['prediction']} (score: {result['confidence']:.4f})")

else:
    st.error("No models available.")

