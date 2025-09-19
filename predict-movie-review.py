import streamlit as st
import joblib
from joblib import dump, load
import os
import re
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import zipfile
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

download_nltk_data()

TRANSFORMER_MODEL_DIR = "./sentiment_model"
TRANSFORMER_ZIP_FILE = "sentiment_model.zip"

def extract_transformer_model(zip_path, extract_dir):
    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        try:
            # check if the zip file exists at the provided path
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                #st.success("Transformer model extracted successfully.")
            else:
                 st.error(f"Error: Transformer zip file not found at {zip_path}.")
        except Exception as e:
            st.error(f"An error occurred during Transformer model extraction: {e}")
    else:
        pass

# ADD WORD COUNTING FUNCTION FOR RELIABILITY INDICATORS
def count_meaningful_words(text):
    """
    Count meaningful words (excluding stopwords, punctuation, etc.)
    """
    # Basic cleaning
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = word_tokenize(text)

    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in tokens
                       if word not in stop_words
                       and len(word) > 2
                       and word.isalpha()]

    return len(meaningful_words)


# ALL METHODS
def preprocess_review_standard_improved(review_text):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded_contractions = contractions.fix(text_without_br)
    lowercased_review = text_expanded_contractions.lower()
    tokenized_review = word_tokenize(lowercased_review)

    cleaned_tokens = []
    for word in tokenized_review:
        word = re.sub(r'\d+', '', word)
        word = re.sub(r'[^\w\s]', '', word)
        word = re.sub(r'^s$', '', word)
        if word and word.strip():
            cleaned_tokens.append(word.strip())

    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords

    cleaned_review_without_stopword = []
    for word in cleaned_tokens:
        if word.lower() not in filtered_stopwords:
            cleaned_review_without_stopword.append(word)

    return ' '.join(cleaned_review_without_stopword)

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

def extract_compound_terms(tokens, compounds):
    result = []
    i = 0
    while i < len(tokens):
        found = False
        for compound in compounds:
            compound_words = compound.split('_')
            if i + len(compound_words) <= len(tokens):
                match = True
                for j in range(len(compound_words)):
                    if tokens[i+j] != compound_words[j]:
                        match = False
                        break
                if match:
                    result.append('_'.join(compound_words))
                    i += len(compound_words)
                    found = True
                    break
        if not found:
            result.append(tokens[i])
            i += 1
    return result

def preprocess_review_pos_driven_improved(review_text, compound_list):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded_contractions = contractions.fix(text_without_br)
    lowercased_review = text_expanded_contractions.lower()
    tokenized_review = word_tokenize(lowercased_review)

    cleaned_tokens = []
    for word in tokenized_review:
        word = re.sub(r'\d+', '', word)
        word = re.sub(r'[^\w\s]', '', word)
        word = re.sub(r'^s$', '', word)
        if word and word.strip():
            cleaned_tokens.append(word.strip())

    # POS Tagging
    pos_tagged_review = pos_tag(cleaned_tokens)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    sentiment_preserve = {'best', 'worst', 'better', 'worse', 'amazing', 'terrible', 'awful', 'excellent',
                         'good', 'decent', 'okay', 'acceptable', 'satisfying', 'engaging', 'strong', 'succeed'}

    cleaned_review_pos = []
    for word, tag in pos_tagged_review:
        if word.lower() in sentiment_preserve or tag.startswith('JJ') or tag.startswith('RB'):
            cleaned_review_pos.append(word.lower())
        else:
            cleaned_review_pos.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    # Extract compound terms
    cleaned_review_with_compounds = extract_compound_terms(cleaned_review_pos, compound_list)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords

    cleaned_review_with_compounds_without_stopword = []
    for word in cleaned_review_with_compounds:
        if word.lower() not in filtered_stopwords:
            cleaned_review_with_compounds_without_stopword.append(word)

    return ' '.join(cleaned_review_with_compounds_without_stopword)


# LOAD MODELS

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    try:
        # Call the extraction function for Transformer model first
        extract_transformer_model(TRANSFORMER_ZIP_FILE, TRANSFORMER_MODEL_DIR)

        # Load Transformer model and tokenizer
        if os.path.exists(TRANSFORMER_MODEL_DIR) and os.listdir(TRANSFORMER_MODEL_DIR):
             try:
                 models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
                 models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
                 models['sentiment_pipeline'] = pipeline("sentiment-analysis", model=models['transformer_model'], tokenizer=models['transformer_tokenizer'])
                 #st.write("Transformer model and tokenizer loaded.")
             except Exception as e:
                 st.error(f"Error loading Transformer model and tokenizer: {e}")
                 # Handle case where transformer model loading fails
        else:
             st.warning(f"Transformer model directory not found or is empty at {TRANSFORMER_MODEL_DIR}. Transformer model will not be available.")
             # Handle case where transformer model is not available


        # Load Standard TF-IDF models from the root directory
        models['lr_std_tfidf'] = load('logistic_regression_model_for_std_tfidf_baseline.joblib')
        models['nb_std_tfidf'] = load('naive_bayes_model_for_std_tfidf_baseline.joblib')
        models['svm_std_tfidf'] = load('svm_model_for_std_tfidf_baseline.joblib')
        models['tfidf_vectorizer_std'] = load('tfidf_vectorizer_for_std_tfidf_baseline.joblib')
        #st.write("Standard TF-IDF models loaded.")


        # Load POS-Driven models from the root directory
        models['lr_pos_driven'] = load('logistic_regression_model_for_pos_driven.joblib')
        models['nb_pos_driven'] = load('naive_bayes_model_for_pos_driven.joblib')
        models['svm_pos_driven'] = load('svm_model_for_pos_driven.joblib')
        models['tfidf_vectorizer_pos'] = load('tfidf_vectorizer_for_pos_driven.joblib')
        #st.write("POS-Driven models loaded.")


        # Load compound_list from the root directory
        try:
            models['compound_list'] = load('compound_list.joblib')
            #st.write("Compound list loaded.")
        except FileNotFoundError:
             st.warning(f"Compound list not found at 'compound_list.joblib'. POS-Driven models might not work correctly.")
             models['compound_list'] = [] # Placeholder

        return models, data
    except FileNotFoundError as e:
        st.error(f"Error loading a model file: {e}. Please ensure all model files are at the repository root or in the specified directories.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models and data: {e}")
        return None, None

models, data = load_models_and_data()

if models:
    # FRONT END
    st.title("Large-Scale Movie Reviews Sentiment Analysis Through TF-IDF, POS-Driven Phrase-Level Feature Engineering and Transformer")
    st.set_page_config(layout="wide")

# Predict Review
    st.subheader("Predict Movie Review")

    user_input = st.text_area("Enter your movie review here:", height=200)

    if st.button("Analyze Sentiment"):
        if user_input:
            # Count meaningful words for reliability indicator
            word_count = count_meaningful_words(user_input.strip())

            st.subheader("Analysis Results:")

            # Add reliability indicator
            if word_count >= 5:
                reliability_icon = "🟢"
                reliability_text = "High"
                st.success(f"Input quality: {reliability_icon} {reliability_text} ({word_count} meaningful words)")
            elif word_count >= 3:
                reliability_icon = "🟡"
                reliability_text = "Medium"
                st.info(f"Input quality: {reliability_icon} {reliability_text} ({word_count} meaningful words)")
            else:
                reliability_icon = "🔴"
                reliability_text = "Low"
                st.warning(f"Input quality: {reliability_icon} {reliability_text} ({word_count} meaningful words) - Results may be less accurate")

            results = {}

            if all(model_name in models for model_name in ['lr_std_tfidf', 'nb_std_tfidf', 'svm_std_tfidf', 'tfidf_vectorizer_std']):
                try:
                    processed_std = preprocess_review_standard_improved(user_input)
                    std_tfidf_features = models['tfidf_vectorizer_std'].transform([processed_std])

                    lr_pred_std = models['lr_std_tfidf'].predict(std_tfidf_features)[0]
                    lr_prob_std = models['lr_std_tfidf'].predict_proba(std_tfidf_features)[0]
                    results['Standard TF-IDF + Logistic Regression'] = {
                        'prediction': lr_pred_std,
                        'confidence': max(lr_prob_std)
                    }

                    nb_pred_std = models['nb_std_tfidf'].predict(std_tfidf_features)[0]
                    nb_prob_std = models['nb_std_tfidf'].predict_proba(std_tfidf_features)[0]
                    results['Standard TF-IDF + Naive Bayes'] = {
                        'prediction': nb_pred_std,
                        'confidence': max(nb_prob_std)
                    }

                    svm_pred_std = models['svm_std_tfidf'].predict(std_tfidf_features)[0]
                    svm_decision_std = models['svm_std_tfidf'].decision_function(std_tfidf_features)[0]
                    results['Standard TF-IDF + SVM'] = {
                        'prediction': svm_pred_std,
                        'confidence': abs(svm_decision_std)
                    }
                except Exception as e:
                     st.error(f"Error during Standard TF-IDF model prediction: {e}")
                     results['Standard TF-IDF + Logistic Regression'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['Standard TF-IDF + Naive Bayes'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['Standard TF-IDF + SVM'] = {'prediction': 'Error', 'confidence': 0.0}

            else:
                 st.warning("Standard TF-IDF models were not loaded successfully. Skipping predictions for these models.")
                 results['Standard TF-IDF + Logistic Regression'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['Standard TF-IDF + Naive Bayes'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['Standard TF-IDF + SVM'] = {'prediction': 'Not Loaded', 'confidence': 0.0}

            if all(model_name in models for model_name in ['lr_pos_driven', 'nb_pos_driven', 'svm_pos_driven', 'tfidf_vectorizer_pos']) and 'compound_list' in models:
                try:
                    compound_list = models.get('compound_list', [])
                    processed_pos = preprocess_review_pos_driven_improved(user_input, compound_list)
                    pos_tfidf_features = models['tfidf_vectorizer_pos'].transform([processed_pos])

                    lr_pred_pos = models['lr_pos_driven'].predict(pos_tfidf_features)[0]
                    lr_prob_pos = models['lr_pos_driven'].predict_proba(pos_tfidf_features)[0]
                    results['POS-Driven + Logistic Regression'] = {
                        'prediction': lr_pred_pos,
                        'confidence': max(lr_prob_pos)
                    }

                    nb_pred_pos = models['nb_pos_driven'].predict(pos_tfidf_features)[0]
                    nb_prob_pos = models['nb_pos_driven'].predict_proba(pos_tfidf_features)[0] # Corrected typo here
                    results['POS-Driven + Naive Bayes'] = {
                        'prediction': nb_pred_pos,
                        'confidence': max(nb_prob_pos)
                    }

                    svm_pred_pos = models['svm_pos_driven'].predict(pos_tfidf_features)[0]
                    svm_decision_pos = models['svm_pos_driven'].decision_function(pos_tfidf_features)[0]
                    results['POS-Driven + SVM'] = {
                        'prediction': svm_pred_pos,
                        'confidence': abs(svm_decision_pos)
                    }
                except Exception as e:
                     st.error(f"Error during POS-Driven model prediction: {e}")
                     results['POS-Driven + Logistic Regression'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['POS-Driven + Naive Bayes'] = {'prediction': 'Error', 'confidence': 0.0}
                     results['POS-Driven + SVM'] = {'prediction': 'Error', 'confidence': 0.0}

            else:
                 st.warning("POS-Driven models were not loaded successfully. Skipping predictions for these models.")
                 results['POS-Driven + Logistic Regression'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['POS-Driven + Naive Bayes'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
                 results['POS-Driven + SVM'] = {'prediction': 'Not Loaded', 'confidence': 0.0}

            try:
                if 'sentiment_pipeline' in models and models['sentiment_pipeline']:
                     transformer_result = models['sentiment_pipeline'](user_input)[0]
                     label_map = {0: "negative", 1: "positive"}
                     label_str = transformer_result['label'].replace("LABEL_", "")
                     try:
                         label_id = int(label_str)
                         sentiment_label = label_map.get(label_id, "unknown")
                         results['Transformer'] = {
                             'prediction': sentiment_label,
                             'confidence': transformer_result['score']
                         }
                     except ValueError:
                         st.warning(f"Could not parse transformer label ID: {label_str}")
                         results['Transformer'] = {'prediction': 'Error', 'confidence': 0.0}
                else:
                     results['Transformer'] = {'prediction': 'Not Loaded', 'confidence': 0.0}

            except Exception as e:
                 st.error(f"Error with Transformer prediction: {e}")
                 results['Transformer'] = {'prediction': 'Error', 'confidence': 0.0}



            # Display results with reliability context
            st.write("---")
            st.subheader("Individual Model Predictions:")

            for model_name, result in results.items():
                sentiment = result['prediction'].upper()
                confidence = result['confidence']

                if 'SVM' in model_name:
                    conf_str = f"Decision Score: {confidence:.4f}"
                elif model_name == 'Transformer':
                    conf_str = f"Score: {confidence:.4f}"
                else:
                    conf_str = f"Confidence: {confidence:.4f}"

                if sentiment == 'POSITIVE':
                    st.markdown(f"✓ **{model_name}**: <span style='color:green'>**{sentiment}**</span> ({conf_str})", unsafe_allow_html=True)
                elif sentiment == 'NEGATIVE':
                    st.markdown(f"✗ **{model_name}**: <span style='color:red'>**{sentiment}**</span> ({conf_str})", unsafe_allow_html=True)
                else:
                    st.markdown(f"  **{model_name}**: {sentiment} ({conf_str})", unsafe_allow_html=True)

            positive_count = sum(1 for result in results.values() if result['prediction'].lower() == 'positive')
            negative_count = sum(1 for result in results.values() if result['prediction'].lower() == 'negative')
            total_predictions = positive_count + negative_count # Only count models that successfully predicted

            st.write("---")
            st.subheader("Overall Summary:")
            st.write(f"Positive predictions: {positive_count}/{total_predictions}")
            st.write(f"Negative predictions: {negative_count}/{total_predictions}")

            if positive_count > negative_count:
                overall = "POSITIVE"
                st.markdown(f"Overall sentiment: <span style='color:green'>**{overall}**</span>", unsafe_allow_html=True)
            elif negative_count > positive_count:
                overall = "NEGATIVE"
                st.markdown(f"Overall sentiment: <span style='color:red'>**{overall}**</span>", unsafe_allow_html=True)
            else:
                overall = "MIXED (TIE)"
                st.markdown(f"Overall sentiment: **{overall}**", unsafe_allow_html=True)


        else:
            st.warning("Please enter a movie review to analyze.")

else:
    st.error("Models could not be loaded. Please ensure models are saved and accessible.")
