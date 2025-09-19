import streamlit as st
import joblib
from joblib import load
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
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# Download required NLTK data
# ----------------------------
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

# ----------------------------
# Utility Functions
# ----------------------------
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
        return df_copy.groupby('Approach')[metrics].mean()
    except Exception as e:
        st.error(f"Error calculating average metrics: {e}")
        return None

def count_meaningful_words(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    meaningful_words = [w for w in tokens if w not in stop_words and len(w) > 2 and w.isalpha()]
    return len(meaningful_words)

def preprocess_review_standard_improved(review_text):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded = contractions.fix(text_without_br)
    lowercased = text_expanded.lower()
    tokenized = word_tokenize(lowercased)

    cleaned_tokens = []
    for word in tokenized:
        word = re.sub(r'\d+', '', word)
        word = re.sub(r'[^\w\s]', '', word)
        if word and word.strip():
            cleaned_tokens.append(word.strip())

    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords

    return ' '.join([w for w in cleaned_tokens if w.lower() not in filtered_stopwords])

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def extract_compound_terms(tokens, compounds):
    result, i = [], 0
    while i < len(tokens):
        found = False
        for compound in compounds:
            words = compound.split('_')
            if i + len(words) <= len(tokens) and tokens[i:i+len(words)] == words:
                result.append('_'.join(words))
                i += len(words)
                found = True
                break
        if not found:
            result.append(tokens[i])
            i += 1
    return result

def preprocess_review_pos_driven_improved(review_text, compound_list):
    text_without_br = re.sub(r'<br\s*/>', ' ', review_text)
    text_expanded = contractions.fix(text_without_br)
    lowercased = text_expanded.lower()
    tokenized = word_tokenize(lowercased)

    cleaned_tokens = []
    for word in tokenized:
        word = re.sub(r'\d+', '', word)
        word = re.sub(r'[^\w\s]', '', word)
        if word and word.strip():
            cleaned_tokens.append(word.strip())

    pos_tagged = pos_tag(cleaned_tokens)
    lemmatizer = WordNetLemmatizer()
    sentiment_preserve = {'best','worst','better','worse','amazing','terrible','awful',
                         'excellent','good','decent','okay','acceptable','satisfying',
                         'engaging','strong','succeed'}

    lemmatized = []
    for word, tag in pos_tagged:
        if word.lower() in sentiment_preserve or tag.startswith(('JJ','RB')):
            lemmatized.append(word.lower())
        else:
            lemmatized.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    with_compounds = extract_compound_terms(lemmatized, compound_list)
    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very','really','quite','rather','too','so','not','no','never'}
    filtered_stopwords = stop_words - sentiment_stopwords

    return ' '.join([w for w in with_compounds if w.lower() not in filtered_stopwords])

# ----------------------------
# Load Models & Data
# ----------------------------
@st.cache_resource
def load_comparison_data(file_path):
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        st.error(f"Error loading comparison data: {e}")
        return None

@st.cache_resource
def load_models_and_data():
    models, data = {}, {}
    try:
        # Load Transformer from Hugging Face Hub
        HF_MODEL_REPO = "kelvindabest/sentiment-model"  # 👈 Replace with your HF repo
        try:
            models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
            models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)
            models['sentiment_pipeline'] = pipeline("sentiment-analysis",
                                                    model=models['transformer_model'],
                                                    tokenizer=models['transformer_tokenizer'])
        except Exception as e:
            st.error(f"Error loading Transformer from Hugging Face Hub: {e}")
            models['sentiment_pipeline'] = None

        # Standard TF-IDF
        models['lr_std_tfidf'] = load('logistic_regression_model_for_std_tfidf_baseline.joblib')
        models['nb_std_tfidf'] = load('naive_bayes_model_for_std_tfidf_baseline.joblib')
        models['svm_std_tfidf'] = load('svm_model_for_std_tfidf_baseline.joblib')
        models['tfidf_vectorizer_std'] = load('tfidf_vectorizer_for_std_tfidf_baseline.joblib')

        # POS-Driven
        models['lr_pos_driven'] = load('logistic_regression_model_for_pos_driven.joblib')
        models['nb_pos_driven'] = load('naive_bayes_model_for_pos_driven.joblib')
        models['svm_pos_driven'] = load('svm_model_for_pos_driven.joblib')
        models['tfidf_vectorizer_pos'] = load('tfidf_vectorizer_for_pos_driven.joblib')

        try:
            models['compound_list'] = load('compound_list.joblib')
        except FileNotFoundError:
            st.warning("Compound list missing. POS-Driven may be inaccurate.")
            models['compound_list'] = []

        # Comparison Data
        data['comparison_df'] = load_comparison_data('comparison_df.pkl')

        # Confusion Matrix Data
        try:
            data['y_test_std'] = load('y_test_for_std_tfidf_baseline.joblib')
            data['lr_pred_std'] = load('lr_predictions_for_std_tfidf_baseline.joblib')
            data['nb_pred_std'] = load('nb_predictions_for_std_tfidf_baseline.joblib')
            data['svm_pred_std'] = load('svm_predictions_for_std_tfidf_baseline.joblib')

            data['y_test_pos'] = load('y_test_for_pos_driven.joblib')
            data['lr_pred_pos'] = load('lr_predictions_for_pos_driven.joblib')
            data['nb_pred_pos'] = load('nb_predictions_for_pos_driven.joblib')
            data['svm_pred_pos'] = load('svm_predictions_for_pos_driven.joblib')

            transformer_true = load('true_labels.joblib')
            transformer_pred = load('predicted_labels.joblib')
            label_map = {0: 'negative', 1: 'positive'}
            data['true_labels_transformer_str'] = [label_map.get(l, 'unknown') for l in transformer_true]
            data['predicted_labels_transformer_str'] = [label_map.get(l, 'unknown') for l in transformer_pred]
        except Exception as e:
            st.error(f"Error loading Confusion Matrix data: {e}")

        return models, data
    except Exception as e:
        st.error(f"Error loading models and data: {e}")
        return None, None

# ----------------------------
# MAIN APP
# ----------------------------
models, data = load_models_and_data()

if models:
    st.set_page_config(layout="wide")
    st.title("Movie Review Sentiment Analysis (TF-IDF, POS-Driven & Transformer)")

    # --- Model Comparison ---
    st.subheader("Model Performance Comparison")
    if data and data.get('comparison_df') is not None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        plot_df = data['comparison_df'].set_index('Model')[metrics]

        fig, ax = plt.subplots(figsize=(16, 8))
        bar_width, x = 0.20, np.arange(len(plot_df.index))
        for i, metric in enumerate(metrics):
            values = plot_df[metric].values
            bars = ax.bar(x + i * bar_width, values, bar_width, label=metric)
            for bar in bars:
                height = bar.get_height()
                if pd.notna(height):
                    ax.annotate(f'{height:.4f}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + bar_width * (len(metrics)-1)/2)
        ax.set_xticklabels(plot_df.index, rotation=45, ha='right')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Comparison data not available.")

    # --- Confusion Matrices ---
    st.subheader("Confusion Matrices")
    classes = ['negative','positive']
    if data and all(k in data for k in ['y_test_std','lr_pred_std','nb_pred_std','svm_pred_std',
                                        'y_test_pos','lr_pred_pos','nb_pred_pos','svm_pred_pos',
                                        'true_labels_transformer_str','predicted_labels_transformer_str']):
        # Standard TF-IDF
        st.write("### Standard TF-IDF")
        col1, col2, col3 = st.columns(3)
        for model_name, preds, col in [
            ('LR (Standard TF-IDF)', data['lr_pred_std'], col1),
            ('Naive Bayes (Standard TF-IDF)', data['nb_pred_std'], col2),
            ('SVM (Standard TF-IDF)', data['svm_pred_std'], col3),
        ]:
            with col:
                cm = confusion_matrix(data['y_test_std'], preds, labels=classes)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                fig, ax = plt.subplots()
                disp.plot(cmap=plt.cm.Blues, ax=ax)
                ax.set_title(model_name)
                st.pyplot(fig)

        # POS-Driven
        st.write("### POS-Driven")
        col1, col2, col3 = st.columns(3)
        for model_name, preds, col in [
            ('LR (POS-Driven)', data['lr_pred_pos'], col1),
            ('Naive Bayes (POS-Driven)', data['nb_pred_pos'], col2),
            ('SVM (POS-Driven)', data['svm_pred_pos'], col3),
        ]:
            with col:
                cm = confusion_matrix(data['y_test_pos'], preds, labels=classes)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                fig, ax = plt.subplots()
                disp.plot(cmap=plt.cm.Blues, ax=ax)
                ax.set_title(model_name)
                st.pyplot(fig)

        # Transformer
        st.write("### Transformer")
        cm = confusion_matrix(data['true_labels_transformer_str'], data['predicted_labels_transformer_str'], labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title("Transformer")
        st.pyplot(fig)
    else:
        st.warning("Confusion Matrix data missing.")

    # --- Predict Review ---
    st.subheader("Predict Movie Review")
    user_input = st.text_area("Enter your movie review here:", height=200)

    if st.button("Analyze Sentiment"):
        if user_input:
            word_count = count_meaningful_words(user_input.strip())
            if word_count >= 5:
                st.success(f"Input quality: 🟢 High ({word_count} meaningful words)")
            elif word_count >= 3:
                st.info(f"Input quality: 🟡 Medium ({word_count} meaningful words)")
            else:
                st.warning(f"Input quality: 🔴 Low ({word_count} meaningful words)")

            results = {}

            # Standard TF-IDF
            try:
                processed = preprocess_review_standard_improved(user_input)
                features = models['tfidf_vectorizer_std'].transform([processed])
                lr_pred = models['lr_std_tfidf'].predict(features)[0]
                lr_prob = max(models['lr_std_tfidf'].predict_proba(features)[0])
                results['Standard TF-IDF + Logistic Regression'] = {'prediction': lr_pred, 'confidence': lr_prob}

                nb_pred = models['nb_std_tfidf'].predict(features)[0]
                nb_prob = max(models['nb_std_tfidf'].predict_proba(features)[0])
                results['Standard TF-IDF + Naive Bayes'] = {'prediction': nb_pred, 'confidence': nb_prob}

                svm_pred = models['svm_std_tfidf'].predict(features)[0]
                svm_score = abs(models['svm_std_tfidf'].decision_function(features)[0])
                results['Standard TF-IDF + SVM'] = {'prediction': svm_pred, 'confidence': svm_score}
            except Exception as e:
                st.error(f"Error during Standard TF-IDF prediction: {e}")

            # POS-Driven
            try:
                processed_pos = preprocess_review_pos_driven_improved(user_input, models['compound_list'])
                features_pos = models['tfidf_vectorizer_pos'].transform([processed_pos])
                lr_pred = models['lr_pos_driven'].predict(features_pos)[0]
                lr_prob = max(models['lr_pos_driven'].predict_proba(features_pos)[0])
                results['POS-Driven + Logistic Regression'] = {'prediction': lr_pred, 'confidence': lr_prob}

                nb_pred = models['nb_pos_driven'].predict(features_pos)[0]
                nb_prob = max(models['nb_pos_driven'].predict_proba(features_pos)[0])
                results['POS-Driven + Naive Bayes'] = {'prediction': nb_pred, 'confidence': nb_prob}

                svm_pred = models['svm_pos_driven'].predict(features_pos)[0]
                svm_score = abs(models['svm_pos_driven'].decision_function(features_pos)[0])
                results['POS-Driven + SVM'] = {'prediction': svm_pred, 'confidence': svm_score}
            except Exception as e:
                st.error(f"Error during POS-Driven prediction: {e}")

            # Transformer
            try:
                if models['sentiment_pipeline']:
                    transformer_result = models['sentiment_pipeline'](user_input)[0]
                    label = transformer_result['label'].replace("LABEL_", "")
                    if label.isdigit():
                        label = "positive" if int(label) == 1 else "negative"
                    results['Transformer'] = {'prediction': label, 'confidence': transformer_result['score']}
                else:
                    results['Transformer'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
            except Exception as e:
                st.error(f"Error during Transformer prediction: {e}")

            # Display results
            st.write("---")
            st.subheader("Individual Model Predictions:")
            for model_name, result in results.items():
                sentiment = str(result['prediction']).upper()
                conf = result['confidence']
                if 'SVM' in model_name:
                    conf_str = f"Decision Score: {conf:.4f}"
                elif model_name == 'Transformer':
                    conf_str = f"Score: {conf:.4f}"
                else:
                    conf_str = f"Confidence: {conf:.4f}"

                if sentiment == 'POSITIVE':
                    st.markdown(f"✓ **{model_name}**: <span style='color:green'>**{sentiment}**</span> ({conf_str})", unsafe_allow_html=True)
                elif sentiment == 'NEGATIVE':
                    st.markdown(f"✗ **{model_name}**: <span style='color:red'>**{sentiment}**</span> ({conf_str})", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{model_name}**: {sentiment} ({conf_str})", unsafe_allow_html=True)

            # Overall Summary
            pos_count = sum(1 for r in results.values() if str(r['prediction']).lower() == 'positive')
            neg_count = sum(1 for r in results.values() if str(r['prediction']).lower() == 'negative')
            total = pos_count + neg_count
            st.write("---")
            st.subheader("Overall Summary:")
            st.write(f"Positive predictions: {pos_count}/{total}")
            st.write(f"Negative predictions: {neg_count}/{total}")
            if pos_count > neg_count:
                st.markdown(f"Overall sentiment: <span style='color:green'>**POSITIVE**</span>", unsafe_allow_html=True)
            elif neg_count > pos_count:
                st.markdown(f"Overall sentiment: <span style='color:red'>**NEGATIVE**</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Overall sentiment: **MIXED**", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review before analyzing.")
