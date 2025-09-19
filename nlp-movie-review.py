import os
import re
import streamlit as st
import joblib
from joblib import load
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="wide")

# ----------------------------
# Download NLTK data (cached)
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
# Utility functions
# ----------------------------
def safe_load_joblib(path, name):
    try:
        return load(path)
    except FileNotFoundError:
        st.warning(f"File not found: {path} ({name}) — this feature will be disabled.")
        return None
    except Exception as e:
        st.error(f"Error loading {path} ({name}): {e}")
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
    result = []
    i = 0
    while i < len(tokens):
        found = False
        for compound in compounds:
            compound_words = compound.split('_')
            if i + len(compound_words) <= len(tokens):
                match = True
                for j in range(len(compound_words)):
                    if tokens[i + j] != compound_words[j]:
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

    cleaned_review_pos = []
    for word, tag in pos_tagged:
        if word.lower() in sentiment_preserve or tag.startswith('JJ') or tag.startswith('RB'):
            cleaned_review_pos.append(word.lower())
        else:
            cleaned_review_pos.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    cleaned_review_with_compounds = extract_compound_terms(cleaned_review_pos, compound_list)

    stop_words = set(stopwords.words('english'))
    sentiment_stopwords = {'very', 'really', 'quite', 'rather', 'too', 'so', 'not', 'no', 'never'}
    filtered_stopwords = stop_words - sentiment_stopwords

    return ' '.join([w for w in cleaned_review_with_compounds if w.lower() not in filtered_stopwords])

# ----------------------------
# Load comparison data (cached)
# ----------------------------
@st.cache_resource
def load_comparison_data(file_path):
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        st.warning(f"Comparison data not loaded: {e}")
        return None

# ----------------------------
# Load models & data (cached)
# ----------------------------
@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}

    # ------------- Transformer from HF ----------------
    # CHANGE THIS to your Hugging Face repo id:
    HF_MODEL_REPO = "KelvinIsDaBest/sentiment_model"  # <-- replace with your HF repo id

    # Look for Hugging Face token in Streamlit secrets or env
    hf_token = None
    try:
        if "hf_token" in st.secrets:
            hf_token = st.secrets["hf_token"]
    except Exception:
        # st.secrets may be empty or not present
        pass
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")

    try:
        # pass token if available (works for private repos)
        if hf_token:
            models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(HF_MODEL_REPO, use_auth_token=hf_token)
            models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO, use_auth_token=hf_token)
        else:
            models['transformer_tokenizer'] = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
            models['transformer_model'] = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)

        # create pipeline
        models['sentiment_pipeline'] = pipeline("sentiment-analysis", model=models['transformer_model'], tokenizer=models['transformer_tokenizer'])
    except Exception as e:
        st.warning(f"Could not load Transformer model from Hugging Face: {e}")
        models['sentiment_pipeline'] = None

    # ------------- Classic models saved in repo (joblib) ----------------
    # Load these with safe loader (they might be missing)
    models['lr_std_tfidf'] = safe_load_joblib('logistic_regression_model_for_std_tfidf_baseline.joblib', 'LR std tfidf')
    models['nb_std_tfidf'] = safe_load_joblib('naive_bayes_model_for_std_tfidf_baseline.joblib', 'NB std tfidf')
    models['svm_std_tfidf'] = safe_load_joblib('svm_model_for_std_tfidf_baseline.joblib', 'SVM std tfidf')
    models['tfidf_vectorizer_std'] = safe_load_joblib('tfidf_vectorizer_for_std_tfidf_baseline.joblib', 'TFIDF std')

    models['lr_pos_driven'] = safe_load_joblib('logistic_regression_model_for_pos_driven.joblib', 'LR pos-driven')
    models['nb_pos_driven'] = safe_load_joblib('naive_bayes_model_for_pos_driven.joblib', 'NB pos-driven')
    models['svm_pos_driven'] = safe_load_joblib('svm_model_for_pos_driven.joblib', 'SVM pos-driven')
    models['tfidf_vectorizer_pos'] = safe_load_joblib('tfidf_vectorizer_for_pos_driven.joblib', 'TFIDF pos')

    # compound list
    models['compound_list'] = safe_load_joblib('compound_list.joblib', 'compound_list') or []

    # comparison data
    data['comparison_df'] = load_comparison_data('comparison_df.pkl')

    # confusion matrix data (optional)
    data['y_test_std'] = safe_load_joblib('y_test_for_std_tfidf_baseline.joblib', 'y_test_std')
    data['lr_pred_std'] = safe_load_joblib('lr_predictions_for_std_tfidf_baseline.joblib', 'lr_pred_std')
    data['nb_pred_std'] = safe_load_joblib('nb_predictions_for_std_tfidf_baseline.joblib', 'nb_pred_std')
    data['svm_pred_std'] = safe_load_joblib('svm_predictions_for_std_tfidf_baseline.joblib', 'svm_pred_std')

    data['y_test_pos'] = safe_load_joblib('y_test_for_pos_driven.joblib', 'y_test_pos')
    data['lr_pred_pos'] = safe_load_joblib('lr_predictions_for_pos_driven.joblib', 'lr_pred_pos')
    data['nb_pred_pos'] = safe_load_joblib('nb_predictions_for_pos_driven.joblib', 'nb_pred_pos')
    data['svm_pred_pos'] = safe_load_joblib('svm_predictions_for_pos_driven.joblib', 'svm_pred_pos')

    transformer_true = safe_load_joblib('true_labels.joblib', 'true_labels')
    transformer_pred = safe_load_joblib('predicted_labels.joblib', 'predicted_labels')
    if transformer_true is not None and transformer_pred is not None:
        label_map = {0: 'negative', 1: 'positive'}
        data['true_labels_transformer_str'] = [label_map.get(l, 'unknown') for l in transformer_true]
        data['predicted_labels_transformer_str'] = [label_map.get(l, 'unknown') for l in transformer_pred]
    else:
        data['true_labels_transformer_str'] = None
        data['predicted_labels_transformer_str'] = None

    return models, data

# ----------------------------
# Run app
# ----------------------------
models, data = load_models_and_data()

st.title("Large-Scale Movie Reviews Sentiment Analysis")

# Model Performance Comparison
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
                ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + bar_width * (len(metrics)-1)/2)
    ax.set_xticklabels(plot_df.index, rotation=45, ha='right')
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Comparison data not available.")

# Confusion Matrices
st.subheader("Confusion Matrices")
classes = ['negative', 'positive']
if data and data.get('y_test_std') is not None and data.get('lr_pred_std') is not None:
    # Standard TF-IDF
    st.write("### Standard TF-IDF")
    col_std_lr, col_std_nb, col_std_svm = st.columns(3)
    with col_std_lr:
        cm = confusion_matrix(data['y_test_std'], data['lr_pred_std'], labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('LR (Standard TF-IDF)')
        st.pyplot(fig)
    with col_std_nb:
        cm = confusion_matrix(data['y_test_std'], data['nb_pred_std'], labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('Naive Bayes (Standard TF-IDF)')
        st.pyplot(fig)
    with col_std_svm:
        cm = confusion_matrix(data['y_test_std'], data['svm_pred_std'], labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('SVM (Standard TF-IDF)')
        st.pyplot(fig)
else:
    st.warning("Standard TF-IDF confusion matrix data missing or incomplete.")

if data and data.get('y_test_pos') is not None and data.get('lr_pred_pos') is not None:
    st.write("### POS-Driven")
    col_pos_lr, col_pos_nb, col_pos_svm = st.columns(3)
    with col_pos_lr:
        cm = confusion_matrix(data['y_test_pos'], data['lr_pred_pos'], labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('LR (POS-Driven)')
        st.pyplot(fig)
    with col_pos_nb:
        cm = confusion_matrix(data['y_test_pos'], data['nb_pred_pos'], labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('Naive Bayes (POS-Driven)')
        st.pyplot(fig)
    with col_pos_svm:
        cm = confusion_matrix(data['y_test_pos'], data['svm_pred_pos'], labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title('SVM (POS-Driven)')
        st.pyplot(fig)
else:
    st.warning("POS-Driven confusion matrix data missing or incomplete.")

if data.get('true_labels_transformer_str') is not None and data.get('predicted_labels_transformer_str') is not None:
    st.write("### Transformer")
    cm = confusion_matrix(data['true_labels_transformer_str'], data['predicted_labels_transformer_str'], labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Transformer')
    st.pyplot(fig)
else:
    st.warning("Transformer confusion matrix data missing or incomplete.")

# Predict Movie Review
st.subheader("Predict Movie Review")
user_input = st.text_area("Enter your movie review here:", height=200)

if st.button("Analyze Sentiment"):
    if not user_input:
        st.warning("Please enter a movie review to analyze.")
    else:
        word_count = count_meaningful_words(user_input.strip())
        if word_count >= 5:
            st.success(f"Input quality: 🟢 High ({word_count} meaningful words)")
        elif word_count >= 3:
            st.info(f"Input quality: 🟡 Medium ({word_count} meaningful words)")
        else:
            st.warning(f"Input quality: 🔴 Low ({word_count} meaningful words) - Results may be less accurate")

        results = {}

        # Standard TF-IDF predictions (if models loaded)
        if all(models.get(k) is not None for k in ['lr_std_tfidf', 'nb_std_tfidf', 'svm_std_tfidf', 'tfidf_vectorizer_std']):
            try:
                processed_std = preprocess_review_standard_improved(user_input)
                features_std = models['tfidf_vectorizer_std'].transform([processed_std])

                lr_pred_std = models['lr_std_tfidf'].predict(features_std)[0]
                lr_prob_std = max(models['lr_std_tfidf'].predict_proba(features_std)[0])
                results['Standard TF-IDF + Logistic Regression'] = {'prediction': lr_pred_std, 'confidence': lr_prob_std}

                nb_pred_std = models['nb_std_tfidf'].predict(features_std)[0]
                nb_prob_std = max(models['nb_std_tfidf'].predict_proba(features_std)[0])
                results['Standard TF-IDF + Naive Bayes'] = {'prediction': nb_pred_std, 'confidence': nb_prob_std}

                svm_pred_std = models['svm_std_tfidf'].predict(features_std)[0]
                svm_score_std = abs(models['svm_std_tfidf'].decision_function(features_std)[0])
                results['Standard TF-IDF + SVM'] = {'prediction': svm_pred_std, 'confidence': svm_score_std}
            except Exception as e:
                st.error(f"Error during Standard TF-IDF predictions: {e}")
        else:
            st.warning("Standard TF-IDF models not fully available. Skipping those predictions.")
            results['Standard TF-IDF + Logistic Regression'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
            results['Standard TF-IDF + Naive Bayes'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
            results['Standard TF-IDF + SVM'] = {'prediction': 'Not Loaded', 'confidence': 0.0}

        # POS-Driven predictions
        if all(models.get(k) is not None for k in ['lr_pos_driven', 'nb_pos_driven', 'svm_pos_driven', 'tfidf_vectorizer_pos']):
            try:
                processed_pos = preprocess_review_pos_driven_improved(user_input, models.get('compound_list', []))
                features_pos = models['tfidf_vectorizer_pos'].transform([processed_pos])

                lr_pred_pos = models['lr_pos_driven'].predict(features_pos)[0]
                lr_prob_pos = max(models['lr_pos_driven'].predict_proba(features_pos)[0])
                results['POS-Driven + Logistic Regression'] = {'prediction': lr_pred_pos, 'confidence': lr_prob_pos}

                nb_pred_pos = models['nb_pos_driven'].predict(features_pos)[0]
                nb_prob_pos = max(models['nb_pos_driven'].predict_proba(features_pos)[0])
                results['POS-Driven + Naive Bayes'] = {'prediction': nb_pred_pos, 'confidence': nb_prob_pos}

                svm_pred_pos = models['svm_pos_driven'].predict(features_pos)[0]
                svm_score_pos = abs(models['svm_pos_driven'].decision_function(features_pos)[0])
                results['POS-Driven + SVM'] = {'prediction': svm_pred_pos, 'confidence': svm_score_pos}
            except Exception as e:
                st.error(f"Error during POS-Driven predictions: {e}")
        else:
            st.warning("POS-Driven models not fully available. Skipping those predictions.")
            results['POS-Driven + Logistic Regression'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
            results['POS-Driven + Naive Bayes'] = {'prediction': 'Not Loaded', 'confidence': 0.0}
            results['POS-Driven + SVM'] = {'prediction': 'Not Loaded', 'confidence': 0.0}

        # Transformer prediction
        if models.get('sentiment_pipeline') is not None:
            try:
                # pipeline accepts a string; ensure small input or enable truncation via tokenizer if needed
                transformer_result = models['sentiment_pipeline'](user_input, truncation=True)[0]
                label = transformer_result.get('label', '')
                # If label is like 'LABEL_1' -> map to positive/negative
                if label.startswith("LABEL_"):
                    label_id = label.replace("LABEL_", "")
                    try:
                        label_str = "positive" if int(label_id) == 1 else "negative"
                    except Exception:
                        label_str = label
                else:
                    label_str = label.lower()
                results['Transformer'] = {'prediction': label_str, 'confidence': transformer_result.get('score', 0.0)}
            except Exception as e:
                st.error(f"Error during Transformer prediction: {e}")
                results['Transformer'] = {'prediction': 'Error', 'confidence': 0.0}
        else:
            st.warning("Transformer model pipeline not loaded. Transformer predictions unavailable.")
            results['Transformer'] = {'prediction': 'Not Loaded', 'confidence': 0.0}

        # Display individual results
        st.write("---")
        st.subheader("Individual Model Predictions:")
        for model_name, result in results.items():
            sentiment = str(result['prediction']).upper()
            conf = result['confidence'] if result['confidence'] is not None else 0.0
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

        # Overall summary
        positive_count = sum(1 for r in results.values() if str(r['prediction']).lower() == 'positive')
        negative_count = sum(1 for r in results.values() if str(r['prediction']).lower() == 'negative')
        total_predictions = positive_count + negative_count
        st.write("---")
        st.subheader("Overall Summary:")
        st.write(f"Positive predictions: {positive_count}/{total_predictions}")
        st.write(f"Negative predictions: {negative_count}/{total_predictions}")
        if positive_count > negative_count:
            st.markdown(f"Overall sentiment: <span style='color:green'>**POSITIVE**</span>", unsafe_allow_html=True)
        elif negative_count > positive_count:
            st.markdown(f"Overall sentiment: <span style='color:red'>**NEGATIVE**</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"Overall sentiment: **MIXED (TIE)**", unsafe_allow_html=True)
