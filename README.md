# Text Classification for Mental Health Detection

A multi-class NLP pipeline that classifies mental health status from textual statements. The project covers the full workflow from exploratory analysis through model training, evaluation, and interpretability (SHAP).

## Problem Statement

Given a short text statement written by an individual, predict which of seven mental health categories it falls into:

| Category | Description |
|---|---|
| Normal | No specific mental health concern indicated |
| Anxiety | Expressions of worry, nervousness, or unease |
| Depression | Persistent sadness or loss of interest |
| Suicidal | Thoughts related to self-harm |
| Stress | Mental or emotional strain |
| Bipolar | Mood swing indicators |
| Personality Disorder | Patterns of behavior deviating from social norms |

## Dataset

The dataset contains approximately 53,000 text-label pairs (`statement`, `status`) sourced from Kaggle ("Sentiment Analysis for Mental Health"). Class distribution is imbalanced, which the pipeline addresses via upsampling before training.

## Pipeline

### 1. Exploratory Data Analysis

Count plots, pie charts, and a correlation heatmap of engineered numerical features (`num_of_characters`, `num_of_words`, `num_of_sentences`) against the encoded target variable.

### 2. Preprocessing

- Drop rows with missing `statement` values (362 of 53,043).
- **Resample** minority classes to match the majority class count (16,343 per category) using `sklearn.utils.resample`.
- **Text normalization**: lowercase, remove non-alphanumeric characters, apply Porter Stemmer.
- **Vectorization**: TF-IDF with unigrams and bigrams (`max_features=5000`).

### 3. Models

All models are trained on an 80/20 train-test split with `GridSearchCV` for hyperparameter tuning (3-fold CV, weighted F1 scoring).

| Model | Architecture / Config | Test Accuracy |
|---|---|---|
| Logistic Regression | `C=10`, `solver='saga'` | 91.6% |
| Random Forest | 100 trees, default depth | 95.4% |
| Neural Network | Dense(128) > Dropout(0.2) > Dense(64) > Dropout(0.2) > Softmax, Adam lr=0.001, 20 epochs | 93.3% |

### 4. Evaluation and Interpretability

Each model is evaluated with classification reports (per-class precision, recall, F1), confusion matrices, and AUC-ROC curves (one-vs-rest). The notebook also includes:

- **Logistic Regression coefficients**: top weighted features per class.
- **Random Forest tree visualization**: `plot_tree` and `export_text` on a single estimator.
- **SHAP values**: `LinearExplainer` on the Logistic Regression model to surface feature importance per class.

### 5. Inference

The best model (Random Forest) and fitted TF-IDF vectorizer are serialized with `joblib`. A lightweight `predi()` function demonstrates single-statement classification.

## Requirements

- Python 3.x
- Google Colab (the notebook mounts Google Drive for dataset access)

### Python packages

```
pandas, numpy, scikit-learn, imbalanced-learn, tensorflow (Keras), nltk, shap, seaborn, matplotlib, joblib
```

Install with:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow nltk shap seaborn matplotlib joblib
```

NLTK data downloads (handled in the notebook):

```python
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

## Usage

1. Upload the dataset CSV ("Sentiment Analysis for Mental Health.csv") to your Google Drive.
2. Open `DS675_GroupProject_Final.ipynb` in Google Colab.
3. Update the `read_csv` path in the data loading cell to match your Drive location.
4. Run all cells. The notebook will mount Drive, load data, preprocess, train all three models, and produce evaluation plots.

For standalone inference after training:

```python
import joblib

vec = joblib.load('vectorizer.pkl')
model = joblib.load('model_rf.pkl')

text = "i feel so anxious about everything"
text_processed = preprocessing(text)       # apply the same stemming pipeline
prediction = model.predict(vec.transform([text_processed]))[0]
print(labels[prediction])
```

## Project Structure

```
nlp-mental-health-detection/
  DS675_GroupProject_Final.ipynb   # Full pipeline notebook (EDA through inference)
  README.md                       # This file
```

After execution the notebook also produces:

- `DS636_Project_Ananta Dian Pradipta.csv` (cleaned dataset export)
- `vectorizer.pkl` (fitted TF-IDF vectorizer)
- `model_rf.pkl` (trained Random Forest model)
- `x_test.csv` (test set with predicted labels)

## Authors

Ananta Dian Pradipta and collaborators

## License

This project is provided for educational purposes as part of NJIT DS675 coursework.
