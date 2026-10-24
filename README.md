# spam-detection-ro

Spam detection for emails written in Romanian. Project for the Natural Language Processing and Text Mining course.

The plan is to start with classical models (Naive Bayes, Logistic Regression, SVM) on TF-IDF features, then fine-tune a Romanian BERT and compare the results. Preprocessing uses Romanian NLP tools like Stanza, spaCy and the NLTK Snowball stemmer.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download ro_core_news_sm
```

## Structure

- `data/` - datasets (not included in the repo)
- `src/` - preprocessing, features, models
- `notebooks/` - experiments
- `config.yaml` - paths and parameters