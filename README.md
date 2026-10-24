# spam-detection-ro

Spam detection for emails written in Romanian, built for the Natural Language Processing and Text Mining course. It compares TF-IDF baselines, hand-crafted features, fastText embeddings and a fine-tuned Romanian BERT on the same data, and ships a small Streamlit demo.

## Data

There is no public corpus of Romanian spam emails, so the dataset was generated for this project: 974 emails (491 ham, 483 spam) in 23 categories, written directly in Romanian, in `data/raw/generated/`. Spam covers bank and courier phishing, fake invoices, prizes, investment and job scams; ham includes legitimate bank notices, order confirmations and newsletters, which share vocabulary with phishing. About 5% of the emails have no diacritics and some spam uses obfuscations such as `c4știg`.

The split is stratified by category, 70/15/15: 681 train, 146 validation, 147 test.

Because the emails are generated, they are more uniform than a real inbox, and the scores below should be read as an upper bound.

## Pipeline

1. **Cleaning** (`src/preprocessing/cleaning.py`): HTML and email headers are stripped, URLs, email addresses, money amounts and numbers become placeholder tokens, and cedilla diacritics (ş, ţ) are mapped to comma-below (ș, ț).
2. **Normalization** (`src/preprocessing/nlp.py`): lemmatization with Stanza through `spacy_stanza`, or Snowball stemming, plus stopword removal. Stanza output is cached.
3. **TF-IDF baselines** (`src/models/baselines.py`): word 1-2 grams and character 3-5 grams with Naive Bayes, logistic regression and a linear SVM. Regularization is chosen by 5-fold CV on the train split.
4. **Hand-crafted features and ablation** (`src/features/handcrafted.py`, `src/models/ablation.py`): spaCy NER counts, a Matcher with Romanian phishing phrases, SentiWordNet scores from RoWordNet, surface statistics (uppercase ratio, links, exclamation marks) and the share of words missing from RoWordNet.
5. **fastText** (`src/models/fasttext_lr.py`): averaged `cc.ro.300` vectors with logistic regression.
6. **Romanian BERT** (`src/models/bert.py`): `dumitrescustefan/bert-base-romanian-cased-v1` fine-tuned for 5 epochs, chosen on validation F1.
7. **Evaluation** (`src/models/compare.py`, `src/models/explain.py`): one comparison on the test split, per-email predictions and LIME explanations.

## Results

Test split, 147 emails, 73 of them spam. Spam is the positive class; a false positive is a legitimate email flagged as spam.

| Model | Spam precision | Spam recall | Spam F1 | False positives | False negatives |
|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.986 | 0.945 | 0.965 | 1 | 4 |
| TF-IDF + logistic regression | 1.000 | 0.973 | 0.986 | 0 | 2 |
| TF-IDF + linear SVM | 1.000 | 0.973 | 0.986 | 0 | 2 |
| fastText + logistic regression | 0.932 | 0.932 | 0.932 | 5 | 5 |
| Romanian BERT | 1.000 | 0.973 | 0.986 | 0 | 2 |

One email changes F1 by about 1.4 points, so logistic regression, the SVM and BERT are tied: they miss the same two spam emails, a polite KYC phishing message and an email whose only red flag is a `.pdf.exe` attachment. BERT runs on GPU are not fully deterministic; two runs with the same settings gave 0.986 and 0.993 validation F1.

Lemmas and stems give the same test F1 for logistic regression and the SVM (0.986). Naive Bayes is slightly better on lemmas (0.965 against 0.959).

Ablation, 5-fold CV on train+validation with logistic regression (C=100), spam F1:

| Features | Mean | Std |
|---|---|---|
| TF-IDF | 0.983 | 0.008 |
| + NER | 0.983 | 0.011 |
| + phishing phrases | 0.979 | 0.008 |
| + sentiment | 0.978 | 0.008 |
| + surface statistics | 0.975 | 0.011 |
| + out-of-vocabulary ratio | 0.983 | 0.008 |
| + all | 0.977 | 0.008 |
| hand-crafted only | 0.830 | 0.023 |

No group improves on TF-IDF alone by more than the fold-to-fold variation. The hand-crafted features carry signal on their own, but the n-grams already capture it. Surface statistics hurt slightly: the link count pushes legitimate emails with links towards spam.

The misclassified emails and the LIME explanations are discussed in [reports/error_analysis.md](reports/error_analysis.md). It also covers a bias in the data: legitimate emails use invented company names while some spam mentions real brands. All figures are in `reports/figures/`.

## Setup

Python 3.11. Install PyTorch with CUDA first if you have an NVIDIA GPU, then the rest:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m spacy download ro_core_news_sm
```

`stanza` is pinned to 1.10.1: `spacy-stanza` asks for `stanza<1.7`, but those versions cannot load their models with torch 2.6 or later. Pip prints a warning about the pin, which can be ignored. The Stanza Romanian model and the BERT weights are downloaded on first use.

## Running

From the project root, in this order:

```bash
python -m src.data.split                 # stratified split into data/processed/
python -m src.preprocessing.preprocess   # cleaned, lemmatized and stemmed text
python -m src.models.baselines
python -m src.models.ablation
python -m src.models.fasttext_lr         # downloads cc.ro.300.vec.gz (1.2 GB) the first time
python -m src.models.bert
python -m src.models.compare
python -m src.models.explain
python -m pytest tests
```

On a local RTX 4060, BERT trains in under a minute. Without a GPU, use `notebooks/bert_colab.ipynb`, which clones the repository, rebuilds the split and runs the same training script.

## Demo

```bash
streamlit run app/app.py
```

Paste an email or load one of the two examples, pick a model and check it. The demo shows the spam probability and highlights the words that pushed the prediction towards spam or ham, using LIME for every model. The linear SVM is not offered because it does not produce a probability. The demo needs the models trained by the commands above.

## Structure

```
app/            Streamlit demo
config.yaml     paths and hyperparameters
data/raw/       generated emails (versioned); other data folders are rebuilt by the scripts
notebooks/      Colab notebook for BERT
reports/        result tables, error analysis and figures
src/data/       loading and splitting
src/features/   TF-IDF, hand-crafted features, fastText vectors
src/models/     training, evaluation, explanations and single-email prediction
src/preprocessing/  cleaning and Romanian normalization
tests/          pytest tests for cleaning, features, embeddings and the demo's prediction
```
