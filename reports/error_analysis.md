# Error analysis

All numbers are on the test split (147 emails, 73 spam) and come from
`python -m src.models.compare`; per-email scores are in `test_predictions.csv`.

| Model | Spam precision | Spam recall | Spam F1 | False positives | False negatives |
|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.986 | 0.945 | 0.965 | 1 | 4 |
| TF-IDF + Logistic Regression | 1.000 | 0.973 | 0.986 | 0 | 2 |
| TF-IDF + linear SVM | 1.000 | 0.973 | 0.986 | 0 | 2 |
| fastText + Logistic Regression | 0.932 | 0.932 | 0.932 | 5 | 5 |
| Romanian BERT | 1.000 | 0.973 | 0.986 | 0 | 2 |

One email moves F1 by about 1.4 points, so logistic regression, the SVM and
BERT are tied here: they make the same two mistakes. BERT is also not fully
reproducible on GPU; two runs with the same config gave 0.986 and 0.993 F1 on
validation.

## Emails every model gets wrong

**"Stimate client, contul dumneavoastră necesită atenție"** (spam, phishing).
Asks for an ID copy and a selfie for a yearly KYC check, with a link. There is
no urgency, no threat, no prize and the tone is the one of a real bank notice.
LIME (`figures/lime_example_1.png`) shows the polite opening words ("Bună",
"Stimate", "client", "mulțumim") pulling both models towards ham, more than the
link and "accesați" pull towards spam. In this dataset formal greetings occur
mostly in legitimate notices.

**"Te rog, deschide atașamentul"** (spam, fake invoice). The only spam signal
is the attachment name `Comanda_noiembrie.pdf.exe`; the text itself reads like
an ordinary business email. No training email has an `.exe` attachment (the
only other one is in validation), and after lemmatization the file name is
reduced to fragments such as `_noiembriu`. Neither model puts `.exe` among its
top words (`figures/lime_example_3.png`).

## Mistakes of the weaker models

- **fastText** has 5 false positives. Four are legitimate notices about
  payments or orders (a flight refund, a hosting renewal, an easybox delivery,
  a card payment); the fifth is a work email about a report. Averaging the word
  vectors keeps the topic and loses the words that separate a real notice from
  a phishing one.
- **Naive Bayes** misses, besides the two emails above, two short scams: a
  "your son lost his phone" message and a paid party invitation. Both are a few
  lines long and share little vocabulary with the rest of the spam.
- The card payment notice from BancaNova (ham) fools Naive Bayes and fastText.
  LIME (`figures/lime_example_2.png`) shows the opposite problem for the other
  models: the name "BancaNova" pushes towards ham. The legitimate emails use
  invented company names, while about 18% of the spam mentions real brands
  (ING, PayPal, eMAG), so the brand name works as a label shortcut.

## Limitations

- The dataset is generated, so spam and ham are more stereotyped than in a real
  inbox, and the scores above are optimistic.
- The brand shortcut above means a real email from ING or eMAG is more likely
  to be flagged as spam than this test set suggests.
- In a few ham emails without a signature, the link added to balance the URL
  counts ended up above the body instead of before the sign-off.
