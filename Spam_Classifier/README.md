# Machine Learning: Spam Classifier

## Overview
This repository details the implementation of a spam classifier built from scratch. The primary objective was to categorize emails as either spam (+1) or ham (0) using a Naive Bayes Classifier, with custom data preparation and feature extraction.

## Dataset
To address the problem of spam classification, two publicly available datasets were utilized and combined: "SpamAssasin" and "CEAS_08".

### Initial Dataset - SpamAssasin
* **Overview:** Contains emails labeled as spam (1) or ham (0), a foundational set for spam classification.
* **Class Imbalance:** Initially, this dataset showed a significant imbalance, with a larger proportion of ham emails, posing a risk of classifier bias.

### Additional Dataset - CEAS_08
* **Purpose:** Used to balance the class distribution by providing a larger volume of spam emails.
* **Sampling:** A sample of 2,800 spam emails from CEAS_08 was selected and merged with the SpamAssasin dataset to achieve a near-equal balance between spam and ham.

### Combining and Preprocessing Datasets
* **Concatenation:** Sampled CEAS_08 spam emails were appended to the SpamAssasin dataset.
* **Final Dataset Distribution:**
    * Total email count: 8609 emails
    * Spam: 4518 emails
    * Ham: 4091 emails
    * This setup achieved a near-equal balance, reducing bias and improving classifier generalization.
* **Email Text Creation:** Sender, receiver, subject, and body fields were combined for each email to create a complete text representation.
* **Column Dropping:** Irrelevant columns (e.g., date, URLs) were removed to focus on meaningful text data.

**Note:** The original datasets ("SpamAssasin" and "CEAS_08") are **not included** in this repository.

### Data Splitting and Organization
The combined dataset (8609 emails) was split into training and test sets (80% training, 20% testing).
* **Training Set ($X_{train}$):** 6887 emails
* **Test Set ($X_{test}$):** 1722 emails

For compatibility with the classifier, emails in both training and test sets were saved as individual `.txt` files within a specific folder structure:
* `train/spam/`: For spam training emails.
* `train/ham/`: For ham training emails.
* `test/spam/`: For spam testing emails.
* `test/ham/`: For ham testing emails.
Each email file was named sequentially (e.g., `email_0.txt`).

### Supplementary Dataset
An additional supplementary dataset (not included in this repository) was used exclusively for evaluating the classifier's performance on new, external, unseen emails. This dataset also had a balanced distribution and its emails were saved as individual `.txt` files within a structured directory, similar to the primary test set.

## Classifier Implementation: Naive Bayes

### Overview
A Naive Bayes Classifier, specifically the Multinomial Naive Bayes version, was implemented from scratch. This algorithm is highly effective for text classification tasks due to its ability to handle word frequency data.

### Feature Extraction
* **Tokenization:** Email texts were preprocessed by converting to lowercase, removing punctuation and special characters, and splitting into individual words using regular expressions.
* **Bag of Words:** Each email was represented by a word frequency dictionary. This representation was used to calculate the likelihood of each word belonging to spam or ham categories.

### Prior Probabilities
* **Spam Prior:** Calculated as the ratio of spam emails to total emails in the training set.
* **Ham Prior:** Calculated as the ratio of ham emails to total emails in the training set.

### Likelihood Calculation and Smoothing
For each word, the likelihoods were computed based on its occurrences in spam and ham emails.
* **Laplace Smoothing:** Applied to handle words that might appear only in one category, improving model robustness.
    * $P(\text{word } | \text{ spam}) = \frac{\text{count(word in spam)} + 1}{\text{total spam words } + \text{ vocab size}}$
    * $P(\text{word } | \text{ ham}) = \frac{\text{count(word in ham)} + 1}{\text{total ham words } + \text{ vocab size}}$
    Where `vocab size` is the unique word count across all emails.

### Prediction
* **Log Probabilities:** To prevent numerical underflow with very small probabilities, log probabilities were used for calculations.
* **Final Classification:** Each test email's likelihood of being spam or ham was compared based on these log probabilities to assign its final label (+1 for spam, 0 for ham).

## Results and Evaluation

### Main Test Set Performance
The classifier was evaluated on the test set derived from the combined SpamAssasin and CEAS_08 datasets.
* **Spam Emails (898 in test set):** 855 correctly classified (95% accuracy).
* **Ham Emails (824 in test set):** 821 correctly classified (100% accuracy).

### Supplementary Dataset Performance
The classifier's generalization capabilities were further tested on an external, unseen supplementary dataset.
* **Spam Emails (800 in supplementary set):** 617 correctly classified (77% accuracy).
* **Ham Emails (800 in supplementary set):** 796 correctly classified (99% accuracy).

### Conclusion
The Naive Bayes classifier demonstrated high accuracy on both the main and supplementary datasets, effectively distinguishing spam from ham. The strategy of combining datasets to balance class distribution significantly improved performance. Laplace smoothing played a crucial role in enhancing the model's robustness by addressing the issue of unseen words.

## Libraries Used
* Python's built-in functionalities (for string processing, file operations)
* Pandas
* Matplotlib

## How to Run
To run the code:
1.  Ensure you have Python installed.
2.  Place your training emails in the expected `train/spam/` and `train/ham/` subdirectories within your project folder.
3.  Place the test emails (e.g., `email0.txt`, `email1.txt`, etc.) in a `test/` folder within the same directory as your main code file.
4.  Execute the `spam_classifier.ipynb` file. The classifier is designed to automatically read emails from the `test/` folder and output predictions.