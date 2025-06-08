# Legal-Document-Classifier

Multi-Class classification model using Scikit-learn and TF-IDF

This project provides a robust framework for classifying legal documents (e.g., Contracts, NDAs, Policies) into predefined categories using ****TF-IDF** (Term Frequency-Inverse Document Frequency) and** **Scikit-learn** 's powerful machine learning algorithms. The goal is to achieve high accuracy in categorization, demonstrated with a target of 93% accuracy.

## Table of Contents

1. [Project Overview](https://www.google.com/search?q=%23project-overview)
2. [Features](https://www.google.com/search?q=%23features)
3. [Prerequisites](https://www.google.com/search?q=%23prerequisites)
4. [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
5. [Project Structure](https://www.google.com/search?q=%23project-structure)
6. [Step-by-Step Process](https://www.google.com/search?q=%23step-by-step-process)
   * [Step 1: Data Preparation](https://www.google.com/search?q=%23step-1-data-preparation)
   * [Step 2: Data Preprocessing](https://www.google.com/search?q=%23step-2-data-preprocessing)
   * [Step 3: Data Splitting](https://www.google.com/search?q=%23step-3-data-splitting)
   * [Step 4: TF-IDF Vectorization and Model Training](https://www.google.com/search?q=%23step-4-tf-idf-vectorization-and-model-training)
   * [Step 5: Hyperparameter Tuning](https://www.google.com/search?q=%23step-5-hyperparameter-tuning)
   * [Step 6: Model Evaluation](https://www.google.com/search?q=%23step-6-model-evaluation)
   * [Step 7: Model Saving](https://www.google.com/search?q=%23step-7-model-saving)
   * [Step 8: Making New Predictions](https://www.google.com/search?q=%23step-8-making-new-predictions)
7. [Customizing for Your Data](https://www.google.com/search?q=%23customizing-for-your-data)
8. [Troubleshooting and Tips](https://www.google.com/search?q=%23troubleshooting-and-tips)
9. [License](https://www.google.com/search?q=%23license)

---

## Project Overview

In the legal domain, efficiently categorizing documents is crucial. This project addresses the challenge of automatically classifying legal documents into specific categories like **`Contracts`,** **`NDAs`, and** **`Policies`. It leverages text feature extraction using** ****TF-IDF** and a** ****Linear Support Vector Classifier (LinearSVC)** from Scikit-learn, wrapped within a powerful** **Pipeline** for streamlined workflow and robust hyperparameter tuning.

## Features

* **Multi-Class Classification:** Categorizes documents into three or more distinct classes.
* **TF-IDF Feature Extraction:** Converts text into numerical representations, highlighting important words.
* **Scikit-learn Integration:** Utilizes a highly optimized and widely used Python machine learning library.
* **Pipeline for Workflow:** Chains preprocessing and classification steps for cleaner code and consistent application.
* **Hyperparameter Tuning:** Employs `GridSearchCV` with cross-validation to find the optimal model configuration for best performance.
* **Comprehensive Evaluation:** Provides accuracy, precision, recall, F1-score, and a confusion matrix to assess model performance.
* **Model Persistence:** Saves the trained model to disk for later use in making predictions on new data.
* **Text Preprocessing:** Includes steps for lowercasing, punctuation removal, and lemmatization.

---

## Prerequisites

Before running the project, ensure you have the following installed:

* **Python 3.7+**
* **pip** (Python package installer)

---

## Setup and Installation

1. **Clone the Repository (if applicable):**
   **Bash**

   ```
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install Required Libraries:
   Open your terminal or command prompt and run:
   **Bash**

   ```
   pip install scikit-learn matplotlib seaborn joblib nltk pandas
   ```
3. Download NLTK Data:
   The project uses NLTK for lemmatization. You need to download the wordnet corpus. You can do this by running a Python interpreter or adding the following lines to the top of your script and running it once:
   **Python**

   ```
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4') # Often needed with 'wordnet'
   ```

---

## Project Structure

* `legal_classifier.py`: The main Python script containing all the code for model training, evaluation, and prediction.
* `legal_document_classifier_pipeline.pkl`: (Generated after running the script) The saved machine learning pipeline.

---

### Step 1: Data Preparation

**Your Action:**

* **Crucially, replace the** **`generate_simulated_data` function** with code to load your *actual* legal documents and their labels.
* Your data should consist of:
  * A list or array of **document texts** (each document as a string).
  * A corresponding list or array of ****labels** (e.g.,** **`'contract'`,** **`'nda'`,** `'policy'`).

**Code Details:**

* The provided script includes a `generate_simulated_data` function for demonstration purposes. This creates a mock dataset of 5,000 documents across the three categories.
* A placeholder for loading data from a CSV (`pd.read_csv`) is commented out to guide you.

### Step 2: Data Preprocessing

**Your Action:**

* Ensure the `preprocess_text` function aligns with your data's characteristics. For legal documents, consider if numbers, very specific legal jargon, or headers should be treated differently.

**Code Details:**

* The `preprocess_text` function performs essential cleaning:
  * **Lowercasing:** Converts all text to lowercase to standardize words.
  * **Punctuation Removal:** Removes common punctuation.
  * **URL/HTML Removal:** Cleans up web links or HTML tags if present.
  * **Newline/Whitespace Normalization:** Ensures consistent spacing.
  * **Lemmatization:** Reduces words to their base form (e.g., "running," "runs" -> "run") using NLTK's `WordNetLemmatizer`. This helps group similar words.

### Step 3: Data Splitting

**Your Action:**

* No direct action needed here unless you want to change the test set size.

**Code Details:**

* The **`train_test_split` function divides your documents and labels into** ****training** (80%) and** **testing** (20%) sets.
* `test_size=0.2` means 20% of the data is held out for evaluating the model.
* `random_state=42` ensures reproducibility of the split.
* `stratify=labels` is vital for classification, ensuring that the proportion of each document category is maintained in both the training and testing sets.

### Step 4: TF-IDF Vectorization and Model Training

**Your Action:**

* Review the **`TfidfVectorizer` parameters. While** **`GridSearchCV` tunes some, you might manually adjust** **`stop_words` (e.g., create a custom list for specific legal terms) or** `ngram_range` if your preliminary results are poor.

**Code Details:**

* A **`Pipeline`** is created, chaining two steps:
  * **`TfidfVectorizer`:** Transforms text into numerical features.
    * `stop_words='english'`: Filters out common English words (e.g., "the", "is").
    * `max_df=0.85`: Ignores terms that appear in more than 85% of documents (too common to be discriminative).
    * `min_df=5`: Ignores terms that appear in fewer than 5 documents (too rare).
    * `ngram_range=(1, 2)`: Considers both single words (unigrams) and pairs of words (bigrams) as features. This captures more context.
  * **`LinearSVC`:** A robust and efficient Support Vector Machine classifier, well-suited for high-dimensional text data.

### Step 5: Hyperparameter Tuning

**Your Action:**

* If your accuracy isn't hitting 93%, you can ****expand the** **`param_grid`** with more values or even different TF-IDF parameters (`use_idf`,** **`smooth_idf`) or other classifiers like** `LogisticRegression`.

**Code Details:**

* `GridSearchCV` systematically searches through a predefined **`param_grid` of hyperparameter combinations for both** **`TfidfVectorizer` and** `LinearSVC`.
* **Cross-validation (`cv=5`):** The training data is split into 5 folds. The model is trained on 4 folds and validated on the remaining 1, repeated 5 times. This provides a more reliable estimate of performance.
* `n_jobs=-1`: Utilizes all available CPU cores for faster execution.
* The best combination of parameters and the best cross-validation accuracy are printed.

### Step 6: Model Evaluation

**Your Action:**

* Carefully analyze the ***Test Set Accuracy** ,* **Classification Report** , and **Confusion Matrix** . This is where you understand how well your model performs and where it struggles.

**Code Details:**

* The **`best_estimator_` from** **`GridSearchCV` (your optimized model) is used to make predictions on the unseen** `X_test` data.
* **`accuracy_score`** : Provides the overall percentage of correct predictions.
* **`classification_report`** : Details precision, recall, and F1-score for each class, giving a fine-grained view of performance.
* **`confusion_matrix`** : Visualizes where misclassifications occur (e.g., if Contracts are often mistaken for NDAs). This is crucial for identifying areas for improvement.

### Step 7: Model Saving

**Your Action:**

* No direct action, but remember the `model_filename`.

**Code Details:**

* The entire trained **`Pipeline` (including the** **`TfidfVectorizer` and** **`LinearSVC` with their optimal parameters) is saved to a** **`.pkl` file using** `joblib`. This allows you to load and use the model later without retraining.

### Step 8: Making New Predictions

**Your Action:**

* Modify the `new_legal_docs` list with your actual new documents you want to classify.

**Code Details:**

* The script demonstrates how to load the saved pipeline.
* New, unseen documents are passed through the same `preprocess_text` function.
* The `loaded_pipeline.predict()` method automatically performs the TF-IDF transformation and classification in one step, outputting the predicted category.

---

## Customizing for Your Data

1. Replace Simulated Data:
   Locate

   1. Simulated Data Generation (REPLACE WITH YOUR REAL DATA) section in legal_classifier.py.
      Comment out or delete the generate_simulated_data call and replace it with your data loading logic.
      Example (for CSV):
      **Python**

   ```
   import pandas as pd
   # ... other imports ...

   # Load your actual data
   df = pd.read_csv('your_legal_documents.csv') # Ensure this path is correct
   documents = df['document_text_column'].tolist() # Replace 'document_text_column' with your column name
   labels = df['category_label_column'].tolist()   # Replace 'category_label_column' with your label column name

   # Ensure your labels are consistent (e.g., all lowercase 'contract', 'nda', 'policy')
   # If not, you might add: labels = [label.lower() for label in labels]

   print(f"Loaded {len(documents)} actual documents.")
   ```
2. **Verify Labels:** Ensure the labels in your actual data (`'contract'`,** **`'nda'`,** **`'policy'`) exactly match what you expect. Inconsistencies will cause errors.
3. **Adjust Preprocessing:**

   * For very specific legal jargon, you might need to add custom stop words or use a more advanced tokenization technique.
   * Consider if numbers or dates in your legal documents carry semantic meaning for classification; if so, adjust the **`re.sub(r'\w*\d\w*', '', text)` line in** `preprocess_text`.

---

## Troubleshooting and Tips

* **"Missing NLTK Data" Error:** Ensure you've run **`nltk.download('wordnet')` and** `nltk.download('omw-1.4')`.
* **Low Accuracy:**
  * **Data Quality:** The single biggest factor. Ensure your documents are cleanly text-extracted and accurately labeled.
  * **More Data:** While 5,000 is a good start, more diverse, labeled data can often boost accuracy.
  * **Hyperparameter Tuning:** Experiment by expanding the **`param_grid` in** **`GridSearchCV` (e.g., try different** **`max_df`/`min_df` values, or** `ngram_range=(1,3)`).
  * **Examine Misclassifications:** Use the confusion matrix to identify patterns in errors. Manually review documents that were misclassified to understand *why* the model struggled. This can reveal issues with labeling or guide custom preprocessing.
* **Long Training Time:** `GridSearchCV` with many parameters and large datasets can take time. **`n_jobs=-1` helps by using all CPU cores. If it's still too slow, you might reduce the number of** **`cv` folds or the complexity of the** `param_grid`.
* **Memory Issues:** For extremely large datasets (millions of documents), consider using online learning algorithms (`SGDClassifier`) or libraries optimized for scale like **`Gensim` for word embeddings or** `Vowpal Wabbit`. However, for 5,000 documents, Scikit-learn should handle it efficiently.
