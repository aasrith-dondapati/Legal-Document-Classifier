import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
import re
import string

# --- 1. Simulated Data Generation (REPLACE WITH YOUR REAL DATA) ---
# IMPORTANT: For your actual use case, replace this section with loading your 5000+ legal documents
# and their corresponding labels. Ensure your 'documents' list contains the text of each document,
# and your 'labels' list contains the category for each document (e.g., 'contract', 'nda', 'policy').

def generate_simulated_data(num_docs=500):
    documents = []
    labels = []

    # Keywords for each category (simplified for simulation)
    contract_keywords = [
        "agreement", "contract", "parties", "clause", "terms and conditions", "effective date",
        "governing law", "liabilities", "indemnification", "breach", "termination",
        "warranty", "payment terms", "delivery", "scope of work"
    ]
    nda_keywords = [
        "confidential", "disclosure", "non-disclosure", "proprietary information",
        "recipient", "disclosing party", "trade secrets", "injunctive relief",
        "permitted use", "return of materials", "exceptions to confidentiality"
    ]
    policy_keywords = [
        "policy", "procedure", "guidelines", "compliance", "employees", "safety",
        "privacy", "data protection", "code of conduct", "training", "reporting",
        "hr", "company standards", "employee handbook"
    ]

    import random

    for _ in range(num_docs // 3): # Generate roughly equal numbers for each class
        # Contract
        doc = " ".join(random.choices(contract_keywords, k=random.randint(5, 15)))
        doc += " This is a standard contract agreement between parties. " + \
               "It outlines terms and conditions including governing law."
        documents.append(doc)
        labels.append('contract')

        # NDA
        doc = " ".join(random.choices(nda_keywords, k=random.randint(5, 15)))
        doc += " This confidential disclosure agreement protects proprietary information. " + \
               "The recipient agrees to non-disclosure."
        documents.append(doc)
        labels.append('nda')

        # Policy
        doc = " ".join(random.choices(policy_keywords, k=random.randint(5, 15)))
        doc += " This company policy outlines guidelines for employees and compliance. " + \
               "It covers data protection and safety procedures."
        documents.append(doc)
        labels.append('policy')

    # Shuffle the data
    combined = list(zip(documents, labels))
    random.shuffle(combined)
    documents, labels = zip(*combined)

    print(f"Generated {len(documents)} simulated documents.")
    return list(documents), list(labels)

# --- Replace this with your actual data loading ---
# Example for loading from CSV:
# df = pd.read_csv('your_legal_documents.csv')
# documents = df['document_text_column'].tolist()
# labels = df['category_label_column'].tolist()
# --------------------------------------------------

# Use simulated data for demonstration
documents, labels = generate_simulated_data(num_docs=5000) # Aiming for 5000+ like your request

# --- 2. Data Preprocessing Function ---
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\n', ' ', text) # Replace newlines with spaces
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers (optional, depends on legal docs)
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace

    # Lemmatization (optional but recommended)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

print("Applying preprocessing to documents...")
documents = [preprocess_text(doc) for doc in documents]
print("Preprocessing complete.")

# --- 3. Data Splitting ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- 4. Building the Model Pipeline with TF-IDF and LinearSVC ---
# A Pipeline chains together multiple processing steps.
# This ensures that preprocessing (TF-IDF vectorization) is consistently applied
# to both training and new data.

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',      # Remove common English stop words
        max_df=0.85,               # Ignore terms that appear in more than 85% of documents
        min_df=5,                  # Ignore terms that appear in less than 5 documents
        ngram_range=(1, 2)         # Consider single words and two-word phrases (bigrams)
    )),
    ('clf', LinearSVC(random_state=42)) # Classifier: Linear Support Vector Classifier
])

print("Pipeline created. Starting Hyperparameter Tuning (GridSearchCV)...")

# --- 5. Hyperparameter Tuning using GridSearchCV ---
# This step finds the best combination of parameters for your TF-IDF and classifier.
# It can take a while depending on the dataset size and parameter grid.

param_grid = {
    'tfidf__max_df': [0.7, 0.8, 0.9],
    'tfidf__min_df': [3, 5, 7],
    'tfidf__ngram_range': [(1, 1), (1, 2)], # Try unigrams only, and unigrams+bigrams
    'clf__C': [0.5, 0.8, 1.0, 1.2], # Regularization strength for LinearSVC
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,            # 5-fold cross-validation
    n_jobs=-1,       # Use all available CPU cores
    verbose=2,       # Print progress
    scoring='accuracy' # Optimize for accuracy
)

grid_search.fit(X_train, y_train)

print("\nHyperparameter Tuning Complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# --- 6. Model Evaluation on Test Set ---
print("\nEvaluating the best model on the test set...")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 7. Visualize Confusion Matrix ---
print("\nGenerating Confusion Matrix...")
conf_mat = confusion_matrix(y_test, y_pred, labels=best_model.classes_) # Ensure labels order is consistent
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Legal Document Classification')
plt.show()

# --- 8. Save the Trained Model and TF-IDF Vectorizer ---
# Saving the entire pipeline is best practice as it includes both TF-IDF and the classifier.
print("\nSaving the trained model pipeline...")
model_filename = 'legal_document_classifier_pipeline.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved as '{model_filename}'")

# --- 9. Load Model and Make New Predictions ---
print("\n--- Testing Model Loading and New Predictions ---")
loaded_pipeline = joblib.load(model_filename)
print(f"Model loaded from '{model_filename}'")

# Example new documents for prediction
new_legal_docs = [
    "This Purchase Agreement specifies the terms of sale for goods. The parties agree to payment schedule and delivery.",
    "THIS NON-DISCLOSURE AGREEMENT protects confidential and proprietary information exchanged between the parties.",
    "Company Employee Handbook outlines the policy for vacation time, sick leave, and code of conduct for all employees.",
    "A confidential agreement details mutual obligations and non-compete clauses." # Mix of concepts
]

print("\nOriginal new documents:")
for i, doc in enumerate(new_legal_docs):
    print(f"Doc {i+1}: {doc[:70]}...") # Print first 70 chars

# Preprocess new documents using the same function used for training
preprocessed_new_docs = [preprocess_text(doc) for doc in new_legal_docs]

# Make predictions using the loaded pipeline (it handles TF-IDF transformation internally)
predictions = loaded_pipeline.predict(preprocessed_new_docs)

print("\nPredictions for new documents:")
for i, doc in enumerate(new_legal_docs):
    print(f"Document: '{doc[:70]}...' -> Predicted: {predictions[i]}")