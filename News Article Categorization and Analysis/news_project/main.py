import nltk
nltk.data.path.append(r"C:\Users\ZEYNEP\AppData\Roaming\nltk_data")

from data_loader import load_data
from preprocessing import preprocess
from feature_extraction import extract_features
from classifier import train_model
from evaluate import evaluate

print("Loading data...")
train_texts, train_labels, test_texts, test_labels = load_data()

# Original raw data sample (we are saving a sample)
with open("raw_sample.txt", "w", encoding="utf-8") as f:
    f.write(train_texts[0])

print("Pre-processing is in progress...")
train_texts_processed = []
test_texts_processed = []

for text in train_texts:
    processed = preprocess(text)
    train_texts_processed.append(processed)

# Write first preprocessed sample to file
with open("cleaned_normalized_sample.txt", "w", encoding="utf-8") as f:
    f.write(train_texts_processed[0])

for text in test_texts:
    processed = preprocess(text)
    test_texts_processed.append(processed)

# Filter empty documents
filtered_train = [(text, label) for text, label in zip(train_texts_processed, train_labels) if text.strip() != ""]
filtered_test = [(text, label) for text, label in zip(test_texts_processed, test_labels) if text.strip() != ""]

train_texts, train_labels = zip(*filtered_train)
test_texts, test_labels = zip(*filtered_test)

print("Feature is being removed...")
X_train, X_test = extract_features(train_texts, test_texts)

print("The model is being trained...")
model = train_model(X_train, train_labels, model_type="naive_bayes")  

print("The model is being tested...")

# Write evaluation outputs to file as well
import sys
import io

# redirect stdout to file
with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    evaluate(model, X_test, test_labels)

# Reset stdout
sys.stdout = sys.__stdout__

# Press terminal too
evaluate(model, X_test, test_labels)

print("All operations are completed. Outputs are written to files.")

