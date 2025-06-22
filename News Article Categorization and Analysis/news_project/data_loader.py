from nltk.corpus import reuters
import nltk
import random

def load_data():
    nltk.download("reuters")
    nltk.download("punkt")
    docs = reuters.fileids()
    train_docs = [doc for doc in docs if doc.startswith("train")]
    test_docs = [doc for doc in docs if doc.startswith("test")]

    def docs_to_texts(doc_ids):
        texts = []
        labels = []
        for doc_id in doc_ids:
            text = reuters.raw(doc_id)
            label = reuters.categories(doc_id)[0] if reuters.categories(doc_id) else "unknown"
            texts.append(text)
            labels.append(label)
        return texts, labels

    train_texts, train_labels = docs_to_texts(train_docs)
    test_texts, test_labels = docs_to_texts(test_docs)
    return train_texts, train_labels, test_texts, test_labels



