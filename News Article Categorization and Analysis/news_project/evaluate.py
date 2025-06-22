from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average='macro', zero_division=0))
    print("Recall (macro):", recall_score(y_test, y_pred, average='macro', zero_division=0))
    print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro', zero_division=0))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
