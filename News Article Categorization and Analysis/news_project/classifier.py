from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def train_model(X, y, model_type="naive_bayes"):
    if model_type == "naive_bayes":
        model = MultinomialNB()
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "mlp":
        model = MLPClassifier(max_iter=300)
    else:
        raise ValueError("Unsupported model type")
    model.fit(X, y)
    return model



