from sklearn.tree import DecisionTreeClassifier

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
