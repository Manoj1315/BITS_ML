from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model
