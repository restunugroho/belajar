# Core logic will go here


from .utils import split_data

def fit(self, X, y):
    # Cleaning
    X_clean, y_clean, datetime_cols = clean_data(X, y)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X_clean, y_clean,
        method=self.split_method,
        test_size=self.test_size,
        datetime_cols=datetime_cols
    )

    # Train & evaluate
    self.results, self.best_model = train_models(X_train, y_train, self.task)

    # Generate report
    generate_report(X_clean, y_clean, self.results, self.task,
                    self.test_size, self.split_method, len(X), len(X_train), len(X_test))

    return self
