# Utility functions

from sklearn.model_selection import train_test_split

def split_data(X, y, method='time', test_size=0.2, datetime_cols=None):
    if method == 'random' or not datetime_cols:
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Time-based split
    date_col = datetime_cols[0]  # Ambil kolom datetime pertama
    df = X.copy()
    df['__target__'] = y
    df = df.sort_values(by=date_col)

    cutoff = int((1 - test_size) * len(df))
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:]

    X_train = train.drop(columns='__target__')
    y_train = train['__target__']
    X_test = test.drop(columns='__target__')
    y_test = test['__target__']

    return X_train, X_test, y_train, y_test
