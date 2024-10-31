from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

class ReturnPredictor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {'MSE': mse, 'R2': r2}
    
    def cross_validate(self, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(self.model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.mean()

# Esempio di utilizzo
if __name__ == '__main__':
    # Carica e preprocess i dati
    preprocessor = TradingPreprocessor(...)
    df_preprocessed, stationarity = preprocessor.preprocess_data(df, target_column='return', prediction_type='regression')
    
    # Dividi i dati in train, validation e test set
    train_size = int(len(df_preprocessed) * 0.8)
    train_data = df_preprocessed[:train_size]
    test_data = df_preprocessed[train_size:]
    
    X_train, y_train = train_data.drop(columns=['return']), train_data['return']
    X_test, y_test = test_data.drop(columns=['return']), test_data['return']
    
    # Addestra e valuta il modello
    predictor = ReturnPredictor()
    predictor.fit(X_train, y_train)
    
    train_scores = predictor.evaluate(X_train, y_train)
    test_scores = predictor.evaluate(X_test, y_test)
    
    print(f"Train scores: {train_scores}")
    print(f"Test scores: {test_scores}")
    
    # Cross-validation
    cv_score = predictor.cross_validate(X_train, y_train)
    print(f"Cross-validation MSE: {cv_score:.4f}")