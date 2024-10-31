from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

class EnsembleModelSelector:
    def __init__(self, models, cv=5, scoring='neg_mean_squared_error'):
        self.models = models
        self.cv = cv
        self.scoring = scoring
        
    def fit(self, X_train, y_train):
        self.best_model_ = None
        self.best_score_ = -np.inf
        
        for model in self.models:
            scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=self.scoring)
            mean_score = scores.mean()
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_model_ = model
                
        self.best_model_.fit(X_train, y_train)
        
    def predict(self, X):
        return self.best_model_.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return {'MSE': mse, 'R2': r2}

# Esempio di utilizzo
if __name__ == '__main__':
    # Definisci i modelli candidati
    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42),
        LinearRegression(),
        SVR(kernel='rbf', C=1.0, epsilon=0.1),
        Lasso(alpha=0.1),
        Ridge(alpha=1.0)
    ]
    
    # Carica e preprocess i dati
    preprocessor = TradingPreprocessor(...)
    df_preprocessed, stationarity = preprocessor.preprocess_data(df, target_column='return', prediction_type='regression')
    
    X, y = df_preprocessed.drop(columns=['return']), df_preprocessed['return']
    
    # Seleziona e addestra il miglior modello
    selector = EnsembleModelSelector(models, cv=5, scoring='neg_mean_squared_error')
    selector.fit(X, y)
    
    print(f"Best model: {selector.best_model_}")
    print(f"Best cross-validation score: {-selector.best_score_:.4f}")
    
    # Valuta il modello selezionato
    test_scores = selector.evaluate(X_test, y_test)
    print(f"Test scores: {test_scores}")