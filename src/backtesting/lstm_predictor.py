import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class LSTMPredictor:
    def __init__(self, sequence_length=60, n_features=None, lstm_units=100,
                 dropout_rate=0.2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data):
        """Crea sequenze per il training dell'LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Costruisce il modello LSTM con dropout"""
        model = Sequential([
            # Primo layer LSTM con dropout
            LSTM(units=self.lstm_units,
                 return_sequences=True,
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(self.dropout_rate),
            
            # Secondo layer LSTM con dropout
            LSTM(units=self.lstm_units//2,
                 return_sequences=True),
            Dropout(self.dropout_rate),
            
            # Terzo layer LSTM
            LSTM(units=self.lstm_units//4),
            Dropout(self.dropout_rate),
            
            # Layer di output
            Dense(self.n_features)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Addestra il modello con early stopping"""
        if self.n_features is None:
            self.n_features = X_train.shape[2]
            
        if self.model is None:
            self.model = self.build_model()
            
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        
        # Training con validation
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Effettua predizioni"""
        return self.model.predict(X)
    
    def evaluate_model(self, X_test, y_test):
        """Valuta il modello su test set"""
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'test_loss': test_loss,
            'test_mae': test_mae
        }

# Esempio di utilizzo
if __name__ == "__main__":
    # Esempio di preparazione dati
    def prepare_data(data, train_size=0.8):
        # Normalizzazione
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Creazione sequenze
        predictor = LSTMPredictor()
        X, y = predictor.create_sequences(scaled_data)
        
        # Split train/validation/test
        train_size = int(len(X) * train_size)
        val_size = int(len(X) * 0.1)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # Esempio di training
    """
    # Assumendo di avere i dati preparati
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(your_data)
    
    predictor = LSTMPredictor(
        sequence_length=60,
        lstm_units=100,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    history = predictor.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # Valutazione
    results = predictor.evaluate_model(X_test, y_test)
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")
    """