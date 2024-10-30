import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, LSTM, MultiHeadAttention, LayerNormalization,
    Dropout, Concatenate, Add
)
from tensorflow.keras.models import Model
import numpy as np

class TimeSeriesAttention(Layer):
    """Custom attention layer per time series"""
    def __init__(self, num_heads, key_dim, dropout=0.1, **kwargs):
        super(TimeSeriesAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        self.layernorm = LayerNormalization()
        self.dropout = Dropout(dropout)
        
    def call(self, inputs, training=None):
        x = inputs
        # Self attention
        attention_output = self.mha(x, x, x)
        attention_output = self.dropout(attention_output, training=training)
        # Skip connection e layer normalization
        return self.layernorm(x + attention_output)

class LSTMAttentionModel:
    def __init__(self,
                 sequence_length: int,
                 num_features: int,
                 num_heads: int = 4,
                 lstm_units: list = [100, 50],
                 attention_dim: int = 32,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001):
        """
        Args:
            sequence_length: Lunghezza delle sequenze di input
            num_features: Numero di feature per ogni timestep
            num_heads: Numero di attention heads
            lstm_units: Lista con numero di unità per ogni layer LSTM
            attention_dim: Dimensione dello spazio delle chiavi nell'attention
            dropout_rate: Tasso di dropout
            learning_rate: Learning rate per l'ottimizzatore
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_heads = num_heads
        self.lstm_units = lstm_units
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        
    def _build_model(self):
        """Costruisce il modello LSTM con attention e skip connections"""
        # Input layer
        inputs = tf.keras.Input(shape=(self.sequence_length, self.num_features))
        x = inputs
        
        # Lista per skip connections
        skip_connections = []
        
        # Prima parte: LSTM layers con skip connections
        for i, units in enumerate(self.lstm_units):
            lstm_out = LSTM(
                units,
                return_sequences=True,
                name=f'lstm_{i}'
            )(x)
            
            # Aggiungi skip connection
            if i > 0:
                lstm_out = Add()([lstm_out, skip_connections[-1]])
            
            skip_connections.append(lstm_out)
            x = lstm_out
            
            # Aggiungi dropout dopo ogni LSTM
            x = Dropout(self.dropout_rate)(x)
        
        # Seconda parte: Multi-head attention con skip connection dall'ultimo LSTM
        attention_out = TimeSeriesAttention(
            num_heads=self.num_heads,
            key_dim=self.attention_dim,
            dropout=self.dropout_rate,
            name='multi_head_attention'
        )(x)
        
        # Combina output attention con ultimo LSTM usando skip connection
        combined = Add()([attention_out, skip_connections[-1]])
        
        # Feature extraction layer
        features = Dense(
            self.attention_dim,
            activation='relu',
            name='feature_extraction'
        )(combined)
        
        # Output layer
        outputs = Dense(self.num_features, name='output')(features)
        
        # Costruisci il modello
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self,
            X_train,
            y_train,
            validation_data=None,
            epochs=100,
            batch_size=32,
            verbose=1):
        """Addestra il modello"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """Effettua predizioni"""
        return self.model.predict(X)
    
    def get_attention_weights(self, X):
        """Estrae i pesi dell'attention per analisi"""
        attention_layer = [layer for layer in self.model.layers 
                         if isinstance(layer, TimeSeriesAttention)][0]
        
        # Crea un modello intermedio per estrarre i pesi
        intermediate_model = Model(
            inputs=self.model.input,
            outputs=attention_layer.mha.output
        )
        
        return intermediate_model.predict(X)
    
    def analyze_attention(self, X, timestamps=None):
        """Analizza e visualizza i pattern dell'attention"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        attention_weights = self.get_attention_weights(X)
        avg_attention = np.mean(attention_weights, axis=(0, 1))
        
        plt.figure(figsize=(12, 6))
        
        if timestamps is not None:
            plt.plot(timestamps, avg_attention)
            plt.xlabel('Time')
        else:
            plt.plot(avg_attention)
            plt.xlabel('Sequence Position')
            
        plt.title('Average Attention Weights')
        plt.ylabel('Attention Weight')
        plt.grid(True)
        plt.show()

# Esempio di utilizzo
if __name__ == "__main__":
    """
    # Prepara i dati
    sequence_length = 60
    num_features = 10
    
    model = LSTMAttentionModel(
        sequence_length=sequence_length,
        num_features=num_features,
        num_heads=4,
        lstm_units=[100, 50],
        attention_dim=32,
        dropout_rate=0.1
    )
    
    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32
    )
    
    # Predizione
    predictions = model.predict(X_test)
    
    # Analisi attention
    model.analyze_attention(X_test, timestamps=test_dates)
    """