import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization
import numpy as np

class ScaledDotProductAttention(Layer):
    def __init__(self, d_k, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k
        
    def call(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dense = Dense(d_model)
        self.dropout = Dropout(dropout_rate)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.depth)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        output = self.dropout(output)
        
        return output, attention_weights

class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        
    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class LuongAttention(Layer):
    def __init__(self, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.W = Dense(attention_dim)
        
    def call(self, query, values):
        score = tf.matmul(query, self.W(values), transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, values)
        
        return context_vector, attention_weights

class RelativePositionAttention(Layer):
    def __init__(self, d_model, max_relative_position, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.rel_pos_embedding = self.add_weight(
            shape=(2 * max_relative_position + 1, d_model),
            initializer='random_normal',
            trainable=True,
            name='rel_pos_embedding'
        )
        
    def call(self, q, k, v, mask=None):
        seq_length = tf.shape(q)[1]
        
        # Generate relative position matrix
        range_vec = tf.range(seq_length)
        relative_positions = range_vec[None, :] - range_vec[:, None]
        relative_positions = tf.clip_by_value(
            relative_positions + self.max_relative_position,
            0,
            2 * self.max_relative_position
        )
        
        relative_embeddings = tf.gather(self.rel_pos_embedding, relative_positions)
        
        # Compute attention scores with relative positions
        attention_scores = tf.matmul(q, k, transpose_b=True)
        relative_scores = tf.matmul(q, tf.transpose(relative_embeddings, [0, 2, 1]))
        
        attention_scores = attention_scores + relative_scores
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        if mask is not None:
            attention_scores += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights

def create_advanced_attention_model(input_shape, output_shape, base_model_type='lstm', 
                                 attention_type='multi_head', d_model=128, num_heads=8, 
                                 dropout_rate=0.1, units=64):
    """
    Crea un modello con attention layer che può essere integrato con LSTM, GRU o Transformer.
    
    Args:
        input_shape: Forma dell'input (timesteps, features)
        output_shape: Dimensione dell'output
        base_model_type: 'lstm', 'gru', o 'transformer'
        attention_type: 'multi_head', 'bahdanau', 'luong', o 'relative'
        d_model: Dimensione del modello per multi-head attention
        num_heads: Numero di teste per multi-head attention
        dropout_rate: Tasso di dropout
        units: Unità per Bahdanau/Luong attention
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Base model selection
    if base_model_type == 'lstm':
        x = tf.keras.layers.LSTM(units, return_sequences=True)(inputs)
    elif base_model_type == 'gru':
        x = tf.keras.layers.GRU(units, return_sequences=True)(inputs)
    else:  # transformer
        x = inputs  # Transformer processing would go here
    
    # Attention layer selection
    if attention_type == 'multi_head':
        attention_layer = MultiHeadAttention(d_model, num_heads, dropout_rate)
        x, _ = attention_layer(x, x, x)
    elif attention_type == 'bahdanau':
        attention_layer = BahdanauAttention(units)
        x, _ = attention_layer(x[:, -1, :], x)
    elif attention_type == 'luong':
        attention_layer = LuongAttention(units)
        x, _ = attention_layer(x[:, -1, :], x)
    elif attention_type == 'relative':
        attention_layer = RelativePositionAttention(d_model, max_relative_position=10)
        x, _ = attention_layer(x, x, x)
    
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    
    if attention_type != 'bahdanau' and attention_type != 'luong':
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model

"""
# Esempio di utilizzo
input_shape = (30, 10)  # 30 timesteps, 10 features
output_shape = 1  # previsione singola

# Creazione modello con Multi-Head Attention
model = create_advanced_attention_model(
    input_shape=input_shape,
    output_shape=output_shape,
    base_model_type='lstm',  # o 'gru'
    attention_type='multi_head',
    d_model=128,
    num_heads=8
)
"""