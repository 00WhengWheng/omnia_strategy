import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention

def create_transformer_model(input_shape, output_shape, num_layers, d_model, num_heads, dff, dropout_rate=0.1, optimizer='adam', positional_encoding=True):
    inputs = Input(shape=input_shape)
    
    # Positional encoding
    if positional_encoding:
        position = tf.range(start=0, limit=input_shape[-2], delta=1)
        position_embedding = positional_encoding(position, d_model)
        x = inputs + position_embedding
    else:
        x = inputs
    
    # Transformer layers
    for _ in range(num_layers):
        # Multi-head self-attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feedforward network
        ffn_output = Dense(dff, activation='relu')(x)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Output layer
    outputs = Dense(output_shape)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def positional_encoding(position, d_model):
    angle_rads = get_angles(tf.range(position)[:, tf.newaxis],
                            tf.range(d_model)[tf.newaxis, :],
                            d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = tf.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = tf.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[tf.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / tf.pow(10000, (2 * (i//2)) / tf.cast(d_model, tf.float32))
    return pos * angle_rates
