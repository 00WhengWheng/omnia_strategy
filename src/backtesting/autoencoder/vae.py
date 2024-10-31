import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, BatchNormalization, 
                                   Lambda, Layer)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

class Sampling(Layer):
    """Layer di sampling per VAE"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class TradingVAE:
    def __init__(self, input_dim, latent_dim=32, intermediate_dims=[64, 48]):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        
        self.encoder = None
        self.decoder = None
        self.vae = None
        
        self._build_encoder()
        self._build_decoder()
        self._build_vae()
        
    def _build_encoder(self):
        """Costruisce l'encoder con batch normalization"""
        # Input layer
        encoder_input = Input(shape=(self.input_dim,), name='encoder_input')
        x = encoder_input
        
        # Layer intermedi con batch normalization
        for i, dim in enumerate(self.intermediate_dims):
            x = Dense(dim, name=f'encoder_dense_{i}')(x)
            x = BatchNormalization(name=f'encoder_bn_{i}')(x)
            x = tf.keras.layers.LeakyReLU()(x)
            
        # Layer VAE per mean e variance
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # Layer di sampling
        z = Sampling()([z_mean, z_log_var])
        
        # Crea modello encoder
        self.encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
        
    def _build_decoder(self):
        """Costruisce il decoder con batch normalization"""
        # Input layer
        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')
        x = decoder_input
        
        # Layer intermedi inversi
        for i, dim in enumerate(reversed(self.intermediate_dims)):
            x = Dense(dim, name=f'decoder_dense_{i}')(x)
            x = BatchNormalization(name=f'decoder_bn_{i}')(x)
            x = tf.keras.layers.LeakyReLU()(x)
            
        # Output layer
        decoder_output = Dense(self.input_dim, activation='linear', name='decoder_output')(x)
        
        # Crea modello decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        
    def _build_vae(self):
        """Costruisce il modello VAE completo"""
        encoder_input = Input(shape=(self.input_dim,))
        z_mean, z_log_var, z = self.encoder(encoder_input)
        reconstruction = self.decoder(z)
        
        # Custom layer per loss
        class VAELayer(Layer):
            def __init__(self, **kwargs):
                super(VAELayer, self).__init__(**kwargs)

            def vae_loss(self, x, reconstruction, z_mean, z_log_var):
                # Reconstruction loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.mse(x, reconstruction),
                        axis=1
                    )
                )
                
                # KL divergence loss
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                
                return reconstruction_loss + kl_loss
                
            def call(self, inputs):
                x = inputs[0]
                reconstruction = inputs[1]
                z_mean = inputs[2]
                z_log_var = inputs[3]
                loss = self.vae_loss(x, reconstruction, z_mean, z_log_var)
                self.add_loss(loss)
                return reconstruction
        
        # Costruisci modello finale
        self.vae = Model(
            encoder_input,
            VAELayer()([encoder_input, reconstruction, z_mean, z_log_var]),
            name='vae'
        )
        
        # Compile
        self.vae.compile(optimizer='adam')
        
    def fit(self, x_train, epochs=100, batch_size=32, validation_data=None):
        """Addestra il VAE"""
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        return self.vae.fit(
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping]
        )
    
    def encode(self, data):
        """Codifica i dati nello spazio latente"""
        return self.encoder.predict(data)[2]  # Returns z (sampled point)
    
    def decode(self, latent_data):
        """Decodifica dati dallo spazio latente"""
        return self.decoder.predict(latent_data)
    
    def reconstruct(self, data):
        """Ricostruisce i dati attraverso encoding e decoding"""
        return self.vae.predict(data)
    
    def generate(self, n_samples=1):
        """Genera nuovi dati campionando dallo spazio latente"""
        random_latent_points = np.random.normal(size=(n_samples, self.latent_dim))
        return self.decode(random_latent_points)

# Esempio di utilizzo
if __name__ == "__main__":
    """
    # Preparazione dati
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    
    # Creazione e training del modello
    vae = TradingVAE(
        input_dim=x_train_scaled.shape[1],
        latent_dim=32,
        intermediate_dims=[64, 48]
    )
    
    history = vae.fit(
        x_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(x_val_scaled, None)
    )
    
    # Encoding dei dati
    latent_representation = vae.encode(x_train_scaled)
    
    # Ricostruzione
    reconstructed_data = vae.reconstruct(x_train_scaled)
    
    # Generazione di nuovi dati
    new_samples = vae.generate(n_samples=10)
    """