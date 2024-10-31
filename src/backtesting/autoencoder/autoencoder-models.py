import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Dropout, Lambda, BatchNormalization

class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim, type='classic', **kwargs):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.type = type
        self.encoder = self.build_encoder(**kwargs)
        self.decoder = self.build_decoder(**kwargs)

    def build_encoder(self, **kwargs):
        encoder_input = Input(shape=self.input_dim)
        x = encoder_input
        
        if self.type == 'classic':
            x = Dense(self.latent_dim, activation='relu')(x)
        elif self.type == 'denoising':
            noise = Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x), stddev=kwargs.get('noise_stddev', 0.1)))(x)
            x = Dense(self.latent_dim, activation='relu')(noise)
        elif self.type == 'sparse':
            x = Dense(self.latent_dim, activation='relu', activity_regularizer=regularizers.l1(kwargs.get('sparse_reg', 1e-5)))(x)
        elif self.type == 'contractive':
            h = Dense(self.latent_dim, activation='relu')(x)
            dh = Lambda(lambda x: tf.gradients(x, encoder_input)[0]**2)(h)
            x = Dense(self.latent_dim, activation='relu')(h)
            self.add_loss(kwargs.get('contractive_reg', 1e-4) * tf.reduce_mean(dh))
        elif self.type == 'convolutional':
            x = Reshape((self.input_dim[0], 1))(x)
            x = Conv1D(32, 3, activation='relu', padding='same')(x)
            x = Conv1D(64, 3, activation='relu', padding='same')(x)
            x = Flatten()(x)
            x = Dense(self.latent_dim)(x)
        elif self.type == 'variational' or self.type == 'adversarial':
            x = Dense(self.latent_dim)(x)
            z_mean = Dense(self.latent_dim)(x)
            z_log_var = Dense(self.latent_dim)(x)
            x = Lambda(self.sampling)([z_mean, z_log_var])
            
        encoder = Model(encoder_input, x, name='encoder')
        return encoder

    def build_decoder(self, **kwargs):
        decoder_input = Input(shape=(self.latent_dim,))
        x = decoder_input
        
        if self.type == 'classic':
            x = Dense(self.input_dim[0], activation='sigmoid')(x)
        elif self.type in ['denoising', 'sparse', 'contractive']:
            x = Dense(self.input_dim[0], activation='sigmoid')(x)
        elif self.type == 'convolutional':
            x = Dense(self.input_dim[0] // 4 * 64)(x)
            x = Reshape((self.input_dim[0] // 4, 64))(x)
            x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
            x = Conv2DTranspose(1, 3, strides=2, activation='sigmoid', padding='same')(x)
            x = Reshape(self.input_dim)(x)
        elif self.type == 'variational' or self.type == 'adversarial':
            x = Dense(self.input_dim[0], activation='sigmoid')(x)
            
        decoder = Model(decoder_input, x, name='decoder')
        return decoder
    
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        if self.type == 'adversarial':
            # Adversarial part
            discriminator = self.build_discriminator()
            generator_loss = self.generator_loss(discriminator)
            self.add_loss(generator_loss)

        return decoded
      
    def generator_loss(self, discriminator):
        fake_latent = self.encoder(self.decoder(tf.random.normal(shape=(100, self.latent_dim))))
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(discriminator(fake_latent)), discriminator(fake_latent)))
    
    def build_discriminator(self):
        discriminator_input = Input(shape=(self.latent_dim,))
        x = discriminator_input
        x = Dense(128, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        discriminator = Model(discriminator_input, x, name='discriminator')
        return discriminator
