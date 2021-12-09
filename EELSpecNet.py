import tensorflow as tf


class EELSpecNetModel_CNN_10D(tf.keras.Model):

    def __init__(self, ene_dim):
        super(EELSpecNetModel_CNN_10D, self).__init__()

        kerl_size = 4

        self.conv_1024x64 = tf.keras.layers.Conv2D(64, (1, kerl_size), strides=(1, 2),
                                                   activation='relu', padding='same',
                                                   kernel_initializer='random_uniform')

        self.conv_512x128 = tf.keras.layers.Conv2D(128, (1, kerl_size), strides=(1, 2),
                                                   activation='relu', padding='same',
                                                   kernel_initializer='random_uniform')

        self.conv_256x256 = tf.keras.layers.Conv2D(256, (1, kerl_size), strides=(1, 2),
                                                   activation='relu', padding='same',
                                                   kernel_initializer='random_uniform')

        self.conv_128x512 = tf.keras.layers.Conv2D(512, (1, kerl_size), strides=(1, 2),
                                                   activation='relu', padding='same',
                                                   kernel_initializer='random_uniform')

        self.conv_64x1024 = tf.keras.layers.Conv2D(1024, (1, kerl_size), strides=(1, 2),
                                                   activation='relu', padding='same',
                                                   kernel_initializer='random_uniform')

        self.conv_32x2048 = tf.keras.layers.Conv2D(2048, (1, kerl_size), strides=(1, 2),
                                                   activation='relu', padding='same',
                                                   kernel_initializer='random_uniform')

        self.conv_16x2048 = tf.keras.layers.Conv2D(2048, (1, kerl_size), strides=(1, 2),
                                                   activation='relu', padding='same',
                                                   kernel_initializer='random_uniform')

        self.conv_8x2048 = tf.keras.layers.Conv2D(2048, (1, kerl_size), strides=(1, 2),
                                                  activation='relu', padding='same',
                                                  kernel_initializer='random_uniform')

        self.conv_4x2048 = tf.keras.layers.Conv2D(2048, (1, kerl_size), strides=(1, 2),
                                                  activation='relu', padding='same',
                                                  kernel_initializer='random_uniform')

        self.conv_2x2048 = tf.keras.layers.Conv2D(2048, (1, kerl_size), strides=(1, 2),
                                                  activation='relu', padding='same',
                                                  kernel_initializer='random_uniform')

        # =======================================================================

        self.deconv_4x2048 = tf.keras.layers.Conv2DTranspose(2048, (1, kerl_size), strides=(1, 2),
                                                             activation='relu', padding='same',
                                                             kernel_initializer='random_uniform')

        self.deconv_8x2048 = tf.keras.layers.Conv2DTranspose(2048, (1, kerl_size), strides=(1, 2),
                                                             activation='relu', padding='same',
                                                             kernel_initializer='random_uniform')

        self.deconv_16x2048 = tf.keras.layers.Conv2DTranspose(2048, (1, kerl_size), strides=(1, 2),
                                                              activation='relu', padding='same',
                                                              kernel_initializer='random_uniform')

        self.deconv_32x2048 = tf.keras.layers.Conv2DTranspose(2048, (1, kerl_size), strides=(1, 2),
                                                              activation='relu', padding='same',
                                                              kernel_initializer='random_uniform')

        self.deconv_64x1024 = tf.keras.layers.Conv2DTranspose(1024, (1, kerl_size), strides=(1, 2),
                                                              activation='relu', padding='same',
                                                              kernel_initializer='random_uniform')

        self.deconv_128x512 = tf.keras.layers.Conv2DTranspose(512, (1, kerl_size), strides=(1, 2),
                                                              activation='relu', padding='same',
                                                              kernel_initializer='random_uniform')

        self.deconv_256x256 = tf.keras.layers.Conv2DTranspose(256, (1, kerl_size), strides=(1, 2),
                                                              activation='relu', padding='same',
                                                              kernel_initializer='random_uniform')

        self.deconv_512x128 = tf.keras.layers.Conv2DTranspose(128, (1, kerl_size), strides=(1, 2),
                                                              activation='relu', padding='same',
                                                              kernel_initializer='random_uniform')

        self.deconv_1024x64 = tf.keras.layers.Conv2DTranspose(64, (1, kerl_size), strides=(1, 2),
                                                              activation='relu', padding='same',
                                                              kernel_initializer='random_uniform')

        self.deconv_2048x1 = tf.keras.layers.Conv2DTranspose(1, (1, kerl_size), strides=(1, 2),
                                                             activation='tanh', padding='same',
                                                             kernel_initializer='random_uniform')

        self.concat = tf.keras.layers.concatenate
        self.relu = tf.keras.activations.relu

    def call(self, inputs):
        enc_1024x64 = self.conv_1024x64(inputs)

        enc_512x128 = self.conv_512x128(enc_1024x64)

        enc_256x256 = self.conv_256x256(enc_512x128)

        enc_128x512 = self.conv_128x512(enc_256x256)

        enc_64x1024 = self.conv_64x1024(enc_128x512)

        enc_32x2048 = self.conv_32x2048(enc_64x1024)

        enc_16x2048 = self.conv_16x2048(enc_32x2048)

        enc_8x2048 = self.conv_8x2048(enc_16x2048)

        enc_4x2048 = self.conv_4x2048(enc_8x2048)

        enc_2x2048 = self.conv_2x2048(enc_4x2048)

        # =======================================================================

        dcd_4x2048 = self.deconv_4x2048(enc_2x2048)
        dcd_4x2048x2 = self.concat([dcd_4x2048, enc_4x2048], axis=-1)

        dcd_8x2048 = self.deconv_8x2048(dcd_4x2048x2)
        dcd_8x2048x2 = self.concat([dcd_8x2048, enc_8x2048], axis=-1)

        dcd_16x2048 = self.deconv_16x2048(dcd_8x2048x2)
        dcd_16x2048x2 = self.concat([dcd_16x2048, enc_16x2048], axis=-1)

        dcd_32x2048 = self.deconv_32x2048(dcd_16x2048x2)
        dcd_32x2048x2 = self.concat([dcd_32x2048, enc_32x2048], axis=-1)

        dcd_64x1024 = self.deconv_64x1024(dcd_32x2048x2)
        dcd_64x1024x2 = self.concat([dcd_64x1024, enc_64x1024], axis=-1)

        dcd_128x512 = self.deconv_128x512(dcd_64x1024x2)
        dcd_128x512x2 = self.concat([dcd_128x512, enc_128x512], axis=-1)

        dcd_256x256 = self.deconv_256x256(dcd_128x512x2)
        dcd_256x256x2 = self.concat([dcd_256x256, enc_256x256], axis=-1)

        dcd_512x128 = self.deconv_512x128(dcd_256x256x2)
        dcd_512x128x2 = self.concat([dcd_512x128, enc_512x128], axis=-1)

        dcd_1024x64 = self.deconv_1024x64(dcd_512x128x2)
        dcd_1024x64x2 = self.concat([dcd_1024x64, enc_1024x64], axis=-1)

        dcd_2048x1 = self.deconv_2048x1(dcd_1024x64x2)

        return (dcd_2048x1)