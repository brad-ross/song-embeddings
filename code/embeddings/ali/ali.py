from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose, \
                         Dropout, LeakyReLU, BatchNormalization, MaxPool2D

def compose_layers(layers):
    h = layers[0]
    for i in range(1, len(layers)):
        h = layers[i](h)
    return h

class ALIModel:
    def __init__(self):
        self.data_shape = (257, 430, 1)
        self.embedding_size = (64,)

    def encoder_model(self):
        z = compose_layers([
            Input(self.data_shape),
            ZeroPadding2D(padding=(14, 7)),

            Conv2D(filters=32, kernel_size=(5, 4), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=64, kernel_size=(5, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=128, kernel_size=(5, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

            Conv2D(filters=256, kernel_size=(4, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(3, 2), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same'),
        ])

        return Model([x], z)


    def decoder_model(self):
        x = compose_layers([
            Input(self.embedding_size),

            Conv2DTranspose(filters=256, kernel_size=(7, 4), strides=1, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=128, kernel_size=(6, 5), strides=3, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=32, kernel_size=(5, 3), strides=3, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=32, kernel_size=(5, 4), strides=3, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same'),
            #should these be here?
            LeakyReLU(alpha=0.02),
            BatchNormalization(),
        ])


        return Model([z], x)

    def discriminator_model(self):
        z = Input(self.embedding_size)
        Dz = compose_layers([
            z,

            Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same'),
            LeakyReLU(alpha=0.02),
            Dropout(0.2),

            Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same'),
            LeakyReLU(alpha=0.02),
            Dropout(0.2)
        ])

        x = Input(self.data_shape)
        Dx = compose_layers([
            x,

            Conv2D(filters=32, kernel_size=(5, 4), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=64, kernel_size=(5, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=128, kernel_size=(5, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),

            Conv2D(filters=256, kernel_size=(4, 3), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(3, 2), strides=2, padding='same'),
            LeakyReLU(alpha=0.02),
            BatchNormalization()
        ])


        Dxz = compose_layers([
            Concatenate([Dz, Dx])

            Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding='same'),
            LeakyReLU(alpha=0.02),
            Dropout(0.2),

            Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding='same'),
            LeakyReLU(alpha=0.02),
            Dropout(0.2),

            Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same', activation='sigmoid')
        ])

        return Model([x, z], Dxz)

    def ali_model(self):
        encoder = self.encoder_model()
        decoder = self.decoder_model()
        discriminator = self.discriminator_model()

        real_x_discriminator = discriminator([decoder.inputs[0], encoder])
        fake_x_discriminator = discriminator([decoder, encoder.inputs[1]])
