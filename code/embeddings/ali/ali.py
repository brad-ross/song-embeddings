from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, \
                         Dropout, LeakyReLU, BatchNormalization, MaxPool2D,\
                         ZeroPadding2D
from keras_adversarial import AdversarialModel, simple_bigan, AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.optimizers import Adam

def compose_layers(layers):
    h = layers[0]
    for i in range(1, len(layers)):
        h = layers[i](h)
    return h

class ALIModel:
    def __init__(self):
        self.data_shape = (257, 430, 1)
        self.embedding_size = (1, 1, 64)
        self.padding_type = 'valid'

    def encoder_model(self):
        x = Input(self.data_shape)
        z = compose_layers([
            x,

            Conv2D(filters=32, kernel_size=(5, 4), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=64, kernel_size=(5, 3), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=128, kernel_size=(5, 3), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            MaxPool2D(pool_size=(2, 2), strides=2, padding=self.padding_type),

            Conv2D(filters=256, kernel_size=(4, 3), strides=2, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(1, 3), strides=2, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding=self.padding_type)
        ])

        # print z.shape

        return Model(x, z, name='encoder')


    def decoder_model(self):
        z = Input(self.embedding_size)
        x = compose_layers([
            z,

            Conv2DTranspose(filters=256, kernel_size=(6, 4), strides=1, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=128, kernel_size=(7, 4), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=64, kernel_size=(5, 4), strides=2, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=32, kernel_size=(5, 4), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2DTranspose(filters=32, kernel_size=(4, 5), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding=self.padding_type, activation='sigmoid')
        ])
        print x.shape

        return Model(z, x, name='decoder')

    def discriminator_model(self):
        z = Input(self.embedding_size)
        Dz = compose_layers([
            z,

            Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            Dropout(0.2),

            Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            Dropout(0.2)
        ])
        # print Dz.shape

        x = Input(self.data_shape)
        Dx = compose_layers([
            x,

            Conv2D(filters=32, kernel_size=(5, 4), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=64, kernel_size=(5, 3), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=128, kernel_size=(5, 3), strides=3, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            MaxPool2D(pool_size=(2, 2), strides=2, padding=self.padding_type),

            Conv2D(filters=256, kernel_size=(4, 3), strides=2, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(1, 3), strides=2, padding=self.padding_type),
            LeakyReLU(alpha=0.02),
            BatchNormalization(),
        ])
        concat_inputs = concatenate([Dz, Dx])
        conv1 = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding=self.padding_type)
        conv2 = Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding=self.padding_type)
        conv3 = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding=self.padding_type, activation='sigmoid')

        Dxz_train_disc = compose_layers([
            concat_inputs,

            conv1,
            LeakyReLU(alpha=0.02),
            Dropout(0.2),

            conv2,
            LeakyReLU(alpha=0.02),
            Dropout(0.2),

            conv3
        ])
        m_train_disc = Model([z, x], Dxz_train_disc, name='discriminator_train')

        Dxz_train_enc_dec = compose_layers([
            concat_inputs,

            conv1,
            LeakyReLU(alpha=0.02),

            conv2,
            LeakyReLU(alpha=0.02),

            conv3
        ])
        m_train_enc_dec = Model([z, x], Dxz_train_enc_dec, name='discriminator_test')

        return m_train_enc_dec, m_train_disc

    def ali_model(self):
        encoder = self.encoder_model()
        # print encoder.input_shape, encoder.output_shape
        decoder = self.decoder_model()
        disc_train_enc_dec, disc_train_disc = self.discriminator_model()

        bigan_train_enc_dec = simple_bigan(decoder, encoder, disc_train_enc_dec)
        bigan_train_disc    = simple_bigan(decoder, encoder, disc_train_disc)

        x = bigan_train_enc_dec.inputs[1]
        z = normal_latent_sampling(self.embedding_size)(x)
        # print x.shape, z.shape

        #fix names???
        bigan_train_enc_dec = Model(x, bigan_train_enc_dec([z, x]))
        bigan_train_disc    = Model(x, bigan_train_disc([z, x]))

        encoder.summary()
        decoder.summary()
        disc_train_enc_dec.summary()
        disc_train_disc.summary()
        bigan_train_enc_dec.summary()
        bigan_train_disc.summary()

        model = AdversarialModel(player_models=[bigan_train_enc_dec, bigan_train_disc],
                                 player_params=[encoder.trainable_weights + encoder.trainable_weights,\
                                                disc_train_disc.trainable_weights],
                                 player_names=['encoder_decoder', 'discriminator'])
        model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                                  player_optimizers=[Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3),\
                                                     Adam(lr=1e-4, beta_1=0.5, beta_2=1e-3)],
                                  loss='binary_crossentropy')

        return model

ali =  ALIModel()
ali.ali_model()
