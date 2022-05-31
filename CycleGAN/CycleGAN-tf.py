import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

AUTOTUNE = tf.data.AUTOTUNE

path_to_train_horses = 'C:\\Stage5A\\GANs\\CycleGAN\\data_horse2zebra\\trainA'
path_to_train_zebra = 'C:\\Stage5A\\GANs\\CycleGAN\\data_horse2zebra\\trainB'

path_to_test_horses = 'C:\\Stage5A\\GANs\\CycleGAN\\data_horse2zebra\\testA'
path_to_test_zebra = 'C:\\Stage5A\\GANs\\CycleGAN\\data_horse2zebra\\testB'

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

def load_images(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    
    return tf.cast(image, tf.float32)

def random_crop(image):

    return tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS])

def normalize(image):

    return ((image / 127.5) - 1)

def random_jitter(image):
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)

    return tf.image.random_flip_left_right(image)

def load_dataset(path, train=True):

    def preprocess_image(filename):
        image = load_images(filename)

        if train:
            image = random_jitter(image)
            return normalize(image)
        else:
            return normalize(image)

    ds = tf.data.Dataset.list_files(path + "\\*.jpg")
    
    if train:
        ds = ds.cache()
        ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(BUFFER_SIZE)
        ds = ds.batch(BATCH_SIZE)
    else:
        ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(BUFFER_SIZE)
        ds = ds.batch(BATCH_SIZE)

    return ds

train_horses = load_dataset(path_to_train_horses)
train_zebras = load_dataset(path_to_train_zebra)
test_horses = load_dataset(path_to_test_horses, train=False)
test_zebras = load_dataset(path_to_test_zebra, train=False)

# sample_horse = next(iter(train_horses.take(1)))
# sample_zebras = next(iter(train_zebras.take(1)))

# plt.subplot(221)
# plt.title('Horse')
# plt.imshow(sample_horse[0]*0.5 + 0.5)

# plt.subplot(222)
# plt.title('Horse with random jitter')
# plt.imshow(random_jitter(sample_horse[0])*0.5 + 0.5)

for sample_zebras in train_zebras.take(1):
    plt.subplot(223)
    plt.title('Zebra')
    plt.imshow(sample_zebras[0]*0.5 + 0.5)

    plt.subplot(224)
    plt.title('Zebra with random jitter')
    plt.imshow(random_jitter(sample_zebras[0])*0.5 + 0.5)

plt.show()
    
def build_unet_generator(self):
    init = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=self.img_shape)

    def encoder_block(layer_in, filters, instancenorm=True):
        x = tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same', 
                                    kernel_initializer=init, use_bias=False)(layer_in)
        
        if instancenorm:
            x = tfa.layers.InstanceNormalization()(x)

        return tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    def decoder_block(layer_in, skip_in, filters, ratio=8, dropout=True):
        x = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same',
                                            kernel_initializer=init, use_bias=False)(layer_in)
        x = tfa.layers.InstanceNormalization()(x)

        if dropout:
            x = tf.keras.layers.Dropout(0.5)(x, training=True)

        x = tf.keras.layers.ReLU()(x)

        # # Add Squeeze and Exitation block to Decoder for Enc-Dec SEU-Net
        # se = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=True)(x)
        # se = tf.keras.layers.Conv2D(filters//ratio, 1, strides=1, padding='same', kernel_initializer=init, use_bias=False)(se)
        # se = tf.keras.layers.ReLU()(se)
        # se = tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=init, use_bias=False)(se)
        # se = tf.keras.layers.Activation('sigmoid')(se)

        # x = tf.math.multiply(se, x)

        # # Add Squeeze and Exitation block to skip connection for Enc SEU-Net
        # se_in = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=True)(skip_in)
        # se_in = tf.keras.layers.Conv2D(filters//ratio, 1, strides=1, padding='same', kernel_initializer=init, use_bias=False)(se_in)
        # se_in = tf.keras.layers.ReLU()(se_in)
        # se_in = tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=init, use_bias=False)(se_in)
        # se_in = tf.keras.layers.Activation('sigmoid')(se_in)
        
        # skip_in = tf.math.multiply(se_in, skip_in)

        return tf.keras.layers.concatenate([x, skip_in])
    
    e1 = encoder_block(inputs, self.generator_filter, instancenorm=False)
    e2 = encoder_block(e1, self.generator_filter*2)
    e3 = encoder_block(e2, self.generator_filter*4)
    e4 = encoder_block(e3, self.generator_filter*8)
    e5 = encoder_block(e4, self.generator_filter*8)
    e6 = encoder_block(e5, self.generator_filter*8)
    e7 = encoder_block(e6, self.generator_filter*8)

    b = tf.keras.layers.Conv2D(self.generator_filter*8, 4, strides=2, padding='same',
                                kernel_initializer=init, use_bias=False)(e7)
    b = tf.keras.layers.ReLU()(b)

    d1 = decoder_block(b, e7, self.generator_filter*8)
    d2 = decoder_block(d1, e6, self.generator_filter*8)
    d3 = decoder_block(d2, e5, self.generator_filter*8)
    d4 = decoder_block(d3, e4, self.generator_filter*8, dropout=False)
    d5 = decoder_block(d4, e3, self.generator_filter*4, dropout=False)
    d6 = decoder_block(d5, e2, self.generator_filter*2, dropout=False)
    d7 = decoder_block(d6, e1, self.generator_filter, dropout=False)

    x = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                        kernel_initializer=init, use_bias=False)(d7)
    x = tf.keras.layers.Activation('tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def build_resnet_generator():
    init = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS])
    padding = tf.constant([[1, 1,], [1, 1]])

    def convolutional_block(x, filters, kernel, strides, conv=True):
        x = tf.pad(x, padding, mode="REFLECT")
        if conv:
            x = tf.keras.layers.Conv2D(filters, kernel, strides, padding='valid', kernel_initializer=init, use_bias=False)(x)
        else:
            x = tf.keras.layers.Conv2DTranspose(filters, kernel, strides, padding='valid', kernel_initializer=init, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        return x

    def residual_block(x):
        x_res = tf.identity(x)
    
        x = tf.pad(x, padding, mode="REFLECT")
        x = tf.keras.layers.Conv2D(256, 3, strides=1, padding='valid', kernel_initializer=init, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.pad(x, padding, mode="REFLECT")
        x = tf.keras.layers.Conv2D(256, 3, strides=1, padding='valid', kernel_initializer=init, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.math.add(x, x_res)

        return x

    b1 = convolutional_block(inputs, 64, 7, 1)
    b2 = convolutional_block(b1, 128, 3, 2)
    b3 = convolutional_block(b2, 256, 3, 2)
    b4 = residual_block(b3)
    b5 = residual_block(b4)
    b6 = residual_block(b5)
    b7 = residual_block(b6)
    b8 = residual_block(b7)
    b9 = residual_block(b8)

    if IMG_WIDTH == 256:
        b10 = residual_block(b9)
        b11 = residual_block(b10)
        b12 = residual_block(b11)
        b13 = convolutional_block(b12, 128, 3, 2, conv=False)
    elif IMG_WIDTH == 128:
        b13 = convolutional_block(b9, 128, 3, 2, conv=False)

    b14 = convolutional_block(b13, 64, 3, 2, conv=False)
    b15 = convolutional_block(b14, 3, 7, 1)

    return tf.keras.Model(inputs=inputs, outputs=b15)
    
def build_discriminator(self):
    init = tf.random_normal_initializer(0., 0.02)

    in_input_image = tf.keras.layers.Input(shape=self.img_shape, name='input_image')
    in_target_image = tf.keras.layers.Input(shape=self.img_shape, name='target_image')

    x = tf.keras.layers.concatenate([in_input_image, in_target_image])

    filter_size = [2, 4, 8]

    x = tf.keras.layers.Conv2D(self.discriminator_filter, 4, strides=2, padding='same',
                                kernel_initializer=init, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    for i in filter_size:
        x = tf.keras.layers.Conv2D(self.discriminator_filter*i, 4, strides=2, padding='same',
                                    kernel_initializer=init, use_bias=False)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(1, 4, padding='same', kernel_initializer=init)(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    return tf.keras.Model(inputs=[in_input_image, in_target_image], outputs=x)

def generator_loss(disc_gen_out, disc_hat_out):
    pass



def discriminator_loss():
    pass





    

