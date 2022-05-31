import os
import time
import csv
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
from evaluation_metrics import Metrics

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
BUFFER_SIZE = 2000
BATCH_SIZE = 1

path_to_train = 'C:\\Stage5A\\GANs\\Pix2Pix\\dataset\\New folder\\maps\\train'
path_to_val = 'C:\\Stage5A\\GANs\\Pix2Pix\\dataset\\New folder\\maps\\val'
path_to_checkpoint = 'C:\\Stage5A\\GANs\\Pix2Pix\\training_checkpoints'
path_csv = "C:\\Stage5A\\GANs\\Pix2Pix\\Evaluation_score.csv"

if os.path.exists(path_csv):
    os.remove(path_csv)

if os.path.exists(path_to_checkpoint) == False:
    os.makedirs(path_to_checkpoint)
elif os.path.exists(path_to_checkpoint) == True:
    print("Checkpoint folder exists and may contain folders! ")
    quit()

class DataLoader():
    def __init__(self, img_res=[IMG_HEIGHT, IMG_WIDTH]):
        self.img_res = img_res
        

    def load_images(self, filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)

        w = tf.shape(image)[1]
        w = int(w//2)

        sat_img, map_img = image[:, :w, :], image[:, w:, :]

        sat_img = tf.cast(sat_img, tf.float32)
        map_img = tf.cast(map_img, tf.float32)

        return sat_img, map_img

    def resize(self, sat_img, map_img, height, width):
        sat_img = tf.image.resize(sat_img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        map_img = tf.image.resize(map_img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return sat_img, map_img

    def random_crop(self, sat_img, map_img):
        stacked_image = tf.stack([sat_img, map_img], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS])
        
        return cropped_image[0], cropped_image[1]

    def normalize(self, sat_img, map_img):
        sat_img = (sat_img / 127.5) - 1
        map_img = (map_img / 127.5) - 1

        return sat_img, map_img

    @tf.function
    def random_jitter(self, sat_img, map_img):
        sat_img, map_img = self.resize(sat_img, map_img, 286, 286)
        sat_img, map_img = self.random_crop(sat_img, map_img)

        if tf.random.uniform(()) > 0.5:
            sat_img = tf.image.flip_left_right(sat_img)
            map_img = tf.image.flip_left_right(map_img)

        return sat_img, map_img

    def load_dataset(self, path, train=True):

        def load_image(filename):
            sat_img, map_img = self.load_images(filename)

            if train:
                sat_img, map_img = self.random_jitter(sat_img, map_img)
            else:
                sat_img, map_img = self.resize(sat_img, map_img, IMG_HEIGHT, IMG_WIDTH)

            sat_img, map_img = self.normalize(sat_img, map_img)

            return sat_img, map_img
        
        ds = tf.data.Dataset.list_files(path + "\\*.jpg")
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        print(f"Dataset size: {tf.data.experimental.cardinality(ds)}\n")

        if train:
            ds = ds.shuffle(BUFFER_SIZE)

        ds = ds.batch(BATCH_SIZE)

        return ds

class Pix2Pix():
    def __init__(self, img_shape=[IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS]):
            self.img_shape = img_shape

            self.generator_filter = 64
            self.discriminator_filter = 64
            self.SE_ratio = 8
            self.l1_LAMBDA = 100

            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

            self.generator = self.build_generator()

            self.discriminator = self.build_discriminator()

            self.checkpoint_prefix = os.path.join(path_to_checkpoint, "ckpt")
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                    discriminator_optimizer=self.discriminator_optimizer,
                                                    generator=self.generator,
                                                    discriminator=self.discriminator)

            self.evaluator = Metrics()

    def build_generator(self):
        init = tf.random_normal_initializer(0., 0.02)
        inputs = tf.keras.layers.Input(shape=self.img_shape)

        def encoder_block(layer_in, filters, batchnorm=True):
            x = tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same', 
                                        kernel_initializer=init, use_bias=False)(layer_in)
            
            if batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            return tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        def decoder_block(layer_in, skip_in, filters, ratio=8, dropout=True):
            x = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same',
                                                kernel_initializer=init, use_bias=False)(layer_in)
            x = tf.keras.layers.BatchNormalization()(x)

            if dropout:
                x = tf.keras.layers.Dropout(0.5)(x, training=True)

            x = tf.keras.layers.ReLU()(x)

            return tf.keras.layers.concatenate([x, skip_in])
        
        e1 = encoder_block(inputs, self.generator_filter, batchnorm=False)
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
        
    def build_discriminator(self):
        init = tf.random_normal_initializer(0., 0.02)

        in_sat_image = tf.keras.layers.Input(shape=self.img_shape, name='input_image')
        in_map_image = tf.keras.layers.Input(shape=self.img_shape, name='target_image')

        x = tf.keras.layers.concatenate([in_sat_image, in_map_image])

        filter_size = [2, 4, 8]

        x = tf.keras.layers.Conv2D(self.discriminator_filter, 4, strides=2, padding='same',
                                    kernel_initializer=init, use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        for i in filter_size:
            x = tf.keras.layers.Conv2D(self.discriminator_filter*i, 4, strides=2, padding='same',
                                        kernel_initializer=init, use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(1, 4, padding='same', kernel_initializer=init)(x)
        x = tf.keras.layers.Activation('sigmoid')(x)

        return tf.keras.Model(inputs=[in_sat_image, in_map_image], outputs=x)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target-gen_output))
        total_gen_loss = gan_loss + (self.l1_LAMBDA*l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        return real_loss + generated_loss

    def generate_images(self, model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(20,20))

        display_list = [test_input[0], tar[0], prediction[0]]
        print(display_list)
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        rmse_score = self.evaluator.rmse(tar[0].numpy(), prediction[0].numpy(), 255)
        psnr_score =  self.evaluator.psnr(tar[0].numpy(), prediction[0].numpy(), 255)
        ssim_score = self.evaluator.ssim(tar[0].numpy(), prediction[0].numpy())
        uiq_score = self.evaluator.uiq(tar[0].numpy(), prediction[0].numpy())

        print('\n=======Validation Image=======')
        print(f'RMSE Score: {rmse_score}')
        print(f'PSNR Score: {psnr_score}')
        print(f'SSIM Score: {ssim_score}')
        print(f'UIQ Score: {uiq_score}')

        if os.path.isfile(path_csv):
            with open(path_csv, 'a') as csvfile:
                score_list = ['RMSE Score', 'PSNR Score', 'SSIM Score', 'UIQ Score']
                writer = csv.DictWriter(csvfile, fieldnames=score_list, lineterminator='\n')
                writer.writerow({'RMSE Score': rmse_score,
                                'PSNR Score': psnr_score,
                                'SSIM Score': ssim_score,
                                'UIQ Score': uiq_score})
        else:
            with open(path_csv, 'w') as csvfile:
                score_list = ['RMSE Score', 'PSNR Score', 'SSIM Score', 'UIQ Score']
                writer = csv.DictWriter(csvfile, fieldnames=score_list, lineterminator='\n')
                writer.writeheader()
                writer.writerow({'RMSE Score': rmse_score,
                                'PSNR Score': psnr_score,
                                'SSIM Score': ssim_score,
                                'UIQ Score': uiq_score})

        plt.subplot(1, 3, 1)
        plt.title(title[0])
        # plt.imshow(input_image)
        plt.imshow(display_list[0]*0.5+0.5)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(title[1])
        # plt.imshow(ground_truth)
        plt.imshow(display_list[1]*0.5+0.5)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(title[2])
        # plt.imshow(predicted_image)
        plt.imshow(display_list[2]*0.5+0.5)
        plt.axis('off')

        plt.show()

    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

    def train(self, train_ds, steps):
        print("\n=======Start training=======\n")

        start = time.time()

        for step, (input_image, target) in tqdm(train_ds.repeat().take(steps).enumerate()):
            if (step) % 1000 == 0:

                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

                start = time.time()

                print(f'Step: {step//1000}k')

            self.train_step(input_image, target, step)

            if (step+1) % 5000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

if __name__ == '__main__':
    data_loader = DataLoader()

    # Load dataset
    train_ds = data_loader.load_dataset(path_to_train)
    val_ds = data_loader.load_dataset(path_to_val, trains=False)

    pix2pix_gan = Pix2Pix()

    # Model summary
    print("=======Generator Model Summary=======\n")
    pix2pix_gan.generator.summary()
    print("\n=======Discriminator Model Summary=======\n")
    pix2pix_gan.discriminator.summary()

    # Train
    pix2pix_gan.train(train_ds, 60000)

    # Run validation
    pix2pix_gan.checkpoint.restore(tf.train.latest_checkpoint(path_to_checkpoint))

    for val_input, val_target in val_ds.take(10):
        pix2pix_gan.generate_images(pix2pix_gan.generator, val_input, val_target)
