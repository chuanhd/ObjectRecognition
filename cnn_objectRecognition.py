import numpy as np
from scipy import misc
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def loadData():
    # Make a queue of file names including all the PNG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./train_small/*.png"))

    # Read an entire image file which is required since they're PNGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()


    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_png(image_file)

    print(image.shape)

    return 

def cnn_model_fn(_features, _labels, _mode):
    return

def main(_):
    # Start new session
    with tf.Session() as sess:
        # Required to get the filename matching to run
        tf.global_variables_initializer().run()
        loadData()
if __name__ == "__main__":
    tf.app.run()