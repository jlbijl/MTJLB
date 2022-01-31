import IPython.display
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from IPython.display import display, Image
from PIL import Image


save_dir = "C:/Users/janly/Documents/Master Thesis/Undersampled_data/Callbacks/exp_21_01_01/model_3.h5"
model = load_model(save_dir)

img_height = 360
img_width = 220
img_size = (img_width, img_height)



# Grad-CAM visualisation

print("==========Visualisation==========")

preprocess_input = keras.applications.xception.preprocess_input

conv_layer = "conv2d_1"

# path to target image
img_path = "C:/Users/janly/Documents/Master Thesis/Training/interrupted_print/photo_262_carrier_2_left.png"


def get_img_array(img_path, size):
    # img is a PIL image of size 220x360
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # Transform the image into an array with shape (220, 360, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # Add dimension to transform array to a batch with shape (1, 220, 360, 3)
    array = np.expand_dims(array, axis=0)
    return array


def gradcam_heatmap(image_array, loaded_model, last_conv_layer, pred_index=None):
    # create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model([loaded_model.inputs], [loaded_model.get_layer(last_conv_layer).output,
                                                               loaded_model.output])

    # compute the gradient of the top predicted class
    with tf.GradientTape() as tape:
        last_conv_layer_output, predics = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(predics[0])

        class_channel = predics[:, pred_index]

    # gradient of the top prediction with regards to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # vector of the gradient intensity
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiply each channel in the feature map with its "importance" and sum all channels
    last_conv_layer_output_new = last_conv_layer_output[0]
    heatmap = np.dot(last_conv_layer_output_new, tf.expand_dims(pooled_grads, axis=1))
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


img_array = get_img_array(img_path, (img_width, img_height))
model.layers[-1].activation = None


preds = model.predict(img_array)
print("predicted:", preds)
#model.summary()

heatmap = gradcam_heatmap(img_array, model, conv_layer, 2)

plt.matshow(heatmap)
plt.show()


def save_display_gradcam(img_path, heatmap, alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path, target_size=(img_width, img_height))
    img = keras.preprocessing.image.img_to_array(img)

    heatmap_new = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colours = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colours[heatmap_new]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((360, 220))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save("heatmap_262_2_left_v1.png")

    im = Image.open("heatmap_262_2_left_v1.png")
    im.show()


save_display_gradcam(img_path, heatmap)

