import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
# from IPython.display import display, Image
from PIL import Image
import skimage.io
import skimage.segmentation
import skimage.transform
import copy
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageEnhance
import cv2.cv2 as cv2

img_path =      # The path where the input image is saved
save_dir =      # The path where the trained model is saved
model = load_model(save_dir)

img_height = 360
img_width = 220


org_img = skimage.io.imread("photo_xxx_carrier_x_xxxx.png")
org_img = skimage.transform.resize(org_img, (img_width, img_height))

np.random.seed(372)
preds = model.predict(org_img[np.newaxis, :, :, :])
top_pred_classes = preds[0].argsort()

# generate segmentation for image
bright_img = Image.open("photo_xxx_carrier_x_xxxx.png")
# brightness enhancer
enhancer = ImageEnhance.Brightness(bright_img)

factor = 2.5
bright_output = enhancer.enhance(factor)
bright_output.save("brightened_background_xxx_x_xxxx.png")
img = mpimg.imread("brightened_background_xxx_x_xxxx.png")
#img = skimage.transform.resize(img, (img_width, img_height))
superpixels = skimage.segmentation.quickshift(org_img, kernel_size=4, max_dist=50, ratio=0.005)
num_superpixels = np.unique(superpixels).shape[0]

superpixplot = plt.imshow(skimage.segmentation.mark_boundaries(org_img, superpixels))
#plt.show()

# generate perturbations
num_perturb = 300
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))


# create function to apply perturbations to images

def perturb_image(image, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)

    for active in active_pixels:
        mask[segments == active] = 1

    perturbed_image = copy.deepcopy(image)
    perturbed_image = perturbed_image*mask[:, :, np.newaxis]
    return perturbed_image


#predicting class for perturbations
predictions = []

for pert in perturbations:
    perturbed_img = perturb_image(org_img, pert, superpixels)
    pred = model.predict(perturbed_img[np.newaxis, :, :, :])
    predictions.append(pred)

predictions = np.array(predictions)


# compute weights for the perturbations
# compute distances to original image
original_image = np.ones(num_superpixels)[np.newaxis, :]
distances = pairwise_distances(perturbations, original_image, metric="cosine").ravel()

# transform distances to a value between 0 and 1 using a kernel function
kernel_width = 0.5
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2))


# fit a explainable linear model
class_to_explain = top_pred_classes[0]
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:, :, class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]

num_top_features = 4
top_features = np.argsort(coeff)[-num_top_features]

mask = np.zeros(num_superpixels)
mask[top_features] = True

# use original image as background to highlight activated area
# adjust the brightness of the background to let the superpixel stand out

# load the background into the new plot
lime_img = perturb_image(img, mask, superpixels)
lime_imgplot = plt.imshow(lime_img)
plt.imsave("lime_img_xxx_x_xxxx_with_black.png", lime_img)
plt.show()

new_lime_img = skimage.io.imread(img_path)
# remove the background from perturbed image
img_float32 = np.float32(new_lime_img)
gray = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)

mask = cv2.inRange(gray, 1, 255)

# get contours # OpenCV 3.4, in OpenCV 2* or 4* it returns (contours, _)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# get biggest contour
biggest = None
biggest_size = 0

for con in contours:
    area = cv2.contourArea(con)

    if area > biggest_size:
        biggest_size = area
        biggest = con

# make mask with biggest contour
#biggest = max(contours, key=cv2.contourArea)
new_mask = np.zeros_like(mask)
cv2.drawContours(new_mask, contours, 0, 255, -1)
# redraw with white
#lime_img[new_mask == 0] = (255, 255, 255)


# put mask into alpha channel of input
result = cv2.cvtColor(new_lime_img, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = new_mask

plt.imsave("lime_img_262_2_left_no_black.png", result)


def creatingBackground(filepath: str, brightness: float, img_size: tuple):
    back_img = skimage.io.imread(filepath)
    back_img = skimage.transform.resize(back_img, img_size)
    background_float32 = np.float32(back_img)
    plt.imsave(filepath, background_float32)

    background_image = Image.open(filepath)

    enhancer = ImageEnhance.Brightness(background_image)

    factor = brightness
    background_output = enhancer.enhance(factor)
    background_output.save("darkened_background_"+filepath[-22:])

    return Image.open("darkened_background_"+filepath[-22:])

final_lime_image = Image.open("lime_img_xxx_x_xxxx_no_black.png")

final_background = creatingBackground(img_path, 0.5, (img_width, img_height))

final_background.paste(final_lime_image, (0, 0), final_lime_image)
final_background.save("final_lime_xxx_x_xxxx_first_try.png")
final_background.show()
