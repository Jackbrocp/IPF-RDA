import numpy as np
import numpy.linalg as LA


def compute_distance(image, adv_image, mod_list, data_name):
    image = image.numpy()
    adv_image = adv_image.numpy()
    distance_list = []
    for mod in mod_list:
        if mod == "L2":
            distance = LA.norm(image - adv_image)
        if mod == "L_inf":
            distance = np.max(np.abs(image - adv_image))
        if mod == "L1":
            distance = np.sum(np.abs(image - adv_image))
        if mod == "L0_total":
            if data_name == "mnist":
                distance = (image.shape[0] * image.shape[1]) - \
                    np.sum(np.abs(image - adv_image) <= 1e-6)
            else:
                distance = (image.shape[0] * image.shape[1] * image.shape[2]
                            ) - np.sum(np.abs(image - adv_image) <= 1e-6)
        if mod == "L0_pixel":
            if data_name == "mnist":
                distance = (image.shape[0] * image.shape[1]) - \
                    np.sum(np.abs(image - adv_image) <= 1e-6)
            else:
                distance = (image.shape[1] * image.shape[2]) - \
                    np.sum(np.all(np.abs(image - adv_image) <= 1e-6, 0))
        distance_list.append(distance)
    return distance_list


def compute_distance_batch(images, adv_images, mod="L2"):
    if mod == "L2":
        distance = LA.norm(images - adv_images, axis=(1, 2, 3))
        print(distance.shpae)
        distance = np.sum(distance)
        print(distance.shpae)
    return distance
