import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import entropy

# from algorithm.entropy_estimators import get_h_mvn, get_h


def extract_features(
        image: np.ndarray,
        mask: np.ndarray,
        pixels: np.ndarray,
        prefix: str
) -> dict:
    d1 = extract_features_by_pixels(pixels, prefix)
    d2 = extract_features_by_mask(mask, prefix)
    d3 = extract_features_by_image(image, mask, prefix)

    feature_dict = {**d1, **d2, **d3}
    return feature_dict


def extract_features_by_pixels(pixels: np.ndarray, prefix: str) -> dict:
    feature_dict = {
        # size: pixel count
        'size': len(pixels),
        # mean/std
        'mean': np.mean(pixels),
        'std': np.std(pixels),
        # min/quantile/median/max
        'min': np.min(pixels),
        'q01': np.quantile(pixels, 0.01),
        'median': np.median(pixels),
        'q99': np.quantile(pixels, 0.99),
        'max': np.max(pixels),
        # entropy
        'entropy_simple': get_entropy_simple(pixels),
        #'entropy_analytic': get_entropy_analytic(pixels),
    }

    feature_dict = add_prefix_to_dict(feature_dict, prefix)
    return feature_dict


def extract_features_by_mask(mask: np.ndarray, prefix: str) -> dict:
    feature_dict = {
        # roundness
        'roundness': get_mask_roundness(mask),
    }

    additional_features = get_mask_enclosing_rectangle_features(mask)
    feature_dict = {**feature_dict, **additional_features}

    feature_dict = add_prefix_to_dict(feature_dict, prefix)
    return feature_dict


def extract_features_by_image(image: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    # ToDo: loop through kernel list to extract features
    activation_mean, activation_q95 = get_laplace_feature(image, mask, kernel, display=False)

    feature_dict = {
        'act_mean': activation_mean,
        'act_q95': activation_q95
    }

    feature_dict = add_prefix_to_dict(feature_dict, prefix)
    return feature_dict


# Shared functions
def add_prefix_to_dict(feature_dict: dict, prefix: str):
    new_keys = list(map(lambda x: prefix + '_' + x, feature_dict.keys()))
    feature_dict = dict(zip(new_keys, feature_dict.values()))

    return feature_dict


def add_suffix_to_dict(feature_dict: dict, suffix: str):
    new_keys = list(map(lambda x: x + '_' + suffix, feature_dict.keys()))
    feature_dict = dict(zip(new_keys, feature_dict.values()))

    return feature_dict


def get_features_diff(features_before: dict, features_after: dict) -> dict:
    key_list = list(features_after.keys())
    feature_diff = {key: features_after[key] - features_before.get(key, 0) for key in key_list}

    return feature_diff


# Pixel level extraction
def get_entropy_simple(pixels: np.ndarray, n_bins=30):
    discretized_dist = np.histogram(pixels, bins=n_bins)[0] / len(pixels)
    simple_entropy = entropy(discretized_dist)
    return simple_entropy


# def get_entropy_analytic(pixels: np.ndarray):
#     """
#     compute the entropy from the determinant of the multivariate normal distribution:
#     """
#     assert len(pixels.shape) == 1
#     pixels_n_dim = pixels[:, np.newaxis]
#     analytic_entropy = get_h_mvn(pixels_n_dim)
#     return analytic_entropy


# def get_entropy_kozachenko(pixels: np.ndarray, k_neighbor=50):
#     """
#     compute the entropy using the k-nearest neighbour approach
#     developed by Kozachenko and Leonenko (1987)
#     :param pixels: ndarray, lesion pixel array
#     :param k_neighbor: int, k-nearest neighbour
#     """
#     assert len(pixels.shape) == 1
#     pixels_n_dim = pixels[:, np.newaxis]
#     kozachenko_entropy = get_h(pixels_n_dim, k=k_neighbor)
#     return kozachenko_entropy


# Mask level extraction
def get_max_contour(mask: np.ndarray):
    contours, hierarchy = cv2.findContours(
        np.array(mask, dtype='uint8'),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # mask could have multiple lesion, select largest lesion contour
    contours_length = [len(contours[i]) for i in range(len(contours))]
    max_contour = contours[contours_length.index(max(contours_length))]

    return max_contour


def get_mask_roundness(mask: np.ndarray):
    """
    cv2.minEnclosingCircle: finds a circle of the minimum area enclosing a 2D point set
    cv2 returns ( center(x,y), radius)
    """
    lesion_contour = get_max_contour(mask)
    assert (len(lesion_contour.shape) == 3)
    # calculate roundness = area / (3.14*radius^2)
    center, min_circle_radius = cv2.minEnclosingCircle(lesion_contour)
    contour_area = cv2.contourArea(lesion_contour)
    assert (min_circle_radius > 0)
    roundness = contour_area / (min_circle_radius * min_circle_radius * 3.14)
    return roundness


def get_mask_enclosing_rectangle_features(mask: np.ndarray):
    """
    cv2.minAreaRect: finds a rotated rectangle of the minimum area enclosing the input 2D point set
    cv2 returns ( center(x,y), (width, height), angle of rotation)
    WARNING: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/
    py_contour_features/py_contour_features.html reference of is incorrect
    """
    lesion_contour = get_max_contour(mask)
    assert (len(lesion_contour.shape) == 3)
    center_point, width_height, rotation = cv2.minAreaRect(lesion_contour)

    max_edge = max(width_height)

    contour_center_distance = [np.linalg.norm(point[0] - center_point) for point in lesion_contour]
    distance_variance = np.var(contour_center_distance)

    # ToDo: also consider - distance min/mean/max

    feature_dict = {
        'max_edge': max_edge,
        'distance_variance': distance_variance
    }

    return feature_dict


# Image level extraction
def get_laplace_feature(image, mask, kernel, display=False):
    """
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    """
    filtered_image = signal.convolve2d(image, kernel, mode="same")
    filtered_pixels = filtered_image[np.where(mask != 0)]
    if display:
        plt.imshow(filtered_image * mask, cmap='gray')

    activation_mean = np.mean(filtered_pixels)
    activation_q95 = np.quantile(filtered_pixels, 0.95)

    return activation_mean, activation_q95

