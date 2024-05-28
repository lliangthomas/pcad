 
import os
import json
import numpy as np
from sklearn.metrics import pairwise_distances
from PIL import Image
import cv2

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def generate_samples(n_samples, reference_translations, distance_factor=0.8, deviation_factor=4):
    """
    Generates new points by:
        - Calculating the middle of all reference translation positions
        - Estimates the average distance from the middle to all other points
        - According to estimates, samples n new distances, which are placed closer to the middle than the reference
        - Samples n direction vectors in all possible directions
        - Each sample marches from the middle in the sampled direction until the sampled distance is reached
    """
    middle = np.mean(reference_translations, axis=0)
    # distances from middle to train points
    distances = np.linalg.norm(reference_translations - middle, axis=1)
    mu, sigma = np.mean(distances), np.std(distances)
    print(f"Sampling with mu: {mu:1.3f} * {distance_factor} = {(mu * distance_factor):1.3f} " \
            f"and sigma: {sigma:1.3f} * {deviation_factor} = {(sigma * deviation_factor):1.3f}")
    # sampled distances of our new points to the middle
    sample_dists = np.random.normal(mu * distance_factor,
                                    sigma * deviation_factor, size=n_samples)    
    # random directional vectors in R^3  
    direction = (np.random.rand(n_samples, 3) * 2) - 1
    dist_to_dir = np.linalg.norm((middle - (middle + direction)), axis=1)
    direction_multiplier = sample_dists / dist_to_dir
    # march from middle_point in the sampled direction until distance is reached
    new_points = middle + np.broadcast_to(direction_multiplier[...,None], (n_samples, 3)) * direction
    return new_points


k_augments = 3
result_path = "processed_data"
base_path = "data"

split_train = 1.
os.makedirs(result_path, exist_ok=True)
classnames = {"class-01"}

for cl in classnames:
    class_path = os.path.join(result_path, cl)
    os.makedirs(class_path, exist_ok=True)

    orig_train_dir = os.path.join(base_path, cl, "train", "good")
    train_dir = os.path.join(class_path, "train")
    os.makedirs(new_train_dir, exist_ok=True)

    test_dir = os.path.join(class_path, "test")
    os.makedirs(new_test_dir, exist_ok=True)