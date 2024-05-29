 
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
    print(middle)
    print(direction)
    dist_to_dir = np.linalg.norm((middle - (middle + direction)), axis=1)
    direction_multiplier = sample_dists / dist_to_dir
    # march from middle_point in the sampled direction until distance is reached
    new_points = middle + np.broadcast_to(direction_multiplier[...,None], (n_samples, 3)) * direction
    return new_points

PARENT = "/home/thomasl"
k_augments = 3
result_path = f"{PARENT}/tmdt-benchmark/processed_data"
base_path = f"{PARENT}/tmdt-benchmark/data"

split_train = 1.
os.makedirs(result_path, exist_ok=True)
classnames = {"class-01"}

for cl in classnames:
    class_path = os.path.join(result_path, cl)
    os.makedirs(class_path, exist_ok=True)

    orig_train_dir = os.path.join(base_path, cl, "train", "good")
    train_dir = os.path.join(class_path, "train")
    os.makedirs(train_dir, exist_ok=True)

    test_dir = os.path.join(class_path, "test")
    os.makedirs(test_dir, exist_ok=True)

    train_samples = len(os.listdir(orig_train_dir))

    train_index = np.random.choice(train_samples, int(train_samples * split_train), replace=False)
    test_index = []

    train_samples = os.listdir(orig_train_dir)

    # Discard transparent images

    training_poses = []

    with open(os.path.join(base_path, cl, "transforms.json"), "r") as f:
        train_transforms = json.load(f)
    camera_angle_x = train_transforms["camera_angle_x"]

    # # refactor training filepaths
    for frame in train_transforms["frames"]:
    #     cur_num = int(frame["file_path"].split("/")[-1].split(".")[0])
    #     frame["file_path"] = f"./train/train_{cur_num:03d}"
        training_poses.append(np.array(frame["transform_matrix"])[None,...])
    
    # dump training poses back to the data set
    with open(os.path.join(class_path, "transforms_train.json"), "w") as f:
        json.dump(train_transforms, f, indent=2)

    # gather training pose information
    training_poses = np.concatenate(training_poses, axis=0)
    all_translations = training_poses[:,:3,3]
    all_rotations = training_poses[:,:3,:3]
    mean_point = np.mean(all_translations, axis=0)

    # generate test poses (translation vectors for now)
    test_translations = generate_samples(k_augments, all_translations)
    test_translations_2 = generate_samples(k_augments, all_translations, 1.2)
    test_translations = np.concatenate((test_translations, test_translations_2), axis=0)
    # for each test sample grab the closest rotations and translations in the trainset
    distances = pairwise_distances(X=test_translations, Y=all_translations).argmin(axis=1)
    closest_translations = all_translations[distances]
    closest_rotations = all_rotations[distances]

    test_transforms = {
        "camera_angle_x" : camera_angle_x,
        "frames" : []
    }
    test_poses = np.zeros((test_translations.shape[0], 4, 4))

    empty_image = Image.fromarray(np.ones((800,800), dtype=np.uint8) * 255)
    # print(test_translations)
    for idx in range(test_translations.shape[0]):

        # calculate rotation from given translation
        cur_vec = test_translations[idx]
        base_vec = closest_translations[idx] - mean_point
        # NOTE: original procedure where we first rotate /w nearest available rotation and then to our point from
        #       there to our desired point
        rot = closest_rotations[idx]
        rot = rotation_matrix_from_vectors(base_vec, cur_vec)
        rot = rot @ closest_rotations[idx]
        
        test_poses[idx, 3,3] = 1
        test_poses[idx, :3,3] = cur_vec
        test_poses[idx, :3,:3] = rot

        # print(test_poses)
        # append to transforms json
        test_transforms["frames"].append(
            {
                "file_path" : f"./test/test_{idx:03d}",
                "transform_matrix" : test_poses[idx].tolist()
            }
        )
        empty_image.save(os.path.join(class_path, "test", f"test_{idx:03d}.png"))

    # dump the json
    with open(os.path.join(class_path, "transforms_test.json"), "w") as f:
        json.dump(test_transforms, f, indent=2)