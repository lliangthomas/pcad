import json
import os
import shutil

PARENT = "/home/thomasl/tmdt-benchmark/0702_dataset"
ORIG_PREFIX = f"{PARENT}/bus_coppler_gray"
NEW_PREFIX = ORIG_PREFIX
# NEW_PREFIX = f"{PARENT}/switch_8_port"
CAM_ANGLE = 1.1091441906486563

def split(data, start, end, name):
    split_data = {'camera_angle_x': CAM_ANGLE, 'frames': []}
    os.makedirs(f"{NEW_PREFIX}/{name}", exist_ok=True)
    for i in range(start, end):
        split_data['frames'].append({
            'file_path': f"{name}/{i}",
            'transform_matrix': data['frames'][i]['transform_matrix']
        })
        shutil.copy(src=f"{ORIG_PREFIX}/images/render_{i}.png", dst=f"{NEW_PREFIX}/{name}/{i}.png")
    with open(f"{NEW_PREFIX}/transforms_{name}.json", 'w') as outputf:
        json.dump(split_data, outputf, indent=4)

def nerf_data():
    with open(f'{ORIG_PREFIX}/transforms.json', 'r') as inputf:
        data = json.load(inputf)
    files = os.listdir(f'{ORIG_PREFIX}/images')

    train_split = int(len(files) * 0.8)
    val_split = train_split + int(len(files) * 0.1)
    test_split = len(files)

    print(train_split, val_split, test_split)

    # print(files)
    # new_files = []
    # for i in files:
    #     num = int(i.split('_')[-1].split('.')[0])
    #     num = f"{num:03}"
    #     new_files.append(num)
    # files = new_files

    split(data, 0, train_split, "train")
    split(data, train_split, val_split, "val")
    split(data, val_split, len(files), "test")

def rename():
    with open(f'{ORIG_PREFIX}/old.json', 'r') as inputf:
        data = json.load(inputf)
    files = os.listdir(f'{ORIG_PREFIX}/train/good')
    split_data = {'camera_angle_x': CAM_ANGLE, 'frames': []}
    name = "train/good"
    for i in range(len(files)):
        print(i)
        split_data['frames'].append({
            'file_path': f"{name}/{i:03d}",
            'transform_matrix': data[str(i)]['transform_matrix']
        })
    with open(f"{NEW_PREFIX}/transforms_train.json", 'w') as outputf:
        json.dump(split_data, outputf, indent=4)

def mad_rename(dir):
    files = os.listdir(dir)
    files.sort()
    for i in range(len(files)):
        os.rename(os.path.join(dir, files[i]), os.path.join(dir, f"{i:03d}.png"))

for j in ["ground_truth", "test"]:
    for i in ["Burrs", "Missing", "Stains"]:
        mad_rename(f"MAD-Sim/04Turtle/{j}/{i}")

# def resize():
#     for root, dirs, files in os.walk(ORIG_PREFIX):
#         for file in files:
#             if file[-4:] == ".png":
#                 full_path = os.path.join(root, file)
#                 img = cv2.imread(full_path)
#                 resized = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
#                 if file[-8:] == "mask.png":
#                     print(file)
#                     resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#                 cv2.imwrite(full_path, resized)

# def mask_resize():
#     upper_left = (1090, 0)
#     bottom_right = (upper_left[0] + 2048, upper_left[1] + 2138)
#     files = os.listdir(ORIG_PREFIX)

#     for f in files:
#         full_path = os.path.join(ORIG_PREFIX, f)
#         img = cv2.imread(full_path)
#         resized = img[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
#         resized = cv2.resize(resized, (800, 800), interpolation=cv2.INTER_AREA)
#         resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

#         new_name = f"{int(f.split('.')[0].split('=')[-1]):03}_mask.png"
#         cv2.imwrite(os.path.join(NEW_PREFIX, new_name), resized)