import json
import os
import shutil

PARENT = "/data/tliang/tmdt-benchmark"
ORIG_PREFIX = f"{PARENT}/orig-data/mguard"
NEW_PREFIX = f"{PARENT}/data/mguard"

# def nerf_data():
    # with open(f'{ORIG_PREFIX}/transforms.json', 'r') as inputf:
    #     data = json.load(inputf)
#     files = os.listdir(f'{ORIG_PREFIX}/train/good')
#     files.sort()
#     train_split = int(len(files) * 0.8)
#     val_split = train_split + int(len(files) * 0.1)
#     test_split = len(files)

#     print(train_split, val_split, test_split)

#     def split(start, end, name):
#         split_data = {'frames': [], 'camera_angle_x': 63.3536538458742}
#         for i in range(start, end):
#             split_data['frames'].append({
#                 'file_path': f"{name}/{files[i][:-4]}",
#                 'transform_matrix': data['frames'][i]['transform_matrix']
#             })
#             # if name != "train":
#                 # shutil.move(src=f"{NEW_PREFIX}/train/{files[i]}", dst=f"{NEW_PREFIX}/{name}/{files[i]}")
#         with open(f"{NEW_PREFIX}/transforms_{name}.json", 'w') as outputf:
#             json.dump(split_data, outputf, indent=2)

#     split(0, train_split, "train")
#     split(train_split, val_split, "val")
#     split(val_split, len(files), "test")

# nerf_data()

def data():
    with open(f'{ORIG_PREFIX}/transforms.json', 'r') as inputf:
        data = json.load(inputf)
    files = os.listdir(f'{ORIG_PREFIX}/train/good')
    files.sort()
    for i in range(len(files)):
        data['frames'][i]['file_path'] = f'test/good/{files[i][:-4]}'

    with open(f'{NEW_PREFIX}/transforms_test.json', 'w') as outputf:
        json.dump(data, outputf, indent=2)

# dest = f"{ORIG_PREFIX}/ground_truth/defect-bottom-right"
# mask_files = os.listdir(dest)
# # mask_files.sort(key=lambda x:x[])

# for i in range(len(mask_files)):
#     old = mask_files[i]
#     frame_number = int(old.split('=')[1].split('.')[0])
#     new_name = f"{frame_number:03}_mask.png"

#     shutil.move(src=f"{dest}/{old}", dst=f"{dest}/{new_name}")
