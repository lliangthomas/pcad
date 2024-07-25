import os
import matplotlib.pyplot as plt
import math

PARENT = "/home/thomasl/tmdt-benchmark"
save_path = "plots"
os.makedirs(save_path, exist_ok=True)

def plot_heatmap_grid(imgs):
    heatmaps_dir = f"{PARENT}/anomalib/heatmap"
    heatmaps = os.listdir(heatmaps_dir)
    labels = ["input", "Stfpm", "EAD", "Patchcore", "RD", "Padim", "Splatpose", "gt"]
    dirs = [f"{PARENT}/latest_dataset"]
    anomalib_models = {"EfficientAd", "Patchcore", "Padim", "ReverseDistillation", "Stfpm"}
    for i in heatmaps:
        if i in anomalib_models:
            dirs.append(f"{heatmaps_dir}/{i}")
    dirs.append(f"{PARENT}/bench-sp/splatpose/heatmaps/Splatpose")
    dirs.append(f"{PARENT}/latest_dataset")
    fig, ax = plt.subplots(len(imgs), len(dirs), figsize=(8, 5), dpi=300)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for j, location in enumerate(dirs):
        label = labels[j]
        for i, img in enumerate(imgs):
            gray_scale = None
            read_location = f"{location}/{img[0]}/{img[1]}/{img[2]}.png"
            if j == len(dirs) - 2:
                read_location = f"{location}/{img[0]}/{img[1]}/heatmap_{img[2]}.png"
            if j == 0:
                read_location = f"{location}/{img[0]}/test/{img[1]}/{img[2]}.png"
            if j == len(dirs) - 1:
                read_location = f"{location}/{img[0]}/ground_truth/{img[1]}/{img[2]}.png"
                gray_scale = 'gray'
            load_img = plt.imread(read_location)
            ax[i, j].imshow(load_img, cmap=gray_scale)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(label)
    fig.savefig(f"{save_path}/heatmap.png")
    plt.close(fig)

def plot_gt_img():
    dir = f"{PARENT}/0712_lighting_dataset"
    interval = 10
    classes = os.listdir(dir)
    for cur_class in classes:
        # if cur_class[:6] != "switch": continue
        print(cur_class)
        defect_path = os.path.join(dir, f"{cur_class}/ground_truth")
        defects = os.listdir(defect_path)
        for cur_defect in defects:
            # if cur_defect != "defect-exchange-rj-45": continue
            fig, ax = plt.subplots(2, interval, figsize=(6, 1.5), dpi=400)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            gt_path = os.path.join(dir, f"{cur_class}/ground_truth/{cur_defect}")
            gt_imgs = os.listdir(gt_path)
            img_path = os.path.join(dir, f"{cur_class}/test/{cur_defect}")
            print(cur_defect, len(gt_imgs))
            print(math.ceil(len(gt_imgs) / interval))
            for i in range(2):
                cur_column = 0
                count = 0
                for j in range(0, len(gt_imgs), int(len(gt_imgs) / interval)):
                    if count == 10: break
                    gray_scale = None
                    if i == 0:
                        read_location = os.path.join(gt_path, gt_imgs[j])
                        gray_scale = 'gray'
                    else:
                        read_location = os.path.join(img_path, gt_imgs[j])
                    load_img = plt.imread(read_location)
                    ax[i, cur_column].imshow(load_img, cmap=gray_scale)
                    ax[i, cur_column].set_xticks([])
                    ax[i, cur_column].set_yticks([])
                    cur_column += 1
                    count += 1
            fig.savefig(f"{save_path}/{cur_class}_{cur_defect}.png")
            plt.close(fig)

if __name__ == "__main__":
    # plot_heatmap_grid(imgs=[
    #     ("bus_coppler_green", "defect-exchange-rails-l-cr", 250), 
    #     ("bus_coppler_gray", "defect-rail-l", 275),
    #     ("switch_8_port", "defect-mount-clipper", 117),
    #     ("switch_16_port", "defect-connector-side", 137),
    #     ("2700642", "defect-bottom-left", 207)
    # ])
    plot_gt_img()