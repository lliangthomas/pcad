
import os
import wandb

from gaussian_splatting.train import *

import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from scipy.ndimage import gaussian_filter

# needed for PAD code
from easydict import EasyDict
import yaml

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm

classnames = {}

pre_parser = ArgumentParser(description="Training parameters")
pre_parser.add_argument("-k", metavar="K", type=int, help="number of pose estimation steps", default=175)
pre_parser.add_argument("-c", "-classname", metavar="c", type=str, help="current class to run experiments on",
                        default="class-01")
pre_parser.add_argument("-wandb_config", metavar="WC", type=str, help="the wandb config to use", default="None")
pre_parser.add_argument("-p", "-prefix", metavar="pf", type=str, help="prefix for the wandb run name", default="to_delete")
pre_parser.add_argument("-seed", type=int, help="seed for random behavior", default=0)
pre_parser.add_argument("-gauss_iters", type=int, help="number of training iterations for 3DGS", default=30000)
pre_parser.add_argument("-wandb", type=int, help="whether we track with wandb", default=0)
pre_parser.add_argument("-train", type=int, help="whether we train or look for a saved model", default=0)                   
pre_parser.add_argument("-v", type=int, help="verbosity", default=1)                        
pre_parser.add_argument("-data_path", type=str, help="path pointing towards the usable data set", default="/workspace/data")
pre_parser.add_argument("-skip", type=int, help="number of test images to skip", default=150)
pre_parser.add_argument("-defect", type=str, help="perform tests on one defect at a time, specify a defect that the class has here")                                

args = pre_parser.parse_args()

data_base_dir = args.data_path

config = {
    "k" : args.k,
    "classname" : args.c,
    "seed" : args.seed,
    "3dgs_iters" : args.gauss_iters,
    "prefix" : args.p,
    "wandb" : args.wandb,
    "train" : args.train,
    "data_dir" : data_base_dir,
    "verbose" : args.v != 0
}

projectname = config["prefix"]
if config["wandb"] != 0:
    run = wandb.init(project=projectname, config=config, name=f"{config['prefix']}_{config['classname']}")

data_path = os.path.join(data_base_dir, config["classname"]) # where the training dataset should be saved
result_dir = "/home/thomasl/tmdt-benchmark/gaussian-splatting/output/" + config["classname"] + "_real" # where the gaussian splatting is
data_dir = "/home/thomasl/tmdt-benchmark/0702_dataset" # where the original dataset is
# print("saving model to: ", result_dir)
os.makedirs(result_dir, exist_ok=True)

if config["train"] != 0:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    # Set up command line argument parser
    training_args = ["-w", "--eval", "-s", data_path, "-m", result_dir, "--iterations", str(config["3dgs_iters"]), "--sh_degree", "0"]
    print("training args: ", training_args)
    parser = ArgumentParser(description="3DGS Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, config["3dgs_iters"]])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[config["3dgs_iters"]])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(training_args)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, config["seed"])
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    end.record()
    torch.cuda.synchronize()
    train_time_millis = start.elapsed_time(end)
    
    if config["wandb"] != 0:
        wandb.log({
            "train_seconds" : train_time_millis / 1000
        })
        
else:
    if config["wandb"] != 0:
        wandb.log({
            "train_seconds" : 0
        })
    print("skipping training!")
    
from pose_estimation import main_pose_estimation
from utils_pose_est import ModelHelper, update_config
from aupro import calculate_au_pro_au_roc


test_images, reference_images, all_labels, gt_masks, times = main_pose_estimation(
    cur_class=config["classname"],
    model_dir_location=result_dir,
    k=config["k"],
    verbose=config["verbose"],
    data_dir=data_dir,
    skip=args.skip,
    separate_by_defect=args.defect
)

if config["wandb"] != 0:
    my_data = [[i, times[i]] for i in range(len(times))]
    columns = ["index", "time_millis"]
    cur_table = wandb.Table(data=my_data, columns=columns)
    wandb.log({"time_millis": cur_table})

with open("PAD_utils/config_effnet.yaml") as f:
    mad_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
mad_config = update_config(mad_config)
model = ModelHelper(mad_config.net)
model.eval()
model.cuda()


# evaluation Code taken from PAD/MAD data set paper at https://github.com/EricLee0224/PAD
criterion = torch.nn.MSELoss(reduction='none')
tf_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

tf_mask = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    ])

test_imgs = list()
score_map_list=list()
scores=list()
pred_list=list()
recon_imgs=list()
with torch.no_grad():
    for i in range(len(test_images)):
        ref=tf_img(reference_images[i]).unsqueeze(0).cuda()
        rgb=tf_img(test_images[i]).unsqueeze(0).cuda()
        ref_feature=model(ref)
        rgb_feature=model(rgb)
        score = criterion(ref, rgb).sum(1, keepdim=True)
        for i in range(len(ref_feature)):
            s_act = ref_feature[i]
            mse_loss = criterion(s_act, rgb_feature[i]).sum(1, keepdim=True)
            score += torch.nn.functional.interpolate(mse_loss, size=224, mode='bilinear', align_corners=False)

        score = score.squeeze(1).cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=4)
        recon_imgs.extend(rgb.cpu().numpy())
        test_imgs.extend(ref.cpu().numpy())
        scores.append(score)

# scores = np.asarray(scores).squeeze()
# max_anomaly_score = scores.max()
# min_anomaly_score = scores.min()
# scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score) 
# gt_mask = np.concatenate([np.asarray(tf_mask(a))[None,...] for a in gt_masks], axis=0)
# gt_mask[gt_mask == 255] = 1
# results = {args.c: {}}
# classes = os.listdir(os.path.join(data_dir, f"{args.c}/ground_truth"))
# num_classes = len(classes) if args.defect else 1
# num_per_class = int(288 / args.skip)
# num_good_mask = int(288 / (args.skip / 4))
# cur_good_mask = num_good_mask
# good_mask_per_class = int(num_good_mask / num_classes)

# gt_mask = torch.tensor(gt_mask)
# scores = torch.tensor(scores)

# for i in range(0, len(gt_masks) - num_good_mask, num_per_class):
#     cur_gt_mask = np.concatenate((gt_mask[i:i + num_per_class], gt_mask[cur_good_mask:cur_good_mask + good_mask_per_class]))
#     cur_score = np.concatenate((scores[i:i + num_per_class], scores[cur_good_mask:cur_good_mask + good_mask_per_class]))
#     cur_good_mask += good_mask_per_class

#     per_pixel_rocauc = roc_auc_score(cur_gt_mask.flatten(), cur_score.flatten())
#     # print('PIXEL ROCAUC: %.3f' % (per_pixel_rocauc))
#     print(cur_gt_mask)
#     print(cur_score)
#     au_pro, au_roc, pro_curve, roc_curve = calculate_au_pro_au_roc(cur_gt_mask, cur_score)

#     img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
#     gt_list_isano = np.asarray(all_labels) != 0
#     img_roc_auc = roc_auc_score(gt_list_isano, img_scores)
#     # print('IMAGE ROCAUC: %.3f' % (img_roc_auc))
    
#     class_dict = {}
#     cur_anomaly = classes[i // num_per_class]
#     class_dict[cur_anomaly] = {} 
#     class_dict[cur_anomaly]["PIXEL AUPRO"] = au_pro
#     class_dict[cur_anomaly]["PIXEL AUROC"] = per_pixel_rocauc
#     class_dict[cur_anomaly]["IMAGE AUROC"] = img_roc_auc
#     results[args.classname] = class_dict

scores = np.asarray(scores).squeeze()
max_anomaly_score = scores.max()
min_anomaly_score = scores.min()
scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

plt.imsave(fname="sp_heatmap.png", arr=scores[0])
print(scores[0].shape)
transform_img = cv2.resize(test_images[0].permute(1, 2, 0).numpy(), dsize=(224, 224))
# gray_image_3ch = np.stack((scores[0], scores[0], scores[0]), axis=-1)
gray_image_3ch = cm.viridis(scores[0])[..., :3].astype(np.float32)
print(transform_img.dtype)
heatmap_overlay_img = cv2.addWeighted(src1=gray_image_3ch, alpha=0.5, src2=transform_img, beta=0.5, gamma=0) 
plt.imsave(fname="sp_heatmap_overlay.png", arr=heatmap_overlay_img)

gt_mask = np.concatenate([np.asarray(tf_mask(a))[None,...] for a in gt_masks], axis=0)
gt_mask[gt_mask == 255] = 1
precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
a = 2 * precision * recall
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]

fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

au_pro, au_roc, pro_curve, roc_curve = calculate_au_pro_au_roc(gt_mask, scores)
print(f"aupro: {au_pro}. and other au_roc: {au_roc}")

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list_isano = np.asarray(all_labels) != 0
img_roc_auc = roc_auc_score(gt_list_isano, img_scores)
print('image ROCAUC: %.3f' % (img_roc_auc))

if config["wandb"] != 0:
    wandb.log({
        "avg_time" : np.mean(times),
        "pixel_roc" : per_pixel_rocauc,
        "image_roc" : img_roc_auc,
        "aupro" : au_pro
    })


