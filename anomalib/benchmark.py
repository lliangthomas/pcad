from anomalib import TaskType
from anomalib.data import Folder
import anomalib.models as Models
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC
from anomalib.deploy.inferencers import TorchInferencer
from anomalib.deploy import ExportType
from anomalib.data.utils import read_image
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from aupro import compute_classification_roc, trapezoid

from typing import Union
import sys
import argparse
import os
import json
import numpy as np
import torch
from matplotlib import pyplot as plt

classnames = {
    # "02Unicorn": ["Burrs", "Missing", "Stains"],
    # "03Mallard": ["Burrs", "Missing", "Stains"],
    # "04Turtle": ["Burrs", "Missing", "Stains"]
    # "bus_coppler_green": ["defect-exchange-rails-l-cr", "defect-exchange-rails-cl-cr",
                        # "defect-mount-clipper", "defect-rail-cr", "defect-rail-l"], 
    # "bus_coppler_gray" : ["defect-exchange-rails-l-cr", "defect-exchange-rails-cl-cr",
                        # "defect-mount-clipper", "defect-rail-cr", "defect-rail-l"],
    # "2700642": ["defect-bottom-right", "defect-bottom-left", "defect-top-right", "defect-top-left"],
    "switch_8_port": ["defect-exchange-rj-45"], # "defect-mount-clipper", "defect-connector-side",
    # "switch_16_port": ["defect-mount-clipper", "defect-connector-side", "defect-exchange-rj-45"], # 
}

anomalib_models = {"EfficientAd", "Patchcore", "Padim", "ReverseDistillation", "Stfpm"}

SEED = 0
ROOT_DATA = ""
results = {}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_benchmark(model_name, cur_model, task, output_file, train=False, train_batch_size=32):
    for cur_class in classnames.keys():
        callbacks = [
            ModelCheckpoint(
                mode="max",
                monitor="pixel_AUROC",
                dirpath=f"results/{model_name}/{cur_class}"
            ),
            EarlyStopping(
                monitor="pixel_AUROC",
                mode="max",
                patience=10,
            ),
        ]\
        
        model = cur_model()
        engine = Engine(task=task, callbacks=callbacks, pixel_metrics="AUROC")
        cur_root = os.path.join(ROOT_DATA, cur_class)
        for i, cur_anomaly in enumerate(classnames[cur_class]):
            if task != TaskType.SEGMENTATION and task != TaskType.CLASSIFICATION: pass
            
            print(f"{model_name=}, {cur_class=}, {cur_anomaly=}")

            # the following code resolves the issue of any disparity
            # between the ground truth masks and test images

            # test_images = os.path.join(cur_root, "test", cur_anomaly)
            # a = set(os.listdir(test_images))
            # test_gt = os.path.join(cur_root, "ground_truth", cur_anomaly)
            # b = set(os.listdir(test_gt))
            # diff1 = a - b
            # diff2 = b - a
            # for i in diff1:
            #     os.remove(os.path.join(test_images, i))
            # for i in diff2:
            #     os.remove(os.path.join(test_gt, i))

            datamodule = Folder(
                name=cur_class,
                root=cur_root,
                normal_dir="train/good",
                abnormal_dir=f"test/{cur_anomaly}",
                mask_dir=f"ground_truth/{cur_anomaly}",
                # image_size=(224, 224),
                # seed=SEED,
                task=task,
                train_batch_size=train_batch_size,
                # test_split_ratio=0.99
            )
            
            datamodule.setup()
            # for i, data in enumerate(datamodule.test_dataloader()):
            #     print(data['image_path'])
            # exit(0)
            if train and i == 0:
                # only need to fit once to the good data instead of fitting every single time
                engine.fit(datamodule=datamodule, model=model, ckpt_path=None)
                # test_results = engine.test(
                #     model=model,
                #     datamodule=datamodule,
                #     ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
                #     verbose=False
                # )
                engine.export(model=model, export_type=ExportType.TORCH)

            predict = engine.predict(
                datamodule=datamodule, 
                model=model,
                ckpt_path=f"results/{model_name}/{cur_class}/latest/weights/lightning/model.ckpt"
                # ckpt_path="best"
            )

            aupro = AUPRO()
            auroc = AUROC()
            all_labels = []
            all_pred_labels = []
            all_masks = []
            all_pred_masks = []

            for batch in predict:
                # print(batch.keys())
                # Loop through batches
                all_masks.extend(batch['mask'].tolist()) # mask only exists when task=TASK.SEGMENTATION
                all_pred_masks.extend(batch['anomaly_maps'].tolist())
                # labels = []
                # gt_labels = []
                # for i in range(len(batch['pred_masks'])):
                #     labels.append(int(torch.any(batch['anomaly_maps'][i]).item()))
                #     gt_labels.append(int(torch.any(batch['mask'][i]).item()))
                # print(f"{labels=}")
                # print(f"{gt_labels=}")
                
                # print(gt_labels)
                # labels = torch.tensor(labels, dtype=torch.int32)
                # print(batch['pred_scores'])
                # print(labels)
                # print(labels - batch['pred_labels'].int())
                # print((batch['mask']))
                # print(batch['pred_labels'])
                # print(batch['label'])
                # exit(0)
                # plt.imsave("pred_mask.png", )
                # plt.imsave("gt_mask.png", batch['mask'][0])
                all_labels.extend(batch['label'].tolist())
                all_pred_labels.extend(batch['pred_scores'].tolist())
                # print(batch['anomaly_maps'][0].shape)
                plt.imsave(fname="pred_mask.png", arr=batch['anomaly_maps'][-1].squeeze())
                plt.imsave(fname="gt_mask.png", arr=batch['mask'][-1].squeeze())
                # print(f"{batch['label']=}")
                # print(f"{batch['pred_labels']=}")
                # print(f"{batch['pred_scores']=}")
                # exit(0)

            all_masks = torch.tensor(all_masks, dtype=torch.long)
            all_pred_masks = torch.tensor(all_pred_masks, dtype=torch.float32)
            all_labels = torch.tensor(all_labels, dtype=torch.long)
            all_pred_labels = torch.tensor(all_pred_labels, dtype=torch.float32)

            class_dict = results.get(cur_class, {})
            class_dict[cur_anomaly] = {}
            class_dict[cur_anomaly]["PIXEL AUPRO"] = aupro(all_pred_masks, all_masks).item()
            class_dict[cur_anomaly]["PIXEL AUROC"] = auroc(all_pred_masks, all_masks).item()
            class_dict[cur_anomaly]["IMAGE AUROC"] = auroc(all_pred_labels, all_labels).item()
            # roc_curve = compute_classification_roc(
            #     anomaly_maps=all_pred_labels.numpy(),
            #     scoring_function=np.max,
            #     ground_truth_labels=all_labels.numpy())
            # print(roc_curve)
            # au_roc = trapezoid(roc_curve[0], roc_curve[1])
            # print(au_roc)
            # print(all_labels)
            # print(all_pred_labels)
            results[cur_class] = class_dict
            # print(results)
            # exit(0)
        # print(results)
        # exit(0)

    # exit(0)
    results["METHOD"] = model_name
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def heatmap(path, skip, output_name):

    def get_heatmap(inferencer, path, save_path):
        os.makedirs(save_path, exist_ok=True)
        files = os.listdir(path)
        for filename in files[::skip]:
            try:
                image = read_image(f"{path}/{filename}", as_tensor=True) # as_tensor is required
                predict = inferencer.predict(image)
                plt.imsave(f"{save_path}/{filename}", predict.heat_map)
                plt.imsave(f"{save_path}/{filename.split('.')[0]}_predmask.png", predict.pred_mask)
            except FileNotFoundError:
                continue

    for model_name in os.listdir("results"):
        print(f"Generating heatmaps for {model_name}")
        if model_name not in anomalib_models: continue
        for cur_class in classnames.keys():
            inferencer = TorchInferencer(path=f"results/{model_name}/{cur_class}/latest/weights/torch/model.pt", device='auto')
            for cur_anomaly in classnames[cur_class]:
                get_heatmap(inferencer, f"{path}/{cur_class}/test/{cur_anomaly}", save_path=f"heatmap_{output_name}/{model_name}/{cur_class}/{cur_anomaly}")
            get_heatmap(inferencer, f"{path}/{cur_class}/train/good", save_path=f"heatmap_{output_name}/{model_name}/{cur_class}/good")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomalib benchmark')
    parser.add_argument('--model', type=str, help="Model to train")
    parser.add_argument('--data', type=str, help="Root data directory for Anomalib")
    parser.add_argument('--heatmap', action='store_true')
    parser.add_argument('--train', action='store_true', help="Determines if the model should train a model per object class. If not, it will assume that a pre-trained model exists and can be used")
    parser.add_argument('--skip', type=int, default=30)
    parser.add_argument('--output_name', type=str)
    args = parser.parse_args()

    if args.heatmap:
        heatmap(args.data, args.skip, args.output_name)
    else:
        ROOT_DATA = args.data
        cur_model = None
        train_batch_size = 32
        # try:
        #     cur_model = eval(f"Models.{args.model}")
        # except Exception as e:
        #     print("Model not valid. Use one from anomalib.")
        #     print(e, file=sys.stderr)
        #     exit(0)
        # model = args.model
        for model in anomalib_models:
            cur_model = eval(f"Models.{model}")
            if args.model == "EfficientAd": train_batch_size = 1
            train_benchmark(
                model_name=model, 
                cur_model=cur_model, 
                task = TaskType.SEGMENTATION, 
                output_file=f"results_{model}_{args.output_name}.json", 
                train=args.train,
                train_batch_size=train_batch_size
            )
