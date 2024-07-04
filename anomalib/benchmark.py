from anomalib import TaskType
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
import anomalib.models as Models
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC
from anomalib.deploy.inferencers import TorchInferencer
from anomalib.deploy import ExportType
from anomalib.data.utils import read_image
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from typing import Union
import sys
import argparse
import os
import json
import torch
from matplotlib import pyplot as plt

classnames = {
    "bus_coppler_green": ["defect-exchange-rails-l-cr", "defect-exchange-rails-cl-cr",
                        "defect-mount-clipper", "defect-rail-cr", "defect-rail-l"], 
    "bus_coppler_gray" : ["defect-exchange-rails-l-cr", "defect-exchange-rails-cl-cr",
                        "defect-mount-clipper", "defect-rail-cr", "defect-rail-l"],
    "2700642": ["defect-bottom-right", "defect-bottom-left", "defect-top-right", "defect-top-left"],
    "switch_8_port": ["defect-mount-clipper", "defect-connector-side"], 
    # "switch_16_port": [""],
}
SEED = 0
ROOT_DATA = ""
results = {}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
callbacks = [
    ModelCheckpoint(
        mode="max",
        monitor="pixel_AUROC",
    ),
    EarlyStopping(
        monitor="pixel_AUROC",
        mode="max",
        patience=15,
    ),
]

def train_benchmark(model_name, cur_model, task, output_file, train_batch_size=32):
    for cur_class in classnames.keys():
        model = cur_model()
        engine = Engine(task=task, callbacks=callbacks)
        cur_root = os.path.join(ROOT_DATA, cur_class)
        for i, cur_anomaly in enumerate(classnames[cur_class]):
            if task != TaskType.SEGMENTATION and task != TaskType.CLASSIFICATION: pass

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
                train_batch_size=train_batch_size
            )
            
            datamodule.setup()
            print(f"{model_name=}, {cur_class=}, {cur_anomaly=}")
            if i == 0:
                # only need to fit once to the good data instead of fitting every single time?
                engine.fit(datamodule=datamodule, model=model)
                test_results = engine.test(
                    model=model,
                    datamodule=datamodule,
                    ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
                    verbose=False
                )
                engine.export(model=model, export_type=ExportType.TORCH)

            predict = engine.predict(datamodule=datamodule, model=model, ckpt_path=f"results/{model_name}/{cur_class}/latest/weights/lightning/model.ckpt")

            aupro = AUPRO()
            auroc = AUROC()
            all_labels = []
            all_pred_labels = []
            all_masks = []
            all_pred_masks = []

            for batch in predict:
                # Loop through batches
                all_masks.extend(batch['mask'].tolist()) # mask only exists when task=TASK.SEGMENTATION
                all_pred_masks.extend(batch['pred_masks'].tolist())
                all_labels.extend(batch['label'].tolist())
                all_pred_labels.extend(batch['pred_labels'].tolist())

            all_masks = torch.tensor(all_masks, dtype=int)
            all_pred_masks = torch.tensor(all_pred_masks, dtype=torch.float32)
            all_labels = torch.tensor(all_labels, dtype=int)
            all_pred_labels = torch.tensor(all_pred_labels, dtype=torch.float32)

            class_dict = results.get(cur_class, {})
            class_dict[cur_anomaly] = {} 
            class_dict[cur_anomaly]["PIXEL AUPRO"] = aupro(all_pred_masks, all_masks).item()
            class_dict[cur_anomaly]["PIXEL AUROC"] = auroc(all_pred_masks, all_masks).item()
            class_dict[cur_anomaly]["IMAGE AUROC"] = auroc(all_pred_labels, all_labels).item()
            results[cur_class] = class_dict
            print(results)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
   
def benchmark_anomalib(model_name, model, output_file, task: Union[TaskType.SEGMENTATION, TaskType.CLASSIFICATION] = TaskType.SEGMENTATION):
    # reconstruction_models = {Models.Ganomaly, Models.Draem, Models.ReverseDistillation}
    # feature_models = {Models.Patchcore, Models.Stfpm, Models.UflowModel, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim}
    
    # Working: Patchcore (only one epoch), Padim,
    # Not working: CFlow, Csflow
    if not model:
        models = {Models.Stfpm, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim, Models.Patchcore}
    else:
        models = {model}

    for cur_model in models:
        print(f"Testing model: {model_name}", flush=True)
        train_batch_size = 32
        if model_name == "EfficientAd": train_batch_size = 1
        train_benchmark(model_name, cur_model, task, output_file, train_batch_size)

def inference(path, skip=35):

    def get_heatmap(inferencer, path, save_path):
        os.makedirs(save_path, exist_ok=True)
        files = os.listdir(path)
        for filename in files[::skip]:
            try:
                image = read_image(f"{path}/{filename}", as_tensor=True) # as_tensor is required
                predict = inferencer.predict(image)
                plt.imsave(f"{save_path}/{filename}", predict.heat_map)
            except FileNotFoundError:
                continue

    for model_name in os.listdir("results"):
        for cur_class in classnames.keys():
            inferencer = TorchInferencer(path=f"results/{model_name}/{cur_class}/latest/weights/torch/model.pt", device='auto')
            for cur_anomaly in classnames[cur_class]:
                get_heatmap(inferencer, f"{path}/{cur_class}/test/{cur_anomaly}", f"heatmap/{model_name}/{cur_class}/{cur_anomaly}")
            get_heatmap(inferencer, f"{path}/{cur_class}/train/good", f"heatmap/{model_name}/{cur_class}/good")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomalib benchmark')
    parser.add_argument('--model', type=str, help="Model to train")
    parser.add_argument('--data', type=str, help="Root data directory for Anomalib", default="/home/thomasl/tmdt-benchmark/dataset")
    parser.add_argument('--inference', action='store_true')
    args = parser.parse_args()
    output = f"results_{args.model}.json"
    if args.inference:
        inference(args.data)
    else:
        ROOT_DATA = args.data
        model = None
        try:
            model = eval(f"Models.{args.model}")
        except Exception as e:
            print("Model not valid. Use one from anomalib.")
            print(e, file=sys.stderr)
        if model:
            benchmark_anomalib(args.model, model, output)
