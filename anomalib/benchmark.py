from anomalib import TaskType
from anomalib.data import Folder, MVTec
from anomalib.data.utils import TestSplitMode
import anomalib.models as Models
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC

import sys
import argparse
import os
import torch

classnames = {
              "bus_coppler_green": ["defect-exchange-rails-l-cr", "defect-exchange-rails-cl-cr",
                                    "defect-mount-clipper", "defect-rail-cr", "defect-rail-l"], 
              "bus_coppler_gray" : ["defect-exchange-rails-l-cr", "defect-exchange-rails-cl-cr",
                                    "defect-mount-clipper", "defect-rail-cr", "defect-rail-l"]
            #   "router_firewall": ["defect-bottom-right", "defect-bottom-left", "defect-top-right", "defect-top-left"],
            #   "switch_8_port": [""], 
            #   "switch_16_port": [""], 
}
SEED = 0
ROOT_DATA = ""
results = {}

def benchmark_model(model_name, model, train_batch_size=32, task=TaskType.CLASSIFICATION):
    for cur_class in classnames.keys():
        engine = Engine()
        for cur_anomaly in classnames[cur_class]:
            # if cur_anomaly == "defect-rail-cr": pass
            if task == TaskType.SEGMENTATION or task == TaskType.CLASSIFICATION:
                print(f"Class: {cur_class}, Anomaly: {cur_anomaly}")
                datamodule = Folder(name=cur_class,
                                    root=os.path.join(ROOT_DATA, cur_class),
                                    normal_dir="train/good",
                                    abnormal_dir=f"test/{cur_anomaly}",
                                    mask_dir=f"ground_truth/{cur_anomaly}",
                                    image_size=(224, 224),
                                    seed=SEED,
                                    task=task,
                                    train_batch_size=train_batch_size)
                datamodule.setup()
                engine.fit(datamodule=datamodule, model=model)
                predict = engine.predict(datamodule=datamodule, model=model)
                                         # ckpt_path=f"results/experimental")
                aupro = AUPRO()
                auroc = AUROC()
                all_labels = []
                all_pred_labels = []
                all_masks = []
                all_pred_masks = []
                # print(predict[0].keys())

                for batch in predict:
                    # Loop through batches
                    all_masks.extend(batch['mask'].tolist())
                    all_pred_masks.extend(batch['pred_masks'].tolist())
                    all_labels.extend(batch['label'].tolist())
                    # print(batch['label'])
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


def benchmark_anomalib(model_name, model=None, task=TaskType.CLASSIFICATION):
    # reconstruction_models = {Models.Ganomaly, Models.Draem, Models.ReverseDistillation}
    # feature_models = {Models.Patchcore, Models.Stfpm, Models.UflowModel, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim}
    
    # Working: Patchcore, Padim
    # Not working: CFlow, Csflow
    if not model:
        models = {Models.Stfpm, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim, Models.Patchcore}
    else:
        models = {model}

    for cur_model in models:
        print(f"Testing model: {model_name}", flush=True)
        model = cur_model()
        train_batch_size = 32
        if model_name == "EfficientAd":
            train_batch_size = 1
        benchmark_model(model_name, model, train_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomalib benchmark')
    parser.add_argument('--model', dest='model', type=str, help="Model to train.")
    parser.add_argument('--data', dest='root_data', type=str, help="Root data directory for anomalib")

    args = parser.parse_args()
    model = None
    ROOT_DATA = args.root_data

    try:
        model = eval(f"Models.{args.model}")
    except Exception as e:
        print("Model not valid. Use one from anomalib.")
        print(e, file=sys.stderr)
    if model:
        benchmark_anomalib(args.model, model)
