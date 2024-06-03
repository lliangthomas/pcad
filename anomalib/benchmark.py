from anomalib import TaskType
from anomalib.data import Folder, MVTec
from anomalib.data.utils import TestSplitMode
import anomalib.models as Models
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC

from dataclasses import dataclass
from typing import List

classnames = {
            "router_firewall": ["defect-bottom-right", "defect-bottom-left", "defect-top-right", "defect-top-left"],
            #   "switch_8_port": [""], 
            #   "switch_16_port": [""], 
            #   "bus_coppler_green": [""], 
            #   "bus_coppler_gray" : [""]
}
SEED = 0

def benchmark_model(model, task=TaskType.CLASSIFICATION):
    engine = Engine()
    for cur_class in classnames.keys():
        for cur_anomaly in classnames[cur_class]:
            if task == TaskType.SEGMENTATION or task == TaskType.CLASSIFICATION:
                datamodule = Folder(name=cur_class,
                                    root=f"/workspace/data/{cur_class}",
                                    normal_dir="train/good",
                                    abnormal_dir=f"test/{cur_anomaly}",
                                    mask_dir=f"ground_truth/{cur_anomaly}",
                                    image_size=(800, 800),
                                    seed=SEED,
                                    task=task)
                datamodule.setup()
                engine.fit(datamodule=datamodule, model=model)
                predict = engine.predict(datamodule=datamodule,
                                         model=model)
                                         # ckpt_path=f"results/experimental")
                print(predict)

            else:
                print("Nothing to do ")


def benchmark_anomalib(model=None, task = TaskType.CLASSIFICATION):
    print("Benchmark Anomalib")

    # reconstruction_models = {Models.Ganomaly, Models.Draem, Models.ReverseDistillation}
    # Models.UflowModel might not work
    # feature_models = {Models.Patchcore, Models.Stfpm, Models.UflowModel, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim}
    if not model:
        models = {Models.Stfpm, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim, Models.Patchcore}
    else:
        models = {model}
    for cur_model in models:
        print(f"Testing model: {str(cur_model)[7:]}")
        model = cur_model()
        benchmark_model(model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Anomalib benchmark')
    parser.add_argument('--model', dest='model', type=str, help="Model to train.")

    args = parser.parse_args()
    model = None
    try:
        model = eval(f"Models.{args.model}")
    except Exception as e:
        print("Model not valid. Use one of from anomalib.")
        print(e)
    if model:
        benchmark_anomalib(model)
