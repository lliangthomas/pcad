from anomalib import TaskType
from anomalib.data import Folder, MVTec
from anomalib.data.utils import TestSplitMode
import anomalib.models as Models
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC

import sys
import argparse

classnames = {
              "bus_coppler_green": ["defect-exchange-rails-cl-cr", "defect-exchange-rails-l-cr",
                                    "defect-mount-clipper", "defect-rail-cr", "defect-rail-l"], 
            #   "router_firewall": ["defect-bottom-right", "defect-bottom-left", "defect-top-right", "defect-top-left"],
            #   "switch_8_port": [""], 
            #   "switch_16_port": [""], 
            #   "bus_coppler_gray" : [""]
}
SEED = 0

def benchmark_model(model, task=TaskType.CLASSIFICATION, train_batch_size=32):
    engine = Engine()
    for cur_class in classnames.keys():
        for cur_anomaly in classnames[cur_class]:
            if task == TaskType.SEGMENTATION or task == TaskType.CLASSIFICATION:
                print(f"Class: {cur_class}, Anomaly: {cur_anomaly}")
                datamodule = Folder(name=cur_class,
                                    root=f"/workspace/data/{cur_class}",
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
                print(len(predict)) 
                for i in range(len(predict)):
                    print(predict[i].keys())
                #print(predict[1])
                i, test_data = next(enumerate(datamodule.test_dataloader()))
                print(test_data.keys())
                print(test_data['label'])
                aupro = AUPRO()
                #print("IMAGE AUPRO: ", aupro(test_data['label'], predict['label']))
                auroc = AUROC()
                #print("IMAGE AUROC: ", auroc(test_data['label'], predict['label']))
                return



def benchmark_anomalib(model=None, task = TaskType.CLASSIFICATION):
    print("Benchmark Anomalib")

    # reconstruction_models = {Models.Ganomaly, Models.Draem, Models.ReverseDistillation}
    # feature_models = {Models.Patchcore, Models.Stfpm, Models.UflowModel, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim}
    
    # Working: Patchcore, Padim
    # Not working: CFlow, Csflow
    if not model:
        models = {Models.Stfpm, Models.Cflow, Models.EfficientAd, Models.Csflow, Models.Padim, Models.Patchcore}
    else:
        models = {model}

    for cur_model in models:
        model_name = str(cur_model)[7:]
        print(f"Testing model: {model_name}", flush=True)
        model = cur_model()
        benchmark_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Anomalib benchmark')
    parser.add_argument('--model', dest='model', type=str, help="Model to train.")

    args = parser.parse_args()
    model = None
    try:
        model = eval(f"Models.{args.model}")
    except Exception as e:
        print("Model not valid. Use one from anomalib.")
        print(e, file=sys.stderr)
    if model:
        benchmark_anomalib(model)
