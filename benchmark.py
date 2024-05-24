import argparse
import torch

from anomalib import TaskType
from anomalib.data import Folder, MVTec
from anomalib.data.utils import TestSplitMode
import anomalib.models as Models
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def anomalib_data(name, root, normal_dir, abnormal_dir, mask_dir, image_size, task, seed=None):
    """
    Data in anomalib format

    task: TaskType.SEGMENTATION or TaskType.CLASSIFICATION or TaskType.DETECTION
    """
    # transform = get_transforms(image_size=256, normalization=InputNormalizationMethod.NONE)
    datamodule = Folder(
        name=name,
        root=root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        seed=seed,
        task=task
    )

    datamodule.setup()
    return datamodule

def benchmark_anomalib():
    # models = {
    #     'ganomaly', 'uflow', 'dsr', 'fastflow', 'win_clip', 'csflow', 'padim', 
    #     'dfm', 'patchcore', 'reverse_distillation', 'rkde', 'efficient_ad', 
    #     'cfa', 'draem', 'dfkde', 'stfpm', 'cflow'
    # }
    models = {Models.Ganomaly, Models.Patchcore, Models.Draem}
    classes = {
        'class-01'
    }
    engine = Engine()
    for cur_class in classes:
        # Loop over the possible defect types?
        datamodule = anomalib_data(
            name=cur_class, 
            root="old-data/own-texture/", 
            normal_dir="val/good", 
            abnormal_dir="val/defect-bottom-right",
            mask_dir="val_ground_truth/defect-bottom-right", # ??
            image_size=(256, 256),
            task=TaskType.DETECTION
        )
    
        for cur_model in models:
            model = cur_model()
            engine.fit(datamodule=datamodule, model=model)
            predict = engine.predict(
                datamodule=datamodule,
                model=model
                # ckpt_path=f"results/experimental"
            )

            print(predict)

def benchmark_pad():
    pass

def benchmark_splatpose():
    pass

if __name__ == "__main__":
    benchmark_anomalib()