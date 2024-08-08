# tmdt-benchmark

Instructions to run the experiments:

## Anomalib

- Install Anomalib and other packages such as matplotlib
- Modify variables in anomalib/benchmark.py (classnames)
- Train: `python benchmark.py --data <path to data> --train --output <name of output file>`
- Inference: `python benchmark.py --data <path to data> --skip <number of images to skip in experiments> --output <name of output file> --heatmap`

## SplatPose

- 3D Gaussian Splatting renders which can be made using the tool here: https://github.com/graphdeco-inria/gaussian-splatting
- Once the renders are made for each of the classes, you'll need to modify some variables in bench-sp/splatpose/train_and_render.py (result_dir, data_dir, classnames, output_file)
- `cd bench-sp/splatpose && python train_and_render.py -skip <number of images to skip in experiments>`
