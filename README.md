# tmdt-benchmark

docker run --gpus all -v /home/thomasl/tmdt-benchmark/data:/workspace/data -v /home/thomasl/tmdt-benchmark/pad/git-pad:/workspace/pad -v /home/thomasl/tmdt-benchmark/colmap-output:/workspace/colmap-output --rm -it --ipc=host dromni/nerfstudio:1.1.0

ns-process-data images --data data/class-01/train/good --output-dir colmap-output

docker run --rm -it --gpus all -v /home/thomasl/tmdt-benchmark/data:/workspace/data test-nerfdocker