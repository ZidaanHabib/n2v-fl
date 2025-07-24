# N2V-FL

## Getting started
Create a conda environment using the required evironment.yaml file provided ie. "conda create env -f environment.yaml"

## Model training
Add a configuration file in src/config/ directory, using the default.yaml as a template. Adjust values as you see fit. To start training, simply run "python src/train.py --config-name=<your new config file>. 

During training, the model params will get stored to the checkpoints folder. Training and test losses (per epoch) will be be written to the 'epoch_losses.txt' file in src/output/current/.

## Benchmark
A benchmark file has been provided used for the purpose of identifying the best permutation of batch_size, patch_size and number of workers across a distributed GPU cluster. As of now, this will only work if you have multiple CUDA devices available.

When running the benchmark, it is important NOT to use 'python benchmark.py'. Rather, use pytorch's bundled launcher - torchrun. To run: torchrun --nproc_per_node=<gpus available> benchmark.py
If you have multiple nodes, you can use the --nnodes flag as well.