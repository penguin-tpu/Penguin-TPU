# penguin-compiler

Python package for the hyperspecialized compile flow:

- inspect one target PyTorch model
- validate fixed shapes and parameters
- emit Penguin assembly
- emit `manifest.json`
- emit packed binary constant blobs

The output is a program bundle consumable by `penguin-model`, RTL testbenches, and FPGA
host loaders.

