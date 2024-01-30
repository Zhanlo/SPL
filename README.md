# SPL
[AAAI 2022]Not All Parameters Should be Treated Equally: Deep Safe Semi-Supervised Learning under Class Distribution Mismatch

This is the code for paper "Not All Parameters Should be Treated Equally: Deep Safe Semi-Supervised Learning under Class Distribution Mismatch" 

## Setups

The code is implemented with Python and Pytorch.

### Prerequisites:

* Python == 3.8
* PyTorch ==1.7.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.8.2
* Numpy == 1.19.2


## Running SPL for benchmark dataset

Here is an example:


```bash
python SPL.py --config config/cifar10_bi.yaml --gpus 0 --ratio 0.6 --l_batch_size 512 --ul_batch_size 512 --test_batch_size 512 --epochs_m 300 --step_size_m 100  --lr_m 0.1 --epochs_w 300 --step_size_w 100  --lr_w 0.1 --prune_rate 0.5 --optimizer sgd --alg VAT --iteration 1
```
