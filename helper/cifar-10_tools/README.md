# cifar-10_tools
The auxiliary script `helper/cifar-10/preprocess_cifar-10.py` is useful for:
 
1. Subtracting `mean.binaryproto` from a cifar-10 dataset.

To use cifar-10_tools:

1. Place the cifar-10 dataset in `helper/cifar-10_1000/cifar-10_1000.csv`.

2. Run `$ python helper/cifar-10_tools/preprocess_cifar-10.py`.

Results are in `helper/cifar-10/mean_cifar-10_reduced`.

These files can now be moved or linked to an `input/<model_name>/production/` directory.
