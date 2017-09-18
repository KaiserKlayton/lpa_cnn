# imagenet_tools
The auxiliary script `helper/imagenet_tools/preprocess_imagenet.py` is useful for:

1. Subtracting mean images or pixels from an imagenet dataset.

2. Attaching labels to the dataset as a first column.

3. Writing the dataset as a single csv.

To use imagenet_tools:

1. Place the first 1000 JPEGs of the imagenet_2012 dataset (validation), cropped to 224x224, in `helper/imagenet_tools/imagenet_1000/`.

2. Select your option (mean image or pixel normalization), in the source code of `helper/imagenet_tools/preprocess_imagenet.py`.

3. Run `$ python helper/imagenet_tools/preprocess_imagenet.py <mean_image> | <mean_pixel>`.

Results are in `helper/imagenet_tools/mean_image_reduced` or `helper/imagenet_tools/mean_pixel_reduced`. 

These files can now be moved or linked to an `input/<model_name>/production/` directory.
