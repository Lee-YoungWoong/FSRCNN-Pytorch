# FSRCNN

Implementation of FSRCNN in PyTorch.

![fsrcnn](https://github.com/user-attachments/assets/43ac05ca-29b1-4797-92e9-405107d5ae54)

Provide 3 image format "RGB, YCb, Y"

Calculate PSNR only Y channel

# Train (train.py)

Train FSRCNN

Validation Dataset : SET-14

Train Dataset : T91 + Berkeley Segmentation Dataset (Total : 291 images)

You can download more Super Resolution Dataset

↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

DIV2K Dataset : https://data.vision.ee.ethz.ch/cvl/DIV2K/

Flickr2K : https://www.kaggle.com/datasets/daehoyang/flickr2k

Urban100 : https://www.kaggle.com/datasets/harshraone/urban100

※train.py option

  --train_dataset      :     Training dataset path

  --validation_dataset  :    Validation dataset path

  --save_root            :   Model save path

  --crop_size        :       Training image crop size

  --scale_factor       :     Upscale factor

  --batch_size          :    Batch size

  --num_workers         :    Number of workers

  --nb_epochs          :     Number of epochs

  --img_format        :      ['RGB', 'YCbCr', 'Y'] Train Image format

  --cuda              :      Use cuda

  --shuffle          :       data shuffle

  --pin_memory       :       pin_memory

  --drop_last        :       drop_last


# Test (test.py)

Calculate PSNR and dump Bicubic, SR image

※test.py option

  --weight           :       Path to the saved model checkpoint

  --image            :       Path to the input image

  --scale_factor     :       Upscale factor

  --save_root         :      Model Result path

  --img_format       :       ['RGB', 'YCbCr', 'Y'] Train Image format

  --cuda             :       Use cuda

# Validation (val.py)
Calculate Set-14 Average PSNR

※val.py option:

  --weight           :       Path to the saved model checkpoint

  --validation_dataset   :   Validation dataset path

  --scale_factor       :     Upscale factor

  --img_format        :      ['RGB', 'YCbCr', 'Y'] Train Image format

  --cuda             :       Use cuda

# Export (export.py)
Export onnx file

※export.py option

  --load_path        :      Path to the saved model checkpoint

  --save_path        :      Path to save the exported ONNX model

  --verbose           :     Verbose output for onnx export

  --img_format       :      ['RGB', 'YCbCr', 'Y'] Train Image format

  --scale_factor       :     Upscale factor

  --width             :     Width of the input image

  --height            :     Height of the input image

  --opset_version     :     ONNX opset version

# Reference

Paper : Accelerating the Super-Resolution Convolutional Neural Network

link : https://arxiv.org/abs/1608.00367
