<h2>TensorFlow-FlexUNet-Image-Segmentation-Triple-Negative-Breast-Cancer (2025/08/27)</h2>

This is the first experiment of Image Segmentation for Triple Negative Breast Cancer (TNBC)
 based on our 
 <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
<b>TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass)</b></a>
, and a 512x512 pixels 
<a href="https://drive.google.com/file/d/17B1gmovVrPStwtov59Oau7VXUqp6Q3RT/view?usp=sharing">
<b>TNBC-PNG-ImageMask-Dataset.zip</b></a>.
which was derived by us from 
<br><br>
<a href="https://zenodo.org/records/2579118/preview/TNBC_NucleiSegmentation.zip?include_deleted=0#tree_item0">
 TNBC_NucleiSegmentation.zip
</a> in 
<a href="https://zenodo.org/records/2579118"><b>Segmentation of Nuclei in Histopathology Images by deep regression of the distance map</b></a>
<br>
<br>
On the TNBC-ImageMask-Dataset, please refer to our repository 
<a href="https://github.com/sarah-antillia/ImageMask-Dataset-Triple-Negative-Breast-Cancer">ImageMask-Dataset-Triple-Negative-Breast-Cancer</a>

<br>
Please see also our experiment for a singleclass segmentation model 
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Triple-Negative-Breast-Cancer">
Tensorflow-Image-Segmentation-Triple-Negative-Breast-Cancer
</a>
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a> ,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<b>Acutual Image Segmentation for 512x512 TNBC images</b><br>

As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/1010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/1010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/1010.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/1022.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/1022.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/1022.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/1039.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/1039.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/1039.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The original dataset used here has been taken from the following web site.<br>
<a href="https://zenodo.org/records/2579118"><b>Segmentation of Nuclei in Histopathology Images by deep regression of the distance map</b></a>
<br>
Published February 16, 2018 | Version 1.1<br>
<br>
<b>Creators</b><br>
Naylor Peter Jack,Walter Thomas, Laé Marick, Reyal Fabien<br>
<br>
<b>Description</b><br>
This dataset has been announced in our accepted paper "Segmentation of Nuclei in Histopathology Images<br> 
by deep regression of the distance map" in Transcation on Medical Imaging on the 13th of August.<br>
This dataset consists of 50 annotated images, divided into 11 patients.<br>
<br>
 
v1.1 (27/02/19): Small corrections to a few pixel that were labelled nuclei but weren't.<br>
<br>


<br>
<h3>
<a id="2">
2 TNBC ImageMask Dataset
</a>
</h3>
 If you would like to train this TNBC Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/17B1gmovVrPStwtov59Oau7VXUqp6Q3RT/view?usp=sharing">
<b>TNBC-PNG-ImageMask-Dataset.zip</b></a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─TNBC
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>TNBC Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/TNBC/TNBC_Statistics.png" width="512" height="auto"><br>
<br>

On the derivation of the augmented dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained TNBC TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/TNBC/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/TNBC and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for TNBC 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; 1+1 classes
; RGB colors     TNBC;white     
rgb_map = {(0,0,0):0,(255,255,255):1,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 30,31,32)</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 61,62,63)</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 63 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/train_console_output_at_epoch63.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/TNBC/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/TNBC/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/TNBC</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for TNBC.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/evaluate_console_output_at_epoch63.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/TNBC/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this TNBC/test was low and dice_coef_multiclass 
high as shown below.
<br>
<pre>
categorical_crossentropy,0.0659
dice_coef_multiclass,0.9667
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/TNBC</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for TNBC.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/TNBC/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/1029.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/1029.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/1029.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/1039.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/1039.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/1039.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/barrdistorted_1001_0.3_0.3_1034.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/barrdistorted_1001_0.3_0.3_1034.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/barrdistorted_1001_0.3_0.3_1034.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/deformed_alpha_1300_sigmoid_8_1036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/deformed_alpha_1300_sigmoid_8_1036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/deformed_alpha_1300_sigmoid_8_1036.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/deformed_alpha_1300_sigmoid_8_1035.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/deformed_alpha_1300_sigmoid_8_1035.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/deformed_alpha_1300_sigmoid_8_1035.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/images/deformed_alpha_1300_sigmoid_9_1019.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test/masks/deformed_alpha_1300_sigmoid_9_1019.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TNBC/mini_test_output/deformed_alpha_1300_sigmoid_9_1019.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Triple negative breast cancer: Pitfalls and progress</b><br>
npj Breast Cancer volume 8, Article number: 95 (2022) <br>
Paola Zagami & Lisa Anne Carey<br>
<a href="https://www.nature.com/articles/s41523-022-00468-0">
https://www.nature.com/articles/s41523-022-00468-0
</a>
<br>
<br>
<b>2. A Large-scale Synthetic Pathological Dataset for Deep Learning-enabled <br>
Segmentation of Breast Cancer</b><br>
Scientific Data volume 10, Article number: 231 (2023) <br>
Kexin Ding, Mu Zhou, He Wang, Olivier Gevaert, Dimitris Metaxas & Shaoting Zhang<br>
<a href="https://www.nature.com/articles/s41597-023-02125-y">
https://www.nature.com/articles/s41597-023-02125-y
</a>
<br>
<br>
<b>3. A review and comparison of breast tumor cell nuclei segmentation performances<br>
 using deep convolutional neural networks</b><br>
 Scientific Reports volume 11, Article number: 8025 (2021) <br>
 Andrew Lagree, Majidreza Mohebpour, Nicholas Meti, Khadijeh Saednia, Fang-I. Lu,<br>
Elzbieta Slodkowska, Sonal Gandhi, Eileen Rakovitch, Alex Shenfield, <br>
Ali Sadeghi-Naini & William T. Tran<br>
<a href="https://www.nature.com/articles/s41598-021-87496-1">
https://www.nature.com/articles/s41598-021-87496-1
</a>

