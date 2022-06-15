
[//]: # (WIP ZONE: MATTI ///////////////////////////////////////////////////////////////////)

# Sign detections with lottery ticket hypothesis using PyTorch

Paul Féry
Matti Lang
Yohan Le Gars

  In this blog post we discuss our approach of creating a sign language dataset from scratch, traning a deep learning model and implementing the lottery ticket hypothesis. The end-goal is to transfer the model on a RaspberryPi which will be connected to a small robot car. Therefore, it is important for the model to be lightweight and to reduce as many parameters as possible with the lottery ticket implementation in order to run the model for a real-time application. Below is a summary showing the workflow of the pipeline we implemented. 

<p align="middle">
  <img src="blog_images/path.png"/>
</p>
  
## The Signs
 
The dataset consists of five different signs: "forward", "backward", "left" and "right","stop". The images can be seen below:

<p align="middle">
  <img src="blog_images/matti_comp.jpg" width="100" height="180"/>
  <img src="blog_images/yohan_comp_1_min.jpg" width="100" height="180"/> 
  <img src="blog_images/yohan_comp_2-min.jpg" width="100" height="180"/>
  <img src="blog_images/paul_comp_1_min.jpg" width="100" height="180"/>
  <img src="blog_images/paul_comp_2-min.jpg" width="100" height="180"/>
</p>



[//]: # (WIP ZONE: PAUL ///////////////////////////////////////////////////////////////////)

## Image Creation process

We wanted to automate the process of generating image samples for the dataset as much as possible. That way it would facilitate the addition of new samples if performance deemed it necessary. Additionally, if we wanted to add new signs (ie, additional class labels for the model to learn), creating that data would be simple and time-efficient.

With that in mind, we implemented a [simple script](https://github.com/yohanlegars/Hand-Commands/blob/e11a10a30cf5535e66404ad2855839202c5d915f/code/datasets/image_generator.py#L140) that ensures a fast and efficient process for taking images. The user simply needs to specify the following arguments:
- --LABELS, a list of strings, corresponding to the different signs the user would like to create images for
- --NUMBER_IMGS, the number of images they would like to take for each specified sign
- --MODE, which can be:
  - "timed": image samples are taken from the user's webcam/camera sequentially, using a timer. The user performs the first gesture a number of times equal to NUMBER_IMGS, then continues to the second, third, etc.
  - "manual": the user can press the spacebar on their computer to take images using their camera. The advantage is more control on the timing, but they have to stand close to their computer.
- --SAVE_PATH, a path on their computer, where the generated images will be stored.
- --CAPTURE_ARG, a necessary argument for identifying the camera hardware that will be used for the process. This almost always defaults to '0', it is essentially only ever used for computers with multiple webcams connected to them.

Sampling images is made quick and easy, while the script is running, visual feedback is provided to help the user in the process. They directly see what the camera sees, as well as what sign they must be performing a gesture for. If the mode is set to timed, a decreasing timer is also shown, indicating when the next sample will be taken.

[//]: # (TODO: SHORT GIF OF THE PROCESS HERE)


## Annotation process

Sampling images is not the only part of the dataset creation process. If we want to be able to perform nice predictions, we also want the labels corresponding to our sampled images. For our task, the labels are the sign type [REFER TO SIGN TYPE SECTION HERE], and the location of the sign in the image.

This is definitely the most time-consuming part of the dataset creation. Each image must be individually provided with its own label and coordinates for the sign. Nevertheless, it is possible to make the process efficient, to a reasonable extent. For this, we initially used the annotation program [labelImg](https://github.com/tzutalin/labelImg), which provides ample functionalities. However, building a dataset with multiple contributors requires an easy way to pool each team member's generated data and annotations in a common place. As a result, we ended up choosing an [online platform](https://roboflow.com/) to facilitate our work within the team. This is by no means required, and if you were to choose to work on your own on a custom dataset, we would recommend to use the labelImg program.

In the end, each image is provided with its own annotation file, which looks like this:

<p align="middle">
  <img src="blog_images/comp_vis_label.png"/>
</p>

For our project, the hand sign / class label correspondences are as follows:

| Hand Sign | Label |
|-----------|-------|
| backward  | 0     |
| forward   | 1     |
| left      | 2     |
| right     | 3     |
| stop      | 4     |

For the bounding box, each of the dimensions are normalized with respect to the input image resolution.


## The Model

Our initial attempt at an architecture was a reproduction from scratch of the [YOLOv3 model](https://pjreddie.com/media/files/papers/YOLOv3.pdf). This process did unfortunately not fare the way we initially hoped to follow. Complications in implementing the architecture from scratch made it difficult to obtain a working, self-made version of the model. Those manifested themselves mostly in lack of coordination in the creation of the several components of the process: building everything from scratch requires precise coordination between how the architecture is built, how its training loop is built, and how the dataset extraction process is conducted. We arrived at a stage where a training loop and dataset classes were fully implemented and ready, using Pytorch. However, the dataset class does not prepare label tensors in the format that is accepted by YOLO. Although fixing this issue was no concern in terms of our capabilities, heavy time constraints related to the schedule imposed for the project led us to abandon this route, and work using a [YOLOv5 model provided by ultralytics](https://github.com/ultralytics/yolov5). With the benefit of hindsight, a lot could have been done differently with respect to this design process, which would have permitted the success of creation of our own version of a YOLO architecture.

### A (brief) History of YOLO

YOLOv5 is the latest iteration of the YOLO architecture. At its core, the YOLO architecture is a single stage object detection module. It is fast enough to permit real time use with video, and has remained within the state of the art for the object detection task since [it was first introduced in 2015](https://arxiv.org/abs/1506.02640).

The key idea behind the architecture is to divide input images into an *SxS* grid of cells: for example, with *S=7*:

<p align="middle">
  <img src="blog_images/yolo_grid.png"/>
</p>

Each cell is taking care of making a pre-specified number *B* of bounding box predictions (with 4 arguments, indicating the center position and dimensions of the box). Each bounding box prediction is accompanied with a *confidence score*; *P<sub>c</sub>*. Each bounding box is therefore fully specified as a vector [*P<sub>c</sub>*, *x*, *y*, *w*, *h*]. Additionally, cells are also estimating conditional class probabilities for every label the model is being trained on. Those probabilities can be interpreted as "the probability that an object of a given class *C<sub>i</sub>* is present inside the cell, if it is admitted that an object *is* indeed present in the cell".
The output of a prediction made by YOLO on an input image has a total of *SxSx(Bx5+C)* dimensions, where *S* is the number of cells along each dimension, *B* is the number of bounding box predictions per cell, and *C* is the number of classes. In the case that each cell is assigned 3 box predictions to make, and there are 5 distinct classes, each cell produces a prediction tensor of the format:

<p align="middle">
  <img src="blog_images/predict_tensor.gif"/>
</p>


Training instances must also be of the same format: the information contained within the label files (as shown in the previous section) are therefore processed accordingly before training (conditional class probabilities and confidence scores are set to 1, or 0, since the manually labelled images do contain that information with certainty).

The architecture that takes care of converting an input image into a prediction tensor can be seen here:

<p align="middle">
  <img src="blog_images/architecture.PNG"/>
</p>

Let it be noted that this schematic of the architecture is taken from the [original article](https://arxiv.org/abs/1506.02640) of the first version of YOLO. Number of improvements on the architecture have been made over the years, which we will discuss very briefly.

The first iteration over the initial design of YOLO is [YOLOv2](https://arxiv.org/abs/1612.08242v1). Its most notable improvements consist of the introduction [batch normalization](https://arxiv.org/abs/1502.03167), which acts as a regularizer, but most importantly, the choice of using anchor boxes for the bounding box regression task. Output tensors now each have conditional class probabilities estimations for every bounding box within each cell: continuing with the example shown just above for YOLO, YOLOv2 and every one that follow it will output tensors of the following format, for each cell (number of dimensions of the final output tensor is *SxSx(Bx(5+C))*):

<p align="middle">
  <img src="blog_images/predict_tensor_v2.gif"/>
</p>

Note that the introduction of anchor boxes consists in the addition of prior knowledge about the type of images the architecture is expected to work with. Concretely, the authors perform k-means clustering of the ground truth bounding box dimensions in order to construct their anchor box dimensions (you can refer to [the article](https://arxiv.org/abs/1612.08242v1) for more details).

[YOLOv3](https://arxiv.org/abs/1804.02767) compiles a number of minute improvements made by the original authors of YOLO. The classification task is now reformulated into a multilabel classification using Binary Cross Entropy instead of Cross Entropy. The difference is important when training on datasets with non mutually exclusive labels such as "woman" and "human" for example. Additionally, the model now performs multiscale predictions: ie, the boxes are predicted at 3 different scales, in a similar fashion than descirbed in [this article](https://arxiv.org/abs/1612.03144). Finally, they introduce a deeper, more performant network in order to perform the task. It consists of 53 convolutional layers (some of which with *3x3* kernels, others with *1x1*), and it makes use of shortcut connections. They call the architecture Darknet-53.

After YOLOv3, the original creators of the YOLO architecture stopped working on it (or at least, they did not publish any update until this day). But it's not over yet. [Bochkovskiy et al](https://arxiv.org/abs/2004.10934v1) proposed YOLOv4, which encompasses their own set of improvements made on the original authors' work. In short, they investigate a plethora of new practices, split into two separate categories:
- "Bag of Freebies": techniques that improve performance without adding any inference time.
  - Mosaic Data Augmentation
  - Self-adversarial-training
  - CIoU loss
- "Bag of Specials": techniques that slightly increase the model's inference time, but significantly improve prediction accuracy.
  - Mish-activation
  - Cross mini-Batch Normalization
  - Dropblock regularization

They then perform ablation studies on these new techniques, to determine their relevance. Bear in mind that some additional investigations are also conducted on the choice of the architecture itself. In the end, the authors develop a new version of YOLO, which performs better than the previous iteration, and that does not require impressive/inaccessible hardware for training. The work described in the original paper is extremely extensive, we cannot summarize it all here; feel free to [read it](https://arxiv.org/abs/2004.10934v1) for a detailed description of the conducted experiments.

Finally, [YOLOv5](https://github.com/ultralytics/yolov5)'s most important contribution is a port made of the original YOLOv3 architecture, using the [PyTorch](https://pytorch.org/) library. The creators of the YOLOv5 repository aim at making the code open source, letting the deep learning community freely provide their own insights and improvements on YOLO. From this, a number of child-versions of YOLOv5 have emerged; those are all made available and described in the repository itself.

[//]: # (WIP ZONE HERE: MODEL DESCRIPTION ABOVE, ANYTHING ELSE UNDERNEATH ######)

### Training
In order to speed up the training process it was decided to use the Google Cloud Patform (GCP). 
The details of the VM instance used are shown below:

|     GPU type      | Numbers of GPU |    Machine type    |Operating system|          Boot disk type            |
|-------------------|----------------|--------------------|----------------|------------------------------------|
| NVIDIA Tesla P100 |        1       | 8vCPU, 30 GB memory|Ubuntu 20.04 LTS|Balanced persistent disk (size 50GB)|

#### Hyperparameters

We trained a Yolov5 small with a batch size of 42 and 200 epochs. Below we show a list of all the hyperparameters. It is important to point out that we purposely set the image flip left-right (fliplr) to 0 so that the "left" and "right" signs are not confused during training. 

 - lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3) 
 - lrf: 0.01  # final OneCycleLR learning rate (lr0 * lrf) 
 - momentum: 0.937  # SGD momentum/Adam beta1 
 - weight_decay: 0.0005  # optimizer weight decay 5e-4 
 - warmup_epochs: 3.0  # warmup epochs (fractions ok) 
 - warmup_momentum: 0.8  # warmup initial momentum 
 - warmup_bias_lr: 0.1  # warmup initial bias lr 
 - box: 0.05  # box loss gain 
 - cls: 0.5  # cls loss gain 
 - cls_pw: 1.0  # cls BCELoss positive_weight 
 - obj: 1.0  # obj loss gain (scale with pixels) 
 - obj_pw: 1.0  # obj BCELoss positive_weight 
 - iou_t: 0.20  # IoU training threshold 
 - anchor_t: 4.0  # anchor-multiple threshold 
 - fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5) 
 - hsv_h: 0.015  # image HSV-Hue augmentation (fraction) 
 - hsv_s: 0.7  # image HSV-Saturation augmentation (fraction) 
 - hsv_v: 0.4  # image HSV-Value augmentation (fraction) 
 - degrees: 0.0  # image rotation (+/- deg) 
 - translate: 0.1  # image translation (+/- fraction) 
 - scale: 0.5  # image scale (+/- gain) 
 - shear: 0.0  # image shear (+/- deg) 
 - perspective: 0.0  # image perspective (+/- fraction), range 0-0.001 
 - flipud: 0.0  # image flip up-down (probability) 
 - fliplr: 0.0  # image flip left-right (probability) 
 - mosaic: 1.0  # image mosaic (probability) 
 - mixup: 0.0  # image mixup (probability) 
 - copy_paste: 0.0  # segment copy-paste (probability) 

#### Results

The total training time was 46 minutes on the instance. Below we can see the results of the classification and regression tasks. Comparing the training and validation sets we can see that both perform well for both tasks. We do note that in the regression task the training set performs better than the validation set. 

<p align="middle">
  <img src="blog_images/yolov5s-analysis/yolov5s-class-loss.png" width="500" height="350"/>
  <img src="blog_images/yolov5s-analysis/yolov5s-val-train.png" width="500" height="350"/>
</p>

Below is a mosaique of images from the validation set with detection ansd prediction scores shown on the images. 

<p align="middle">
  <img src="blog_images/yolov5s-analysis/yolov5s-200-42-mosaique.png"  width="500" height="500"/>
</p>

[//]: # (WIP ZONE HERE: MODEL DESCRIPTION ABOVE, ANYTHING ELSE UNDERNEATH ######)

### Sparse YOLOv5
Having a nicely working model running on GPU is all nice and dundy until you want to transfer your model 
inside a micro-computer that only runs on CPU, namely the Rasberry 4. 
But do not fret, we have the solution thanks to Neural Magic. By using the open-source tools by Neural Magic, we can supercharge 
our YOLOv5 inferene performance on CPUs by sparsifying the model using SparseML quantization aware training.
Sparsification is the process of removing redundant information from a model and outputting a new smaller and faster model.
<p align="middle">
   <a href="https://docs.neuralmagic.com/sparseml/">
  <img src="blog_images/sparsification.png"/>
</a>
</p>
We have used the 2 general methods to sparsify the model, being -Pruning and Quantization.
Pruning removes redundant parameters or neurons that do not significantly contribute to the accuracy of the 
results. As a result, it reduces the computational complexity. Here, we used static pruning where all pruning steps are  performed offline prior to 
inference.
Quantization reduces computations by reducing the precision of the datatype. The 32-bit floating point (FP32) weights, biases and activations are 
being quantized to smaller width datatypes, typically to 8-bit integers (INT8).
t must be noted that the quantization step overloads the GPU, to the extent of crashing the training process when the batch size is relatively large.
To remedy this issue, we had to divide the training process into 2 training runs. In order to get the best of both worlds, the first run is 200 epochs with a batch size of 42 as explained above, while the second
is a 100 epochs with pruning + quantization and a reduced batch size of 16 performed on the obtained pre-trained weights from the first run.

<p align="center">
  <img width="60%" src="./blog_images/base.gif"/>
</p>

<p align="center">
  <img width="60%" src="./blog_images/sparse.gif"/>
</p>

[//]: # (WIP ZONE HERE: MODEL DESCRIPTION ABOVE, ANYTHING ELSE UNDERNEATH ######)


### Future work

#### Yolov5-small vs Yolov5-nano

In addition to Yolov5-small we also trained Yolov5-nano with the same hyperparameters (200 epochs, batch size 42) for a comparison. From the figures below we see that Yolov5-nano achieves an almost identical performance to Yolov5-small while reducing training time and allocating much less GPU memory. Therefore, when transferring the model on the RaspberryPi we would use a Yolov5-nano instead of a Yolov5-small and implement the pruning and quantization on this model in order to make it as light weight as possible. 

<p align="middle">
  <img src="blog_images/yolov5s-analysis/GPU-small-nano.png" width="500" height="350"/>
  <img src="blog_images/yolov5s-analysis/loss-nano-small.png" width="500" height="350"/>
</p>

