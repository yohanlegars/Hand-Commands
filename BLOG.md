
[//]: # (WIP ZONE: MATTI ///////////////////////////////////////////////////////////////////)

# Sign detections with lottery ticket hypothesis using PyTorch

Paul Féry
Matti Lang
Yohan Le Gars

  In this blog post we discuss our approach of creating a sign language dataset from scratch, traning a deep learning model and implementing the lottery ticket hypothesis. The end-goal is to transfer the model on a RaspberryPi which will be connected to a small robot car. Therefore, it is important for the model to be lightweight and to reduce as many parameters as possible with the lottery ticket implementation in order to run the model for a real-time application. 
  
## The Signs
 
The dataset consists of five different signs: "stop", "forward", "backward", "left" and "right". 

<align="middle">
  <img src="https://github.com/yohanlegars/Hand-Commands/blob/main/blog_images/matti_comp.jpg" width="100" height="250"/>
  <img src="https://github.com/yohanlegars/Hand-Commands/blob/main/blog_images/yohan_comp_1_min.jpg" width="100" height="250"/> 
  <img src="https://github.com/yohanlegars/Hand-Commands/blob/main/blog_images/yohan_comp_2-min.jpg" width="100" height="250"/>
  <img src="https://github.com/yohanlegars/Hand-Commands/blob/main/blog_images/paul_comp_1_min.jpg" width="100" height="250"/>
  <img src="https://github.com/yohanlegars/Hand-Commands/blob/main/blog_images/paul_comp_2-min.jpg" width="100" height="250"/>
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

This process can be automated using the 



