# Hand Commands vision model
This repository is a full implementation of a hand sign recognition computer vision model.
More specifically, this code is intended for those who want to give hand gesture orders to their mobile robots powered by light-weight microcomputers such as Rasberry Pi.
The deep learning model localizes the hand within an image and performs classification after being trained  on a private manually annotated image dataset. In order to scale down the model and being able to upload it into a 2GB Rasberry Pi 4,
the repository provides a pruning technique based on the lottery ticket hypothesis: <a href="https://arxiv.org/abs/1803.03635"> The lottery Ticket hypothesis: Finding Sparse, Trainable Neural Networks.</a>

## Installation Requirements

The code is compatible with python 3.7 and pytorch 1.11. In addition, the following 
packages are required:
numpy, opencv, torchvision...

## Data Annotation

We generated our training data by taking pictures with the webcam of our laptops. Here are the steps required to obtain your personal annotated dataset.
First git clone this repository, move inside the root of the directory and do the following:
``````
mkdir data && cd data
``````
``````
mkdir not_yet_annotated && mkdir annotated 
``````
Once the folders that will contain the generated images are created, you can start taking pictures from the webcam of your laptop. The pictures must be  of yourself or someone
performing a static hand gesture.
To take the pictures, move to `repo/code/datasets/` and print the following command in your command line:
````
python image_generator.py 
````
You will be given 3 seconds to bring your hand in the correct position, a picture will be taken and it will 
move on to the next pictures. This process will be repeated as many times as defined by the number of classes that you chose and the number of instances generated for each class. The process can be aborted by pressing `q`.
The default classes are those from our personal project `['stop', 'backward', 'forward', 'left', 'right']`. For more information regarding our project, you can consult this blog post 'blog post link here'.

We performed the manual annotations using labelImg package. The package can be installed with `pip3 install labelImg`.  By taking createML in the left column of the application,
the generated annotation file will be saved as `.json`. The file contains the label name as well as the bounding boxes coordinates surrounding the hand. The files must be saved in 'not_yet_annotated' dir.
Once done. you can move the image files with their corresponding json files by running the following command:
````
data_mover.py
````






