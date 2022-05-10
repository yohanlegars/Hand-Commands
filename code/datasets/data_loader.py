import sys
import cv2
import os
import time
import uuid
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "config"))
print(CONFIG_PATH)
sys.path.append(CONFIG_PATH)
from paths import DATA_PATH

IMAGES_PATH = DATA_PATH
LABELS = ['stop', 'backward', 'forward', 'left', 'right']
NUMBER_IMGS = 3

class DataGenerator(object):
    def __init__(self,IMAGES_PATH):
        """
        This function is initialsigin the classifier

        :param IMAGES_PATH: the path variable, str
        """
        self.image_path = IMAGES_PATH
        self.labels = LABELS
        self.number_imgs = NUMBER_IMGS

    def data_generation(self):

        for label in self.labels:
            os.mkdir(os.path.join(self.image_path, label))
            cap = cv2.VideoCapture(0)
            print('Collecting images for {}'.format(label))
            time.sleep(5)
            for imgnum in range(self.number_imgs):
                ret, frame = cap.read()
                imagename = os.path.join(self.image_path, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(imagename, frame)
                cv2.imshow('frame', frame)
                time.sleep(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()

if __name__ == '__main__':

    data = DataGenerator(IMAGES_PATH)
    data.data_generation()


            
            



