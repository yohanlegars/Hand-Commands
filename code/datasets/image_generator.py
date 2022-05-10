import sys
import cv2
import os
import time
import uuid
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "config"))
print(CONFIG_PATH)
sys.path.append(CONFIG_PATH)
import paths

IMAGES_PATH = paths.DATA_PATH
LABELS = ['stop', 'backward', 'forward', 'left', 'right']
NUMBER_IMGS = 5

class DataGenerator(object):
    def __init__(self, IMAGES_PATH):
        """
        This function is initialsigin the classifier

        :param IMAGES_PATH: the path variable, str
        """
        self.image_path = IMAGES_PATH
        self.labels = LABELS
        self.cap = cv2.VideoCapture(2)

    def data_generation(self, number_images):
        for label in self.labels:
            label_path = os.path.join(self.image_path, label)
            if not os.path.isdir(label_path):
                os.mkdir(os.path.join(self.image_path, label))
            print(f'Collecting images for {label} (starting in 5 sec)')
            time.sleep(5)
            for imgnum in range(number_images):
                ret, frame = self.cap.read()
                imagename = os.path.join(label_path, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(imagename, frame)
                cv2.imshow('frame', frame)
                time.sleep(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.cap.release()

if __name__ == '__main__':

    data = DataGenerator(IMAGES_PATH)
    data.data_generation(number_images=5)


            
            



