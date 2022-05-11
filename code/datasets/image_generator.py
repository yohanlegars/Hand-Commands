import cv2
import os
import time
import uuid
# import sys
# CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "config"))
# print(CONFIG_PATH)
# sys.path.append(CONFIG_PATH)
import paths


class DataGenerator(object):
    def __init__(self, IMAGES_PATH):
        """
        The constructor of the Data Generator.

        :param IMAGES_PATH: the path where the images should be saved, str
        """
        self.image_path = IMAGES_PATH
        self.labels = LABELS
        self.capture = cv2.VideoCapture(0)

    def timed_data_generation(self, number_images):
        """
        This method takes snapshots of the webcam on a regular timer.

        :param number_images: number of instances to generate per label.
        :return: None
        """
        window_name = "Python Webcam: Timed Snapshots"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.namedWindow(window_name)

        # Little wait period at the start, to not be surprised
        current_time = time.time()
        print("Ok, prepare for quick snapshots now:\n"
              "First label is: {}".format(self.labels[0]))
        while time.time() < current_time + 3:
            ret, frame = self.capture.read()

            if not ret:
                print("Failed to capture frame")
                break

            cv2.imshow(window_name, frame)
            k = cv2.waitKey(125)

            if k == 27 or k == ord('q'):  # Escape Key or 'Q' Key is pressed
                print("Escape Key/'Q' Pressed: closing now")
                self.capture.release()
                cv2.destroyAllWindows()
                return

        for label in self.labels:
            print("Now Collecting images for label {}".format(label))
            for img_count in range(number_images):
                img_name = '{}_{}.jpg'.format(label, str(uuid.uuid1()))
                current_time = time.time()
                while time.time() < current_time + 3:
                    ret, frame = self.capture.read()

                    if not ret:
                        print("Failed to capture frame")
                        break

                    cv2.putText(frame, str(round(3 - (time.time()-current_time), 2)), (100, 100), font, 4, (255, 0, 0))
                    cv2.imshow(window_name, frame)
                    k = cv2.waitKey(125)

                    if k == 27 or k == ord('q'):    # Escape Key or 'Q' Key is pressed
                        print("Escape Key/'Q' Pressed: closing now")
                        self.capture.release()
                        cv2.destroyAllWindows()
                        return

                else:
                    ret, frame = self.capture.read()

                    cv2.imshow(window_name, frame)
                    cv2.imwrite(os.path.join(self.image_path, img_name), frame)
                    cv2.waitKey(500)
        self.capture.release()
        cv2.destroyAllWindows()

    def manual_data_generation(self, number_images):
        """
        This method takes snapshots of the webcam when the spacebar is pressed.

        :param number_images: number of instances to generate per label.
        :return: None
        """
        window_name = "Python Webcam: Manual Snapshots"
        cv2.namedWindow(window_name)
        img_counter = 0
        label_counter = 0
        finished = False

        print("Beginning Manual Snapshots:\n"
              "Press the Space Bar to take a snapshot\n\n"
              "Take a snapshot for label {}".format(self.labels[label_counter]))
        while not finished:
            ret, frame = self.capture.read()

            if not ret:
                print("Failed to capture frame")
                finished = True

            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1)

            if k == 27 or k == ord('q'):    # Escape Key or 'Q' Key is pressed
                print("Escape Key/'Q' Pressed: closing now")
                finished = True

            elif k == 32:                   # Spacebar Key is pressed
                img_name = '{}_{}.jpg'.format(self.labels[label_counter], str(uuid.uuid1()))
                cv2.imwrite(os.path.join(self.image_path, img_name), frame)
                print('Picture number {} taken for label {}'.format(str(img_counter+1), self.labels[label_counter]))
                img_counter += 1

                if img_counter == number_images:
                    if label_counter == len(self.labels)-1:
                        finished = True
                    else:
                        img_counter = 0
                        label_counter += 1
                        print("Moving to next Label,\n"
                              "Now Take a Screenshot for label {}".format(self.labels[label_counter]))
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    SAVE_PATH = paths.DATA_PATH
    LABELS = ['stop', 'backward', 'forward', 'left', 'right']
    NUMBER_IMGS = 5
    MODE = "manual"     # can either be 'manual' or 'timed'

    data_generator = DataGenerator(SAVE_PATH)
    eval("data_generator.{}_data_generation(number_images=NUMBER_IMGS)".format(MODE))
