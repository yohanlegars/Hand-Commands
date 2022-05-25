import cv2
import os
import time
import uuid
import configargparse
import code.confs.paths as paths


class ImageGenerator(object):
    def __init__(self, image_path, label_list, capture_arg=0):
        """
        The constructor of the Data Generator.

        :param image_path: the path where the images should be saved, str
        :param label_list: the list of labels to create images for
        :param capture_arg: capture argument, for the opencv videocapture object
        (this specified the webcam that will be used)
        """
        self.image_path = image_path
        self.labels = label_list
        self.capture = cv2.VideoCapture(capture_arg)

    def timed_data_generation(self, number_images, headsup_time=3, timer=3):
        """
        This method takes snapshots of the webcam on a regular timer. The images are directly saved into the
        specified image_path of the class.

        :param timer: time between snapshots, in seconds.
        :param headsup_time: time to wait before starting the snapshots, at the start of the process, in seconds.
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
        while time.time() < current_time + headsup_time:
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
                while time.time() < current_time + timer:
                    ret, frame = self.capture.read()

                    if not ret:
                        print("Failed to capture frame")
                        break

                    cv2.putText(frame, label, (100, 100), font, 4, (0, 0, 255), 4)
                    cv2.putText(frame, str(round(3 - (time.time()-current_time), 2)),
                                (100, 200), font, 4, (0, 0, 255), 4)
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
        This method takes snapshots of the webcam when the spacebar is pressed. The images are directly saved into the
        specified image_path of the class.

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

    parser = configargparse.ArgumentParser(default_config_files=[os.path.join(paths.CONFS_PATH, "image_capture.conf")])
    parser.add_argument('--LABELS', type=str, nargs='+', help='list of classes')
    parser.add_argument('--NUMBER_IMGS', type=int, help='number of pictures per classes')
    parser.add_argument('--MODE', type=str, help='either "timed" or "manual"')
    parser.add_argument('--CAPTURE_ARG', type=int, help='Capture argument for the opencv videocapture process')
    parser.add_argument('--SAVE_PATH', type=str, help='the path wherein the images should be saved')

    opt = parser.parse_args()
    print(opt)

    SAVE_PATH = os.path.join(paths.DATA_PATH, "not_yet_annotated")

    data_generator = ImageGenerator(image_path=os.path.normpath(opt.SAVE_PATH),
                                    label_list=opt.LABELS,
                                    capture_arg=opt.CAPTURE_ARG)
    eval("data_generator.{}_data_generation(number_images=opt.NUMBER_IMGS)".format(opt.MODE))
