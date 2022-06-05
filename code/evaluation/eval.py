import os
import cv2
import matplotlib.pyplot as plt
import torch
import code.models.resnet34 as resnet34
import code.confs.paths as paths
import code.utils.visualization as visualization


class RealTimePredictor:

    def __init__(self, model, capture_arg):
        self.label_list = ['backward', 'forward', 'left', 'right', 'stop']
        self.model = model
        self.capture = cv2.VideoCapture(capture_arg)

    def film_me(self):
        window_name = "Python Webcam: Real Time Predictions"

        cv2.namedWindow(window_name)

        finished = False
        while not finished:
            ret, frame = self.capture.read()

            if not ret:
                print("Failed to capture frame")
                finished = True

            cv2.imshow(window_name, frame)

            k = cv2.waitKey(20)

            if k == 27 or k == ord('q'):  # Escape Key or 'Q' Key is pressed
                print("Escape Key/'Q' Pressed: closing now")
                self.capture.release()
                cv2.destroyAllWindows()
                finished = True

            if k == ord('p'):   # 'P' pressed
                print("MAKING A PREDICTION")

                torch_frame = torch.permute(torch.tensor([frame], dtype=torch.float32), dims=(0, 3, 1, 2))
                print(type(torch_frame))
                print(torch_frame.shape)

                label_pred, coord_pred = self.model(torch_frame)

                int_frame = torch.permute(torch.tensor(frame), dims=(2, 0, 1))

                visual = visualization.visualize_single_instance(int_frame, coord_pred[0], label_pred[0], "", label_list=self.label_list)

                plt.figure(1)
                plt.imshow(visual.permute(1, 2, 0))
                plt.show()


if __name__ == '__main__':

    model_name = "BB_model_04-06-2022_17-29_251b1e9a-e41b-11ec-9882-312d0a010c77"
    path = os.path.join(paths.ROOT_PATH, "saved_models", model_name)

    print(path)
    print(os.path.isfile(path))

    model = resnet34.BB_model()
    model.load_state_dict(torch.load(path))
    model.eval()

    predictor = RealTimePredictor(model=model, capture_arg=0)
    predictor.film_me()
