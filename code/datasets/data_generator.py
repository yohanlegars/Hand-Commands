import json
import torch
import glob
import os
import paths
import torchvision
import matplotlib.pyplot as plt
import visualization
from PIL import Image


class HandCommandsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, extension="json"):
        self.root_path = dataset_path
        self.format = extension
        self.tensor_converter = torchvision.transforms.ToTensor()
        assert self.__len__() == len(glob.glob(os.path.join(self.root_path, "*.jpg"))),\
            "Incomplete input/output pairs, check Dataset Folder:\n{}".format(self.root_path)

    def __getitem__(self, idx):
        """
        This function loads individual training instances from the dataset.

        :param idx: an int, which points towards a specific training instance
        :return:
            - img, a torch tensor with shape (Channels, Height, Width) --> for example (3, 480, 640)
            - label_tensor, a torch tensor with the position and size of the bounding box. Shape is (center x, center y, width, height), dtype is torch.int
            - label, a string, specifying the label type (for example, "stop", or "forward"...)
            - instance_name, a string, specifying the file names of the image and labels for this training instance
        """
        annot_file = sorted(glob.glob(os.path.join(self.root_path, "*." + self.format)))[idx]
        image_file = annot_file.split(".")[0] + ".jpg"
        with open(annot_file) as f:
            annot_dict = json.load(f)

        instance_name = annot_dict[0]["image"].split(".")[0]     # removing the ".jpg extension"
        label = annot_dict[0]["annotations"][0]["label"]
        label_tensor = self.get_label_tensor(annot_dict[0]["annotations"][0]["coordinates"])

        img = Image.open(image_file)
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        img = transform(img)
        return img, label_tensor, label, instance_name

    def __len__(self):
        return len(glob.glob(os.path.join(self.root_path, "*." + self.format)))

    def get_label_tensor(self, coords: dict):
        """
        extracts label coordinates from a dictionary into a torch tensor
        :param coords: dictionary of coords
        :return: torch tensor
        """
        x = int(coords["x"])
        y = int(coords["y"])
        width = int(coords["width"])
        height = int(coords["height"])
        return torch.tensor([x, y, width, height], dtype=torch.int)


if __name__ == '__main__':

    DATA_PATH = os.path.join(paths.DATA_PATH, "annotated")
    dataset = HandCommandsDataset(DATA_PATH)
    index = 4

    image, labeltensor, label, name = dataset.__getitem__(index)
    print(f"{image=}")
    print(f"{image.shape=}")
    print(f"{labeltensor=}")
    print(f"{labeltensor.shape=}")
    print(f"{label=}")
    print(f"{name=}")
