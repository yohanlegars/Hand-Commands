import json
import torch
import glob
import os
import code.confs.paths as paths
import torchvision
from PIL import Image


class HandCommandsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, extension="json"):
        """
        The initialization of the dataset

        :param dataset_path: path wherein our label dictionaries and our training images are located. Dictionary/image pairs should have the same name (except for the file extension)
        :param extension: file extension used for the label dictionaries. Only "json" available for now  #TODO: potential future work: making this work with other formats
        """
        self.root_path = dataset_path
        self.format = extension
        self.label_list = self.get_label_list()
        self.tensor_converter = torchvision.transforms.ToTensor()

        assert self.__len__() == len(glob.glob(os.path.join(self.root_path, "*.jpg"))),\
            "Incomplete input/output pairs, check Dataset Folder:\n{}".format(self.root_path)

    def __getitem__(self, idx):
        """
        This function loads individual training instances from the dataset.

        :param idx: an int, which points towards a specific training instance
        :return:
            - img, a torch tensor with shape (Channels, Height, Width) --> for example (3, 480, 640)
            - coord_tensor, a torch tensor with the position and size of the bounding box.
              Shape is (center x, center y, width, height), see get_coord_tensor method
            - label_tensor, a one-hot-encoded representation of the label. Dimension is equal to number of labels.
              See get_label_tensor method
            - instance_name, a string, specifying the file names of the image and labels for this training instance
        """
        annot_file = sorted(glob.glob(os.path.join(self.root_path, "*." + self.format)))[idx]
        image_file = annot_file.split(".")[0] + ".jpg"
        with open(annot_file) as f:
            annot_dict = json.load(f)

        instance_name = annot_dict[0]["image"].split(".")[0]     # removing the ".jpg extension"
        label_tensor = self.get_label_tensor(annot_dict[0]["annotations"][0]["label"])
        coord_tensor = self.get_coord_tensor(annot_dict[0]["annotations"][0]["coordinates"])

        img = Image.open(image_file)
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        img = transform(img)
        return img, coord_tensor, label_tensor, instance_name

    def __len__(self):
        return len(glob.glob(os.path.join(self.root_path, "*." + self.format)))

    def get_coord_tensor(self, coords: dict):
        """
        extracts label coordinates from a dictionary into a torch tensor

        :param coords: dictionary of coords
        :return: torch tensor
        """
        x = int(coords["x"])
        y = int(coords["y"])
        width = int(coords["width"])
        height = int(coords["height"])
        tensor = torch.tensor([x, y, width, height], dtype=torch.float32)
        return tensor

    def get_label_tensor(self, label: str):
        """
        extracts label string and turns it into a one hot encoded torch tensor

        :param label: label, as a string
        :return: torch tensor
        """
        tensor = torch.tensor(data=self.label_list.index(label))
        tensor = torch.nn.functional.one_hot(tensor, num_classes=len(self.label_list))
        tensor = tensor.clone().detach()
        tensor = tensor.float()
        return tensor

    def get_label_list(self):
        """
        Extracts the total list of labels stored within the provided dataset.

        :return: a list of strings, each one corresponding to a label class
        """
        label_list = []
        for file in glob.glob(os.path.join(self.root_path, "*." + self.format)):
            with open(file) as f:
                annot_dict = json.load(f)
            label_list.append(annot_dict[0]["annotations"][0]["label"])
        return sorted(list(dict.fromkeys(label_list)))  # removes duplicates


def generate_dataset_splits(dataset, splitratio):
    split_one_amount = int(splitratio * len(dataset))
    split_two_amount = len(dataset) - split_one_amount
    assert len(dataset) == split_one_amount + split_two_amount

    split_one, split_two = torch.utils.data.random_split(dataset, [split_one_amount, split_two_amount])
    return split_one, split_two


if __name__ == '__main__':

    DATA_PATH = os.path.join(paths.DATA_PATH, "annotated")

    dataset = HandCommandsDataset(dataset_path=DATA_PATH)

    # for index in range(len(dataset)):
    #     image, coord_tensor, label_tensor, name = dataset.__getitem__(index)
    #     print(f"{image=}")
    #     print(f"{image.shape=}")
    #     print(f"{coord_tensor=}")
    #     print(f"{coord_tensor.shape=}")
    #     print(f"{label_tensor=}")
    #     print(f"{label_tensor.shape=}")
    #     print(f"{name=}")
    #     print("")

    image, coord_tensor, label_tensor, name = dataset.__getitem__(4)
    print(f"{image=}")
    print(f"{image.shape=}")
    print(f"{coord_tensor=}")
    print(f"{coord_tensor.shape=}")
    print(f"{dataset.get_label_list()=}")
    print(f"{label_tensor=}")
    print(f"{label_tensor.shape=}")
    print(f"{name=}")
    print("")
