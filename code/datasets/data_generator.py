import json
import torch
import glob
import os
import paths
import torchvision
from PIL import Image


class HandCommandsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, extension="json"):
        self.root_path = dataset_path
        self.format = extension
        self.tensor_converter = torchvision.transforms.ToTensor()
        assert self.__len__() == len(glob.glob(os.path.join(self.root_path, "*.jpg"))),\
            "Incomplete input/output pairs, check Dataset Folder:\n{}".format(self.root_path)

    def __getitem__(self, idx):
        annot_file = sorted(glob.glob(os.path.join(self.root_path, "*." + self.format)))[idx]
        print(annot_file)
        image_file = annot_file.split(".")[0] + ".jpg"
        print(image_file)
        with open(annot_file) as f:
            annot_dict = json.load(f)
        annot_dict = annot_dict[0]["annotations"]
        print(annot_dict)
        img = Image.open(image_file)
        img = self.tensor_converter(img)
        return img, annot_dict

    def __len__(self):
        return len(glob.glob(os.path.join(self.root_path, "*." + self.format)))


if __name__ == '__main__':

    DATA_PATH = os.path.join(paths.DATA_PATH, "annotated")
    dataset = HandCommandsDataset(DATA_PATH)
    index = 4

    image, dictionary = dataset.__getitem__(index)
    print(json.dumps(dictionary, indent=4))
    print(image)
