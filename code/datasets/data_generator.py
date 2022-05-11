import torch
import glob
import xml.etree.ElementTree as ET
import os
from torchvision import transforms

class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.root_path = dataset_path
        self.files = glob.glob(self.root_path+"/*.xml")
        self.transforms = transforms
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        xml_path = os.path.join(self.root_path, self.files[idx])
        root = ET.parse(xml_path).getroot()
        xmin = int(root.find("object").find("bndbox").find("xmin").text)
        ymin = int(root.find("object").find("bndbox").find("ymin").text)
        xmax = int(root.find("object").find("bndbox").find("xmax").text)
        ymax = int(root.find("object").find("bndbox").find("ymax").text)
        class_name = root.find("object").find("name").text.lower()
        if "spa" in class_name:
            class_index = 0
        elif "del" in class_name:
            class_index = 1
        else:
            class_index = string.printable.index(class_name)-8
        img = Image.open(os.path.join(self.root_path, root.find("filename").text))
        if transforms:
            img = self.transforms(img)
        anotation = {"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax,"class_name":class_name,"class_index":class_index, "file_path":os.path.join(self.root_path, root.find("filename").text)}
        return img, anotation