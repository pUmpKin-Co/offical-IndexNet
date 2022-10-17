import os
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset
from IndexNet.dataset.transform import *


class PixProLoveDA(Dataset):
    def __init__(self, data_dir, config):
        self.data_dir = Path(data_dir) / "Rural" / "Images_256"
        self.img_name_list = os.listdir(self.data_dir)
        self.transform = get_PixPro_transform("BYOL", 0.4)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        imgdict = dict()
        img = Image.open(os.path.join(self.data_dir, self.img_name_list[idx]))
        img1, coord1 = self.transform(img)
        img2, coord2 = self.transform(img)

        imgdict["img1"], imgdict["coord1"] = img1, coord1
        imgdict["img2"], imgdict["coord2"] = img2, coord2

        return imgdict


class PixProPotsdam(Dataset):
    def __init__(self, data_dir, txt_file_dir, config):
        self.data_root = os.path.join(data_dir, "Image")
        self.txt_file_dir = txt_file_dir
        self.transform = get_PixPro_transform("BYOL", 0.4)

        self.img_name_list = []
        txt_path = os.path.join(self.txt_file_dir, "pretrain.txt")
        f = open(txt_path, "r")
        for line in f.readlines():
            self.img_name_list.append(line.strip("\n"))
        f.close()

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        imgdict = dict()
        img = Image.open(os.path.join(self.data_root, self.img_name_list[idx]) + ".png")
        img1, coord1 = self.transform(img)
        img2, coord2 = self.transform(img)

        imgdict["img1"], imgdict["coord1"] = img1, coord1
        imgdict["img2"], imgdict["coord2"] = img2, coord2

        return imgdict


class Potsdam(Dataset):
    CLASSES = [
                "Impervious surfaces",
                "Building",
                "Low vegetation",
                "Tree",
                "Car",
                "background"
                ]

    PALETTE = [
                [255, 255, 255],
                [0, 0, 255],
                [0, 255, 255],
                [0, 255, 0],
                [255, 255, 0],
                [255, 0, 0]
            ]

    def __init__(self, data_dir, txt_file_dir, config, split="pretrain"):
        assert split in ("train", "val", "test", "pretrain", "train_transfer")
        self.split = split
        self.no_pyd = config.network.no_pyd
        self.no_index = config.network.no_index

        # path
        self.data_root = os.path.join(data_dir, "Image")
        if self.split in ["train", "val", "train_transfer", "test"]:
            self.label_root = os.path.join(data_dir, "Label")
        self.txt_file_dir = txt_file_dir
        self.txt_name = split + ".txt"

        # config
        self.img_size = config.data.size
        self.backbone_grid_size = config.network.output_size

        # transform
        if self.split == "pretrain":
            self.transform = get_pretrain_transform(size=self.img_size, type="image")
            if not self.no_index:
                self.interpolate = get_pretrain_transform(type="mask")
        elif self.split in ["train", "train_transfer"]:
            self.transform =get_train_transform(size=self.img_size)
        else:
            self.transform = get_test_transform(size=self.img_size)

        self.img_name_list = []
        txt_path = os.path.join(self.txt_file_dir, self.txt_name)
        f = open(txt_path, "r")
        for line in f.readlines():
            self.img_name_list.append(line.strip("\n"))
        f.close()

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        imgdict = dict()
        img = Image.open(os.path.join(self.data_root, self.img_name_list[idx]) + ".png")
        if self.split == "pretrain":
            img = np.asarray(img.resize(self.img_size))
            if self.no_index:
                transformed1 = self.transform(image=img)
                transformed2 = self.transform(image=img)
                img1 = transformed1["image"].type(torch.float32)
                img2 = transformed2["image"].type(torch.float32)

                imgdict["img1"] = img1
                imgdict["img2"] = img2
            else:
                mask1 = np.arange(self.backbone_grid_size * self.backbone_grid_size, dtype=np.uint8).reshape(self.backbone_grid_size, self.backbone_grid_size)
                mask2 = np.copy(mask1)
                mask1 = self.interpolate(image=mask1)["image"]
                mask2 = self.interpolate(image=mask2)["image"]
                transformed1 = self.transform(image=img, mask=mask1)
                transformed2 = self.transform(image=img, mask=mask2)
                img1, mask1 = transformed1["image"].type(torch.float32), transformed1["mask"].float()
                img2, mask2 = transformed2["image"].type(torch.float32), transformed2["mask"].float()

                if self.no_pyd:
                    mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(1), size=28, mode="nearest").squeeze()
                    mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(1), size=28, mode="nearest").squeeze()
                else:
                    mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(1), size=28, mode="nearest").squeeze()
                    mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(1), size=28, mode="nearest").squeeze()

                imgdict["img1"], imgdict["mask1"] = img1, mask1
                imgdict["img2"], imgdict["mask2"] = img2, mask2
        elif self.split in ["train", "val", "test", "train_transfer"]:
            label = np.asarray(Image.open(os.path.join(self.label_root, self.img_name_list[idx] + ".png")))
            transformed = self.transform(image=np.asarray(img), mask=label)
            img, label = transformed["image"].type(torch.float32), transformed["mask"]
            imgdict["img"], imgdict["label"] = img, label

        return imgdict


class LoveDATrain(Dataset):
    def __init__(self, data_dir, config, mode="pretrain"):
        assert mode in ["pretrain", "train"], "Only Support mode: pretrian and train"
        super(LoveDATrain, self).__init__()
        self.dir = data_dir
        self.size = config.data.size
        self.backbone_grid_size = config.network.output_size
        self.img_path = os.path.join(self.dir, "Images_256")
        self.mode = mode
        if self.mode == "pretrain":
            self.transform = get_pretrain_transform(size=self.size, type="image")
            self.interpolate = get_pretrain_transform(type="mask")
        elif self.mode == "train":
            self.transform = get_train_transform(size=self.size)
            self.label_path = os.path.join(self.dir, "Masks_256")

        self.img_name_list = os.listdir(self.img_path)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        imgdict = dict()
        img = Image.open(os.path.join(self.img_path, self.img_name_list[idx]))

        if self.mode == "pretrain":
            img = np.asarray(img.resize(self.size))
            mask1 = np.arange(self.backbone_grid_size * self.backbone_grid_size, dtype=np.uint8).reshape(
                self.backbone_grid_size, self.backbone_grid_size)
            mask2 = np.copy(mask1)
            mask1 = self.interpolate(image=mask1)["image"]
            mask2 = self.interpolate(image=mask2)["image"]
            transformed1 = self.transform(image=img, mask=mask1)
            transformed2 = self.transform(image=img, mask=mask2)
            img1, mask1 = transformed1["image"].type(torch.float32), transformed1["mask"].float()
            img2, mask2 = transformed2["image"].type(torch.float32), transformed2["mask"].float()
            mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(1), size=14, mode="nearest").squeeze()
            mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(1), size=14, mode="nearest").squeeze()

            imgdict["img1"], imgdict["mask1"] = img1, mask1
            imgdict["img2"], imgdict["mask2"] = img2, mask2
        elif self.mode == "train":
            label = np.asarray(Image.open(os.path.join(self.label_path, self.img_name_list[idx])))
            transformed = self.transform(image=np.asarray(img), mask=label)
            img, label = transformed["image"].type(torch.float32), transformed["mask"]
            imgdict["img"], imgdict["label"] = img, label

        return imgdict


class LoveDAVal(Dataset):
    def __init__(self, data_dir, config, mode="pretrain"):
        assert mode in ["pretrain", "train"], "Only Support mode: pretrian and train"
        super(LoveDAVal, self).__init__()
        self.dir = data_dir
        self.size = config.data.size
        self.backbone_grid_size = config.network.output_size
        self.img_path = os.path.join(self.dir, "Images_256")
        self.mode = mode
        if self.mode == "pretrain":
            self.transform = get_pretrain_transform(size=self.size, type="image")
            self.interpolate = get_pretrain_transform(type="mask")
        elif self.mode == "train":
            self.transform = get_train_transform(size=self.size)
            self.label_path = os.path.join(self.dir, "Masks_256")

        self.img_name_list = os.listdir(self.img_path)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        imgdict = dict()
        img = Image.open(os.path.join(self.img_path, self.img_name_list[idx]))

        if self.mode == "pretrain":
            img = np.asarray(img.resize(self.size))
            mask1 = np.arange(self.backbone_grid_size * self.backbone_grid_size, dtype=np.uint8).reshape(
                self.backbone_grid_size, self.backbone_grid_size)
            mask2 = np.copy(mask1)
            mask1 = self.interpolate(image=mask1)["image"]
            mask2 = self.interpolate(image=mask2)["image"]
            transformed1 = self.transform(image=img, mask=mask1)
            transformed2 = self.transform(image=img, mask=mask2)
            img1, mask1 = transformed1["image"].type(torch.float32), transformed1["mask"].float()
            img2, mask2 = transformed2["image"].type(torch.float32), transformed2["mask"].float()
            mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(1), size=14, mode="nearest").squeeze()
            mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(1), size=14, mode="nearest").squeeze()

            imgdict["img1"], imgdict["mask1"] = img1, mask1
            imgdict["img2"], imgdict["mask2"] = img2, mask2
        elif self.mode == "train":
            label = np.asarray(Image.open(os.path.join(self.label_path, self.img_name_list[idx])))
            transformed = self.transform(image=np.asarray(img), mask=label)
            img, label = transformed["image"].type(torch.float32), transformed["mask"]
            imgdict["img"], imgdict["label"] = img, label

        return imgdict


class LoveDATest(Dataset):
    def __init__(self, data_dir, config, mode="pretrain"):
        assert mode in ["pretrain", "test"], "Only Support mode: pretrian and test"
        super(LoveDATest, self).__init__()
        self.dir = data_dir
        self.size = config.data.size
        self.backbone_grid_size = config.network.output_size
        self.img_path = os.path.join(self.dir, "Images_256")
        self.mode = mode
        if self.mode == "pretrain":
            self.transform = get_pretrain_transform(size=self.size, type="image")
            self.interpolate = get_pretrain_transform(type="mask")
        elif self.mode == "test":
            self.transform = get_train_transform(size=self.size)

        self.img_name_list = os.listdir(self.img_path)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        imgdict = dict()
        img = Image.open(os.path.join(self.img_path, self.img_name_list[idx]))

        if self.mode == "pretrain":
            img = np.asarray(img.resize(self.size))
            mask1 = np.arange(self.backbone_grid_size * self.backbone_grid_size, dtype=np.uint8).reshape(
                self.backbone_grid_size, self.backbone_grid_size)
            mask2 = np.copy(mask1)
            mask1 = self.interpolate(image=mask1)["image"]
            mask2 = self.interpolate(image=mask2)["image"]
            transformed1 = self.transform(image=img, mask=mask1)
            transformed2 = self.transform(image=img, mask=mask2)
            img1, mask1 = transformed1["image"].type(torch.float32), transformed1["mask"].float()
            img2, mask2 = transformed2["image"].type(torch.float32), transformed2["mask"].float()
            mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(1), size=56, mode="nearest").squeeze()
            mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(1), size=56, mode="nearest").squeeze()

            imgdict["img1"], imgdict["mask1"] = img1, mask1
            imgdict["img2"], imgdict["mask2"] = img2, mask2
        elif self.mode == "test":
            transformed = self.transform(image=np.asarray(img))
            img = transformed["image"].type(torch.float32)
            imgdict["img"] = img

        return imgdict