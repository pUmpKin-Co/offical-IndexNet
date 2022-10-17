import cv2
import torch
import math
import random
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import warnings
import albumentations as A
from torchvision import transforms
from PIL import ImageFilter, ImageOps
from torchvision.transforms import functional as Func
from albumentations.pytorch.transforms import ToTensorV2


def get_pretrain_transform(size=(224, 224), type="image"):
    assert type in ["image", "mask"]

    if type == "image":
        transform = A.Compose(
            [
                A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
                # A.RandomResizedCrop(size[0], size[1], scale=(0.4, 1.0)),

                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                # A.RandomRotate90(p=0.8),


                # A.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8),
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                # A.GaussianBlur((3, 3), (1.5, 1.5), p=0.6),
                A.GaussianBlur(blur_limit=(3, 7),sigma_limit=[0.1, 2.0], p=0.5),
                A.GaussNoise(p=0.6),
                # A.HueSaturationValue(hue_shift_limit=[-30, 30], sat_shift_limit=[-5, 5], val_shift_limit=[-15, 15],p=0.5),
                # A.Solarize(p=0.2),
                A.ToGray(p=0.2),
                A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
                ToTensorV2()
            ]
        )
    else:
        transform = A.Resize(*size, interpolation=cv2.INTER_NEAREST_EXACT)

    return transform


def get_train_transform(size=(224, 224)):
    transform = A.Compose(
        [
            A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.5),
            A.GaussianBlur((3, 3), (1.5, 1.5), p=0.3),
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2()
        ]
    )

    return transform


def get_test_transform(size=(224, 224)):
    transform = A.Compose(
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2()
        ]
    )

    return transform


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if Func._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        coord = None
        for t in self.transforms:
            if 'RandomResizedCropCoord' in t.__class__.__name__:
                img, coord = t(img)
            elif 'FlipCoord' in t.__class__.__name__:
                img, coord = t(img, coord)
            else:
                img = t(img)
        return img, coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return Func.hflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipCoord(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]
            return Func.vflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        coord = torch.Tensor([float(j) / (width - 1), float(i) / (height - 1),
                              float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        return Func.resized_crop(img, i, j, h, w, self.size, self.interpolation), coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_PixPro_transform(aug_type, crop, image_size=224):
    normalize = transforms.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407])

    if aug_type == "InstDisc":  # used in InstDisc and MoCo v1
        transform = Compose([
            RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            RandomHorizontalFlipCoord(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'MoCov2':  # used in MoCov2
        transform = Compose([
            RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    elif aug_type == 'SimCLR':  # used in SimCLR and PIC
        transform = Compose([
            RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'BYOL':
        transform = Compose([
            RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'NULL':  # used in linear evaluation
        transform = Compose([
            RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            RandomHorizontalFlipCoord(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'val':  # used in validate
        transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
    else:
        supported = '[InstDisc, MoCov2, SimCLR, BYOL, NULL, val]'
        raise NotImplementedError(f'aug_type "{aug_type}" not supported. Should in {supported}')

    return transform
