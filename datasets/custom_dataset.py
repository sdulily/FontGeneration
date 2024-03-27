"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import torch
import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import numpy as np
import json
import random


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    # images = []
    # dir = os.path.expanduser(dir)
    # for target in sorted(class_to_idx.keys()):
    #     d = os.path.join(dir, target)
    #     if not os.path.isdir(d):
    #         continue
    #
    #     for root, _, fnames in sorted(os.walk(d)):
    #         for fname in sorted(fnames):
    #             if has_file_allowed_extension(fname, extensions):
    #                 path = os.path.join(root, fname)
    #                 item = (path, class_to_idx[target])
    #                 images.append(item)
    sources = []
    references = []
    dirA = os.path.join(os.path.expanduser(dir), 'source')
    for root, _, fnames in sorted(os.walk(dirA)):
        for name in sorted(fnames):
            if has_file_allowed_extension(name, extensions):
                path = os.path.join(root, name)
                sources.append(path)
    dirB = os.path.join(os.path.expanduser(dir), 'train')
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dirB, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    references.append(item)

    return sources, references


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        src_paths, samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.src_paths = src_paths
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform
        # self.comp_transform = comp_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        dir = os.path.join(dir, 'train')
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imgname = path.split('/')[-1].replace('.JPEG', '')
        return sample, target, imgname

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


with open('./file_6763.json', 'r+') as file:
    content = file.read()
component_dict = json.loads(content)

MAX_COMP = 15

class ImageFolerRemap(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, remap_table=None, with_idx=False):
        super(ImageFolerRemap, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)

        self.references = self.samples
        self.class_table = remap_table
        self.com_dict = component_dict
        self.component_path = os.path.join(root, 'component')
        self.reference_path = os.path.join(root, 'train')
        self.with_idx = with_idx
        # self.comp_transform = comp_transform

    def __getitem__(self, index):
        lenth = len(self.src_paths)
        idx = np.random.randint(0, lenth)
        source_path = self.src_paths[idx]

        char_name = source_path.split('/')[-1].split('.')[0].split('\\')[-1]
        uni_code = bytes((r'\u' + char_name[4:].lower()).encode()).decode("unicode_escape")

        compo_list = self.com_dict[uni_code]
        component_list = []
        for i in range(len(compo_list)):
            compo_path = os.path.join(self.component_path, compo_list[i] + '.jpg')
            compo_img = self.loader(compo_path)
            compo_img = self.transform(compo_img)
            component_list.append(compo_img)

        path, target = self.samples[index]

        ref_name = path.split('\\')[-2]

        # 正样本
        ref_img_name = path.split('\\')[-1]
        positive_path = os.path.join(self.reference_path, ref_name)
        all_imgs = os.listdir(positive_path)
        all_imgs.remove(ref_img_name)
        positive_names = random.sample(all_imgs, 5)

        positive_imgs = []
        for i in range(len(positive_names)):
            positive_img_path = os.path.join(positive_path, positive_names[i])
            positive_img = self.loader(positive_img_path)
            positive_img = self.transform(positive_img)
            positive_imgs.append(positive_img)
        #####

        # 负样本
        font_names = os.listdir(self.reference_path)
        font_names.remove(ref_name)
        negative_names = random.sample(font_names, 5)

        negative_imgs = []
        for i in range(len(negative_names)):
            negative_path = os.path.join(self.reference_path, negative_names[i], char_name + '.jpg')
            negative_img = self.loader(negative_path)
            negative_img = self.transform(negative_img)
            negative_imgs.append(negative_img)
        ######

        source = self.loader(source_path)
        reference = self.loader(path)
        if self.transform is not None:
            source = self.transform(source)
            reference = self.transform(reference)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = self.class_table[target]
        if self.with_idx:
            return reference, index, target
        # num_comp = len(component_list)
        C, H, W = component_list[0].shape
        for i in range(MAX_COMP - len(component_list)):
            component_list.append(torch.zeros((C, H, W)))

        return source, reference, positive_imgs, negative_imgs, component_list, target


class CrossdomainFolder(data.Dataset):
    def __init__(self, root, data_to_use=['photo', 'monet'], transform=None, loader=default_loader, extensions='jpg'):
        self.data_to_use = data_to_use
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in self.data_to_use]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d in self.data_to_use]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
