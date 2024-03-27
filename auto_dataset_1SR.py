import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random, torch

def get_transform(gray = False):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if gray:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img

class Dataset(data.Dataset):

    def __init__(self, path, component_dict, mode='train'):

        self.mode = mode
        self.component_dict = component_dict

        self.component_path = r'./test_set_8/component'

        train_A_path = r'./test_set_8/source/reference'
        self.trainA = sorted([os.path.join(train_A_path, i) for i in os.listdir(train_A_path)])  # 八个字
        train_B_path = os.path.join(path, 'reference')
        self.trainB = sorted([os.path.join(train_B_path, i) for i in os.listdir(train_B_path)])

        test_A_path = r'./test_set_8/source/test'
        self.testA = sorted([os.path.join(test_A_path, i) for i in os.listdir(test_A_path)])
        test_B_path = os.path.join(path, 'test')
        self.testB = sorted([os.path.join(test_B_path, i) for i in os.listdir(test_B_path)])

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        normalize = transforms.Normalize(mean=mean, std=std)
        self.transform = Compose([transforms.Resize((128, 128)),
                                   transforms.ToTensor(),
                                   normalize])

    def __getitem__(self, index):

        style_list = random.sample(self.trainB, 4)
        style = self.transform(Image.open(style_list[0]).convert('RGB'))

        source, ground_truth, name = None, None, 'name'
        component_img_list = []
        if self.mode == 'train':
            source = self.transform(Image.open(self.trainA[index]).convert('RGB'))

            content_char = self.trainA[index].split('/')[-1].split('.')[0]
            content = bytes((r'\u' + content_char[4:8].lower()).encode()).decode("unicode_escape")
            compo_list = self.component_dict[content]
            for i in range(len(compo_list)):
                compo_path = os.path.join(self.component_path, compo_list[i] + '.jpg')
                component = get_transform()(Image.open(compo_path).convert('RGB').resize((128, 128)))
                component_img_list.append(component)

            ground_truth = self.transform(Image.open(self.trainB[index]).convert('RGB'))

        if self.mode == 'test':
            source = self.transform(Image.open(self.testA[index]).convert('RGB'))

            content_char = self.testA[index].split('/')[-1].split('.')[0]
            content = bytes((r'\u' + content_char[9:14].lower()).encode()).decode("unicode_escape")
            compo_list = self.component_dict[content]
            for i in range(len(compo_list)):
                compo_path = os.path.join(self.component_path, compo_list[i] + '.jpg')
                component = get_transform()(Image.open(compo_path).convert('RGB').resize((128, 128)))
                component_img_list.append(component)

            ground_truth = self.transform(Image.open(self.testB[index]).convert('RGB'))
            name = self.testA[index][-12:]

        return source, ground_truth, style, component_img_list, name

    def __len__(self):

        if self.mode == 'test':
            return len(self.testA)

        return len(self.trainA)
