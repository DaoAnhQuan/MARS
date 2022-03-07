import torch
import torch.utils.data as data_utl
import torchvision

import json
import random
import os
import math


def load_rgb_frames(image_dir, vid, start, num, num_frame):
    frames = []
    i = start
    count = 0
    while count < num:
        if (i + 1) > num_frame:
            i = 0
        path = os.path.join(image_dir, vid, 'frame' + str(i + 1).zfill(6) + '.jpg')
        if not os.path.exists(path):
            raise RuntimeError("RGB file does not exist!")
        try:
            img = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)
        except:
            print('Failed: ', path)
        img = (img / 255.) * 2 - 1
        frames.append(img)
        i += 1
        count += 1
    return frames


def load_flow_frames(image_dir, vid, start, num, num_frame, mode='single'):
    frames = []
    i = start
    count = 0
    while count < num:
        if (i + 1) > num_frame:
            i = 0
        x_path = os.path.join(image_dir, 'u', vid, 'frame' + str(i + 1).zfill(6) + '.jpg')
        y_path = os.path.join(image_dir, 'v', vid, 'frame' + str(i + 1).zfill(6) + '.jpg')
        if (not os.path.exists(x_path)) or (not os.path.exists(y_path)):
            raise RuntimeError("Flow file does not exist!")
        imgx = torchvision.io.read_image(x_path, mode=torchvision.io.ImageReadMode.GRAY)
        imgy = torchvision.io.read_image(y_path, mode=torchvision.io.ImageReadMode.GRAY)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        if mode == 'single':
            img = torch.cat([imgx, imgy])
        else:
            img = torch.cat([imgx, imgy, imgy])
        frames.append(img)
        i += 1
        count += 1
    return frames


def make_dataset(split_dir, split_ord, split, rgb_root, flow_root, num_classes=51):
    dataset = []
    all_split_files = os.listdir(split_dir)
    split_files = []
    label_path = 'label.json'
    if os.path.exists(label_path):
        label_file = open(label_path, 'r')
        labels = json.load(label_file)
        label_file.close()
        for key in labels.keys():
            label_name = key
            labels_path = os.path.join(split_dir, label_name + '_test_split' + str(split_ord) + '.txt')
            split_files.append(labels_path)
    else:
        raise ("Label file does not exist.")

    split_mode = 1
    if split == 'test':
        split_mode = 2
    i = 0
    for file_path in split_files:
        f = open(file_path, 'r')
        data = f.readlines()
        for vid in data:
            vid_name = vid.split()[0].replace('.avi', '')
            vid_id = int(vid.split()[1])
            if vid_id != split_mode:
                continue
            num_frames = min(len(os.listdir(os.path.join(rgb_root, vid_name))),
                             len(os.listdir(os.path.join(flow_root, 'v', vid_name))))
            repeat = 1
            if num_frames < 64:
                repeat = math.ceil(64 / num_frames)
            dataset.append((vid_name, i, num_frames, repeat))
        i += 1

    return dataset


class HMDB51(data_utl.Dataset):
    def __init__(self, split_dir, split_ord, split, rgb_root, flow_root, mode, num_class, transforms=None):
        self.data = make_dataset(split_dir, split_ord, split, rgb_root, flow_root, num_class)
        self.flow_root = flow_root
        self.rgb_root = rgb_root
        self.transforms = transforms
        self.mode = mode
        self.split = split

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf, repeat = self.data[index]
        if self.split == 'train':
            clip = 64
            start_f = random.randint(0, nf * repeat - clip)
        else:
            if nf > 250:
                clip = 250
            else:
                clip = nf
            start_f = 0

        if self.mode == 'rgb':
            rgbs = load_rgb_frames(self.rgb_root, vid, start_f, clip, nf)
            rgbs = torch.stack(rgbs)
            rgbs = rgbs.transpose(0, 1)
            if self.transforms is not None:
                rgbs = self.transforms(rgbs)
            return rgbs, label
        elif self.mode == 'flow':
            flows = load_flow_frames(self.flow_root, vid, start_f, clip, nf)
            flows = torch.stack(flows)
            flows = flows.transpose(0, 1)
            if self.transforms is not None:
                flows = self.transforms(flows)
            return flows, label
        else:
            rgbs = load_rgb_frames(self.rgb_root, vid, start_f, clip, nf)
            flows = load_flow_frames(self.flow_root, vid, start_f, clip, nf, 'dual')
            h = min(rgbs[0].size()[1], flows[0].size()[1])
            w = min(rgbs[0].size()[2], flows[0].size()[2])
            rgbs = torch.stack(rgbs)
            flows = torch.stack(flows)
            imgs = torch.cat((rgbs[:, :, 0:h, 0:w], flows[:, :, 0:h, 0:w]))
            if self.transforms is not None:
                imgs = self.transforms(imgs)
            rgbs = imgs[0:clip, :, :, :]
            flows = imgs[clip:, 0:2, :, :]
            rgbs = rgbs.transpose(0, 1)
            flows = flows.transpose(0, 1)
            return rgbs, flows, label

    def __len__(self):
        return len(self.data)