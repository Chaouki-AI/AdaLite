# This code is mostly taken from BTS; author: Jin Han Lee
# and modified by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# it contains the dataloader used to read, split, preprosess and build the dataloaders
# please Modify the args on the args_****.txt files to controle the Batch size and the number of images used for training 
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications


import os
import random
import copy
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])
class NoOpTransform:
    def __call__(self, x):
        return x

class DepthDataLoader(object):
    def __init__(self, args, images, depths, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, images, depths, mode, transform=preprocessing_transforms(mode))
            self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.bs,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=8,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, images, depths, mode, transform=preprocessing_transforms(mode))
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=8)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, images, depths, mode, transform=None, is_for_online_eval=False):
        self.args = args

        self.images = images
        self.depths = depths
        self.indexes = [i for i in range(len(self.images))]

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def splice_images(self, img1_rgb, img2_rgb, img1_out, img2_out, split_width=5, split_value=0):
        """
        Combines two images and their outputs either horizontally or vertically, with a split between.
        
        Args:
            img1_rgb (np.ndarray): First RGB image (H, W, 3)
            img2_rgb (np.ndarray): Second RGB image (H, W, 3)
            img1_out (np.ndarray): First grayscale output (H, W, 1)
            img2_out (np.ndarray): Second grayscale output (H, W, 1)
            split_width (int): Width of the gap between the parts
            split_value (int or float): Pixel value to fill the gap (default = 0)
            
        Returns:
            composite_rgb (np.ndarray): Combined RGB image with split
            composite_out (np.ndarray): Combined grayscale output with split
        """
        
        assert img1_rgb.shape == img2_rgb.shape, "RGB images must have same shape"
        assert img1_out.shape == img2_out.shape, "Output images must have same shape"

        cut_ratio = random.randint(3, 7) / 10
        axis = np.random.choice([0, 1])                # 0 = horizontal, 1 = vertical
        random_cut = np.random.choice([True, False])

        H, W, _ = img1_rgb.shape

        if axis == 0:  # Horizontal
            
            cut = int(H * cut_ratio)
            
            # RGB split and separator
            rgb_top = img1_rgb[:cut, :, :]
            rgb_bottom = img2_rgb[cut:, :, :]
            rgb_split = np.full((split_width, W, 3), fill_value=split_value, dtype=img1_rgb.dtype)
            composite_rgb = np.concatenate([rgb_top, rgb_split, rgb_bottom], axis=0)

            # Grayscale output split and separator
            out_top = img1_out[:cut, :, :]
            out_bottom = img2_out[cut:, :, :]
            out_split = np.full((split_width, W, 1), fill_value=split_value, dtype=img1_out.dtype)
            composite_out = np.concatenate([out_top, out_split, out_bottom], axis=0)

        else:  # Vertical
            cut = int(W * cut_ratio)
            
            # RGB split and separator
            rgb_left = img1_rgb[:, :cut, :]
            rgb_right = img2_rgb[:, cut:, :]
            rgb_split = np.full((H, split_width, 3), fill_value=split_value, dtype=img1_rgb.dtype)
            composite_rgb = np.concatenate([rgb_left, rgb_split, rgb_right], axis=1)

            # Grayscale output split and separator
            out_left = img1_out[:, :cut, :]
            out_right = img2_out[:, cut:, :]
            out_split = np.full((H, split_width, 1), fill_value=split_value, dtype=img1_out.dtype)
            composite_out = np.concatenate([out_left, out_split, out_right], axis=1)

        return composite_rgb, composite_out


    def __getitem__(self, idx):
        if self.mode == 'train':
            sun = [i for i in self.images] 
            #nyu = [i for i in self.images if 'NYUv2' in i] 
            image_path, depth_path = self.images[idx], self.depths[idx]
            if random.random() >= 0.5 : 
                pick = random.choice(range(len(sun))) #if 'SUNRGBD' in image_path else random.choice(range(len(nyu)))
            else : 
                pick = None
            image_path2 =  self.images[pick] if pick is not None else None
            depth_path2 =  self.depths[pick] if pick is not None else None

            image, image2        = Image.open(image_path), Image.open(image_path2) if pick is not None else None
            depth_gt, depth_gt2  = Image.open(depth_path), Image.open(depth_path2) if pick is not None else None


            
            if self.args.dataset == 'kitti':
                height, height2 = image.height, image2.height if pick is not None else None   
                width, width2   = image.width , image2.width if pick is not None else None
                
                top_margin, top_margin2 = int(height - 352), int(height2 - 352) if pick is not None else None
                left_margin, left_margin2 = int((width - 1216) / 2), int((width2 - 1216) / 2) if pick is not None else None
                
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                depth_gt2 = depth_gt2.crop((left_margin2, top_margin2, left_margin2 + 1216, top_margin2 + 352)) if pick is not None else None
                
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image2 = image2.crop((left_margin2, top_margin2, left_margin2 + 1216, top_margin2 + 352)) if pick is not None else None

            # To avoid blank boundaries due to pixel registration
            else :# self.args.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                depth_gt2 = depth_gt2.crop((43, 45, 608, 472)) if pick is not None else None
                
                image = image.crop((43, 45, 608, 472))
                image2 = image2.crop((43, 45, 608, 472)) if pick is not None else None
                
            if random.random() > 0.5:
                random_angle, random_angle2 = (random.random() - 0.5) * 2 * 2.5 , (random.random() - 0.5) * 2 * 2.5 if pick is not None else None
                image, image2 = self.rotate_image(image, random_angle), self.rotate_image(image2, random_angle2) if pick is not None else None
                depth_gt, depth_gt2 = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST), self.rotate_image(depth_gt2, random_angle2, flag=Image.NEAREST) if pick is not None else None

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.bitwise_or(np.right_shift(depth_gt, 3), np.left_shift(depth_gt, 16 - 3)) if 'SUNRGBD' in depth_path else depth_gt
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if pick is not None : 
                image2 = np.asarray(image2, dtype=np.float32) / 255.0  if pick is not None else None
                depth_gt2 = np.bitwise_or(np.right_shift(depth_gt2, 3), np.left_shift(depth_gt2, 16 - 3)) if 'SUNRGBD' in depth_path2 else depth_gt2
                depth_gt2 = np.asarray(depth_gt2, dtype=np.float32)    if pick is not None else None
                depth_gt2 = np.expand_dims(depth_gt2, axis=2)          if pick is not None else None

            
            if self.args.dataset == 'nyu' or self.args.dataset == 'sunrgbd':
                #depth_gt = np.bitwise_or(np.right_shift(depth_gt, 3), np.left_shift(depth_gt, 16 - 3)) if 'SUNRGBD' in depth_path else depth_gt
                depth_gt = depth_gt.astype(np.single)/1000.  if 'SUNRGBD' in depth_path else depth_gt/1000.
                #print(depth_gt.shape)
                depth_gt = np.where(depth_gt>8, 8, depth_gt) if 'SUNRGBD' in depth_path else depth_gt
                #depth_gt[depth_gt > 8] = 8 if 'SUNRGBD' in depth_path else depth_gt
                #depth_gt  = Image.fromarray(depth_gt) if 'SUNRGBD' in depth_path else depth_gt


                if pick is not None : 
                    
                    #depth_gt2 = np.bitwise_or(np.right_shift(depth_gt2, 3), np.left_shift(depth_gt2, 16 - 3)) if 'SUNRGBD' in depth_path2 else depth_gt2
                    depth_gt2 = depth_gt2.astype(np.single)/1000. if 'SUNRGBD' in depth_path2 else depth_gt2/1000.
                    depth_gt2 = np.where(depth_gt2>8, 8, depth_gt2) if 'SUNRGBD' in depth_path else depth_gt2
                    #depth_gt2 = Image.fromarray(depth_gt2) if 'SUNRGBD' in depth_path else depth_gt2

            #print(image.shape, image2.shape, depth_gt.shape, depth_gt2.shape)

            #image, depth_gt = self.splice_images(image, image2, depth_gt, depth_gt2) if pick is not None else (image, depth_gt)
            

            
            if self.args.dataset == 'kitti':
                depth_gt = depth_gt / 256.0

            image, depth_gt = self.random_crop(image, depth_gt, self.args.image_height, self.args.image_width)
            
            #if pick is None :
            #    image,depth_gt = self.cutflip(image,depth_gt,None) 

            image, depth_gt = self.train_preprocess(image, depth_gt)
            #print(image.shape, depth_gt.shape)
            sample = {'image': image, 'depth': depth_gt}

        else:
            
            image_path = self.images[idx]
            #print(image_path)
            image = Image.open(image_path)

            if self.mode == 'online_eval':
                depth_path = self.depths[idx]
                mask = depth_path.replace('depth.npy', 'depth_mask.npy') if self.args.dataset == 'diode' else None
                has_valid_depth = False
                try:
                    #depth_gt = Image.open(depth_path)
                    depth_gt = np.load(depth_path) if self.args.dataset == 'diode' else Image.open(depth_path)
                    mask_gt  = np.load(mask) if self.args.dataset == 'diode' else None
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                if has_valid_depth:
                    if self.args.dataset == 'diode':
                        depth_gt = (depth_gt*mask_gt[..., None]).squeeze()
                    depth_gt = np.asarray(depth_gt, dtype=np.float32) if self.args.dataset != 'sunrgbd' else depth_gt
                    #depth_gt = np.bitwise_or(np.right_shift(depth_gt, 3), np.left_shift(depth_gt, 16 - 3)) if self.args.dataset == 'sunrgbd'  else depth_gt

                    image    = np.asarray(image, dtype=np.float32) / 255.0
                    if self.args.dataset == 'nyu' or self.args.dataset == 'sunrgbd':
                        if self.args.dataset == 'sunrgbd' :
                            depth_gt = np.bitwise_or(np.right_shift(depth_gt, 3), np.left_shift(depth_gt, 16 - 3))
                            depth_gt = depth_gt.astype(np.single) / 1000
                            depth_gt[depth_gt > 8] = 8
                        else : 
                            depth_gt = depth_gt / 1000.
                    elif self.args.dataset == 'kitti':
                        depth_gt = depth_gt / 256.0
                    depth_gt = np.expand_dims(depth_gt, axis=2)


                
                if self.args.dataset == 'kitti':
                    height = image.shape[0]
                    width = image.shape[1]
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    if self.mode == 'online_eval' and has_valid_depth:
                        depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    

                if self.mode == 'online_eval':
                    sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        #do_faroff = random.random()
        #if do_faroff > 0.5:
        #    values = list(set([int(i) for i in depth_gt.flatten().tolist()]))
        #    if len(values)>3 : 
        #        min = values[-2]
        #        image = np.where(depth_gt>= min, depth_gt, image) #SSif max > 0.5 else image
        return image, depth_gt
    def cutflip(self,image,depth,h_graft=None):

        p = random.random()
        if p>0.3 and p<0.7 :
            return image,depth
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        h,w,c = image.shape
        N = 2    # split numbers
        h_list=[]      
        h_interval_list = []        # hight interval
        for i in range(N-1):
            if h_graft!=None:
                h_list.append(h_graft)
            else:
                h_list.append(random.randint(int(0.2*h),int(0.8*h)))
        h_list.append(h)
        h_list.append(0)  
        h_list.sort()
        h_list_inv = np.array([h]*(N+1))-np.array(h_list)
        for i in range(len(h_list)-1):
            h_interval_list.append(h_list[i+1]-h_list[i])

        for i in range(N):
            image[h_list[i]:h_list[i+1],:,:] = image_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i],:,:]
            depth[h_list[i]:h_list[i+1],:,:] = depth_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i],:,:]
        return image,depth
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.images)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        #self.normalize = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #self.normalize  = transforms.Compose([NoOpTransform()])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth}
        else:
            if self.mode == 'visualize':
                return {'image': image, 'depth': depth}
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth,'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
