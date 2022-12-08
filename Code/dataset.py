import cv2 as cv
import numpy as np
import os
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def gauss_noise_tensor(x):
    '''
    Args:
        img:    image tensor
    Returns:
        out:    image tensor mixed with gaussian noise
    '''
    assert isinstance(x, torch.Tensor)
    sigma = 0.1
    return x + sigma * torch.randn_like(x)

# def gauss_noise_tensor(img):
#     '''
#     Args:
#         img:    image tensor
#     Returns:
#         out:    image tensor mixed with gaussian noise
#     '''
#     # borrowed from @vfdev-5 https://github.com/pytorch/vision/issues/6192
#     
#     assert isinstance(img, torch.Tensor)
#     dtype = img.dtype
#     if not img.is_floating_point():
#         img = img.to(torch.float32)
# 
#     sigma = 0
#     out = img + sigma * torch.randn_like(img)
# 
#     if out.dtype != dtype:
#         out = out.to(dtype)
#         
#     return out

def transformations(imageSize, augmentImage):
    '''
    Args:
        imageSize:  input image size to image encoder
        augmentImage:   True/False
    Return:
        transform:  transform object containing sequence 
                    of transforms to be applied on an image
    '''
    if augmentImage:
        transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.RandomApply([gauss_noise_tensor,], p = 0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(imageSize, interpolation=Image.NEAREST),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
    return transform

class Text2ImgDataset(Dataset):
    '''
    Args:
        imageFolder:    path of imageFolder
        tokenizer:      tokenizer object (DistilBertTokenizer)
        images:         list of images
        captions:       list of captions
        imageSize:      input image size to image encoder
        augmentImage:   True/False
    Returns:
        items:  dictionary conatining 'input_ids', 'attention_mask' and 'image'
    '''
    def __init__(self, imageFolder, tokenizer, text_encoder, images, captions, imageSize, augmentImage=False):
        self.imageFolder = imageFolder
        self.images = images
        self.captions = captions
        # captions_dict = {'input_ids': [list of captions vector], 'attention_mask': [list of attention_mask]}
        if text_encoder == "distilbert-base-uncased":
            self.captions_dict = tokenizer(captions, padding='max_length', truncation=True, max_length=80, return_tensors="pt")
        elif text_encoder == "openai/clip-vit-base-patch32":
            self.captions_dict = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        self.transform = transformations(imageSize, augmentImage)
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        items = {key: value[index] for key, value in self.captions_dict.items()}
        img = cv.imread(os.path.join(self.imageFolder, self.images[index]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.transform(Image.fromarray(img.astype(np.uint8)))
        items['image'] = img.float().cuda()
        return items

class Text2ImgDataset_reformed(Dataset):
    '''
    Args:
        imageFolder:    path of imageFolder
        tokenizer:      tokenizer object (DistilBertTokenizer)
        text_encoder:   type of text encoder
        images:         list of images
        captions:       list of lists of captions
        imageSize:      input image size to image encoder
        augmentImage:   True/False
    Returns:
        items:  dictionary conatining 'input_ids', 'attention_mask', 'image' and 'caption'
    '''
    def __init__(self, imageFolder, tokenizer, text_encoder, images, captions, imageSize, augmentImage=False):
        self.imageFolder = imageFolder
        self.images = images
        self.captions = captions
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transform = transformations(imageSize, augmentImage)
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        caption_list = self.captions[index]
        caption_idx = random.randint(0, len(caption_list)-1)
        caption = caption_list[caption_idx]
        # captions_dict = {'input_ids': [list of captions vector], 'attention_mask': [list of attention_mask]}
        captions_dict = self.tokenizer(caption, padding='max_length', truncation=True, max_length=77, return_tensors="pt")
        items = {key: value[0] for key, value in captions_dict.items()}
        img = cv.imread(os.path.join(self.imageFolder, self.images[index]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.transform(Image.fromarray(img.astype(np.uint8)))
        items['image'] = img.float().cuda()
        items['caption'] = caption
        return items

class Text2ImgDataset_cnnrnn(Dataset):
    '''
    Args:
        imageFolder:    path of imageFolder
        tokenizer:      tokenizer object (DistilBertTokenizer)
        text_encoder:   type of text encoder
        images:         list of images
        captions:       list of lists of captions
        imageSize:      input image size to image encoder
        augmentImage:   True/False
    Returns:
        items:  dictionary conatining 'input_ids', 'attention_mask', 'image' and 'caption'
    '''
    def __init__(self, imageFolder, tokenizer, text_encoder, images, captions, imageSize, augmentImage=False):
        self.imageFolder = imageFolder
        self.images = images
        self.captions = captions
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transform = transformations(imageSize, augmentImage)
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        caption_list = self.captions[index]
        idx = random.randint(0, len(caption_list)-1)
        caption = caption_list[idx]
        embedding_list = self.tokenizer[index]
        embedding = embedding_list[idx]
        img = cv.imread(os.path.join(self.imageFolder, self.images[index]+'.jpg'))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.transform(Image.fromarray(img.astype(np.uint8)))
        items = {'image': img.float().cuda(),
                 'caption': caption,
                 'embedding': torch.tensor(embedding).float().cuda()
                }
        return items
