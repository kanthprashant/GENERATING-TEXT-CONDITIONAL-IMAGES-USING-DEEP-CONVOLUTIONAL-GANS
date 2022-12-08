from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.autograd as autograd
from configs import config
import pickle
import numpy as np

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + logsigma**2 - mu**2 - sigma**2)
    KLD_element = 1 + logvar - mu.pow(2) - logvar.exp()
    KLD = torch.mean(KLD_element).mul(-0.5)
    return KLD

def L1_loss(criterion, fake_imgs, real_images):
    errG_L1 = criterion(fake_imgs , real_images)
    return errG_L1

def compute_gp(netD, real_imgs, fake_imgs, cond, gpus):
        batch_size = real_imgs.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).float().cuda()
        eps = eps.expand_as(real_imgs)
        interpolation = eps * real_imgs + (1 - eps) * fake_imgs

        interp_features = nn.parallel.data_parallel(netD, (interpolation), gpus)
        inputs = (interp_features, cond)
        interp_logits = nn.parallel.data_parallel(netD.cond_discriminator_logits, inputs, gpus)
        grad_outputs = torch.ones_like(interp_logits).cuda()

        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

def compute_discriminator_loss(netD, criterion, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions, gpus):
    # criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions # conditions.detach()
    fake = fake_imgs.detach()
    real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)

    # real pairs
    inputs = (real_features, cond)
    real_logits = nn.parallel.data_parallel(netD.cond_discriminator_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = nn.parallel.data_parallel(netD.cond_discriminator_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.cond_discriminator_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.uncond_discriminator_logits is not None:
        real_logits = nn.parallel.data_parallel(netD.uncond_discriminator_logits, (real_features), gpus)
        fake_logits = nn.parallel.data_parallel(netD.uncond_discriminator_logits, (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real, errD_wrong, errD_fake

def compute_generator_loss(netD, criterion, fake_imgs, real_images, 
                           real_labels, conditions, gpus):
    # criterion = nn.BCELoss()
    batch_size = real_images.size(0)
    cond = conditions # conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)

    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.cond_discriminator_logits, inputs, gpus)
    errG_fake = criterion(fake_logits, real_labels)

    if netD.uncond_discriminator_logits is not None:
        fake_logits = nn.parallel.data_parallel(netD.uncond_discriminator_logits, (fake_features), gpus)
        uncond_errG_fake = criterion(fake_logits, real_labels)
        errG_fake += uncond_errG_fake

    return errG_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('GroupNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def save_img_results(data_img, fake, low_res, caption, epoch, image_dir):
    num = config.VIS_COUNT
    fake = fake.cpu().data
    fake = fake[0:num]
    if caption is not None:
        with open(image_dir+f"/{epoch}_cpt.txt", 'w') as f:
            for cpt in caption[:num]:
                f.write(cpt + '\n')
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img.cpu().data
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples_%03d.png' % 
            (image_dir, epoch), normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
        if low_res is not None:
            low_res = low_res.cpu().data
            low_res = low_res[:num]
            vutils.save_image(
            low_res, '%s/stage1_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)

def load_from_checkpoint(netG, gen_ckpt, netD=None, d_ckpt=None):
    if gen_ckpt is not None:
        state_dict = torch.load(gen_ckpt, map_location=lambda storage, loc:storage)
        epoch = state_dict['epoch']
        state = state_dict['state']
        netG.load_state_dict(state)
        print(f"generator loaded from {gen_ckpt}, starting at epoch: {epoch}")
        if d_ckpt is None:
            return netG
    if d_ckpt is not None:
        state_dict = torch.load(d_ckpt, map_location=lambda storage, loc:storage)
        epoch = state_dict['epoch']
        state = state_dict['state']
        netD.load_state_dict(state)
        print(f"discriminator loaded from {d_ckpt}, starting at epoch: {epoch}")
    return epoch, netG, netD

def save_model(netG, netD, epoch, model_dir, stage):
    state_dict = {'epoch': epoch, 'state': netG.state_dict()}
    torch.save(state_dict, '%s/netG%d_epoch_%d.pth' % (model_dir, stage, epoch))
    state_dict = {'epoch': epoch, 'state': netD.state_dict()}
    torch.save(state_dict, '%s/netD%d_epoch_%d.pth' % (model_dir, stage, epoch))
    print('Save G1/D1 models')

def make_list(imageListpath, captionsListPath):
    '''
    Args:
        imageListpath:  Path of file containing image list
        captionsListPath: Path of file containing captions of each image
    Returns:
        imageList:  list of images, where each image occurs equal to the number of captions it has
        captionList:    list of captions, in same order as images in imageList
        satisfies -> len(imageList) == len(captionList)
    '''
    with open(imageListpath, 'rb') as f:
        images = pickle.load(f)
    with open(captionsListPath, 'rb') as f:
        captions = pickle.load(f)
    imageList = []
    captionList = []
    for id, img in images.items():
        n_captions = len(captions[id])
        im = [img]*n_captions
        caps = captions[id]
        imageList.extend(im)
        captionList.extend(caps) 
    return imageList, captionList

def get_data(imageList, captionList, ids):
    '''
    Args:
        imageList:  list of images, where each image occurs equal to the number of captions it has
        captionList:    list of captions, in same order as images in imageList
        ids:    train or test ids
    Returns:
        images: image list corresponding to ids
        captions:   captions list corresponding to ids
    '''
    images = []
    captions = []
    for id in ids:
        images.append(imageList[id])
        captions.append(captionList[id])
    return images, captions

def make_train_test_split(imageListPath, captionsListPath, test_size):
    '''
    Args:
        imageListPath:  Path of file containing image list
        captionsListpath:   Path of file containing captions of each image
        test_size:  size of test_data, compared to length of dataset
    Returns:
        train_images, train_captions, test_images, test_captions
    '''
    imageList, captionList = make_list(imageListPath, captionsListPath)
    img_ids = np.arange(0, len(imageList))
    test_ids = np.random.choice(img_ids, size = int(test_size*len(imageList)), replace = False)
    train_ids = [id for id in img_ids if id not in test_ids]
    train_images, train_captions = get_data(imageList, captionList, train_ids)
    test_images, test_captions = get_data(imageList, captionList, test_ids)
    return train_images, train_captions, test_images, test_captions

def make_train_test_split2(imageListPath, captionsListPath, test_size):
    '''
    Args:
        imageListPath:  Path of file containing image list
        captionsListpath:   Path of file containing captions of each image
        test_size:  size of test_data, compared to length of dataset
    Returns:
        train_images, train_captions, test_images, test_captions
    '''
    with open(imageListPath, 'rb') as f:
        imgs = pickle.load(f)
    with open(captionsListPath, 'rb') as f:
        captions = pickle.load(f)
    img_ids = np.arange(0, len(imgs))
    test_ids = np.random.choice(img_ids, size = int(test_size*len(imgs)), replace = False)
    train_ids = [id for id in img_ids if id not in test_ids]
    train_images = [imgs[k] for k in train_ids]
    train_captions = [captions[k] for k in train_ids]
    test_images = [imgs[k] for k in test_ids]
    test_captions = [captions[k] for k in test_ids]
    return train_images, train_captions, test_images, test_captions

def load_data(imageListPath, captionsListPath):
    '''
    Args:
        imageListPath:  Path of file containing image list
        captionsListpath:   Path of file containing captions of each image
    Returns:
        train_images, train_captions
    '''
    with open(imageListPath, 'rb') as f:
        train_images = pickle.load(f)
    with open(captionsListPath, 'rb') as f:
        train_captions = pickle.load(f)
    return train_images, train_captions