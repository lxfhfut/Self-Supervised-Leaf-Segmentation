import os
import cv2
import glob
import time
import torch
import numpy as np
import shutil
from PIL import Image
from pathlib import Path
import statistics
from tqdm import tqdm
import torchvision.transforms.functional as trf
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from patchify import patchify, unpatchify

import pathlib
import imutils
from torch import nn, optim
# from evaluation import FGBGDice, image_quality_metrics

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
# from leaf_seg import cal_leaf_area
torch.manual_seed(0)


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', SIZE=256, isAugmented=True):
        if split == 'train':
            if isAugmented:
                print('Augmentations applied!')
                self.transforms = transforms.Compose([
                    transforms.Resize((SIZE, SIZE), trf.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),  # A little data augmentation!
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAdjustSharpness(2.0),
                    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2)),
                    transforms.ColorJitter(0.2, 0.1, 0.1),
                    transforms.RandomAffine(degrees=(-60, 60), scale=(0.8, 1.2))
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize((SIZE, SIZE), trf.InterpolationMode.BICUBIC),
                ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), trf.InterpolationMode.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


# A handy function to make our dataloaders
def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=n_workers, pin_memory=pin_memory)
    return dataloader


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]  # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)

        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


def train_model(model, train_dl, val_dl, epochs, display_every=200):
    for e in range(epochs):
        loss_meter_dict = create_loss_meters()  # function returning a dictionary of objects to
        i = 0                                   # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))  # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict)  # function to print out the losses
                data = next(iter(val_dl))  # getting a batch for visualizing the model output after fixed intrvals
                visualize(model, data, save=False)  # function displaying the model's outputs


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)

            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


def test(model, image_dir):
    test_paths = glob.glob(image_dir + '/*.jpg')
    test_dl = make_dataloaders(paths=test_paths, split='val')
    print(len(test_paths))
    batch_cnt = 0
    for data in tqdm(test_dl):
        model.net_G.eval()
        with torch.no_grad():
            model.setup_input(data)
            model.forward()
        fake_color = model.fake_color.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = lab_to_rgb(L, real_color)
        batch_cnt += 1
        for i in range(len(fake_imgs)):
            cv2.imwrite(os.path.join(image_dir, 'results', 'fake_batch_' + str(batch_cnt) + '_img_' + str(i) + '.png'),
                        cv2.cvtColor((fake_imgs[i]*255).astype('uint8'), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(image_dir, 'results', 'real_batch_' + str(batch_cnt) + '_img_' + str(i) + '.png'),
                        cv2.cvtColor((real_imgs[i]*255).astype('uint8'), cv2.COLOR_RGB2BGR))


def generate_blocks(image_path, blk_sz=256):
    img = cv2.imread(image_path)
    rows, cols = img.shape[:2]
    img = img[:rows//blk_sz*blk_sz, :cols//blk_sz*blk_sz, :]
    rows, cols = img.shape[:2]
    image_name = os.path.basename(image_path)
    base_dir = Path(image_path).parent.absolute()
    blk_dir = os.path.join(base_dir, 'blks')
    Path(blk_dir).mkdir(parents=True, exist_ok=True)
    blk_count = 0
    for row in range(0, rows - blk_sz, blk_sz):
        for col in range(0, cols - blk_sz, blk_sz):
            blk_count += 1
            block = img[row:row + blk_sz, col:col + blk_sz, :]
            cv2.imwrite(os.path.join(blk_dir, image_name[:-9] + str(blk_count) + ".jpg"), block)
    print("{} blocks generated".format(blk_count))


def test_one_image(model, image_path, blk_sz=256):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    # img = imutils.resize(img, width=512)
    rows, cols = img.shape[:2]
    padded_rows = (blk_sz - rows % blk_sz) if rows % blk_sz else 0
    padded_cols = (blk_sz - cols % blk_sz) if cols % blk_sz else 0
    # print("Image has been resized from ({}, {}) to ({}, {})".format(rows, cols, rows+padded_rows, cols+padded_cols))
    padded_img = np.pad(img, ((0, padded_rows), (0, padded_cols), (0, 0)), 'reflect')
    # img = img[:rows // blk_sz * blk_sz, :cols // blk_sz * blk_sz, :]
    patches = patchify(padded_img, (blk_sz, blk_sz, 3), step=blk_sz)
    padded_shape = patches.shape
    patches = patches.reshape((-1, blk_sz, blk_sz, 3))  # shape = (*, blk_sz, blk_sz, 3)
    corrected_patches = np.zeros_like(patches)

    L_patches = torch.zeros((patches.shape[0], 1, blk_sz, blk_sz), dtype=torch.float32)
    ab_patches = torch.zeros((patches.shape[0], 2, blk_sz, blk_sz), dtype=torch.float32)
    transform = transforms.Resize((blk_sz, blk_sz), trf.InterpolationMode.BICUBIC)

    for p_idx, patch in enumerate(patches):
        patch = transforms.ToPILImage()(patch)
        patch = transform(patch)
        patch = np.array(patch)
        patch_lab = rgb2lab(patch).astype("float32")
        patch_lab = transforms.ToTensor()(patch_lab)
        L_patches[p_idx, ...] = patch_lab[[0], ...] / 50. - 1.
        ab_patches[p_idx, ...] = patch_lab[[1, 2], ...] / 110.

    # L_patches = torch.moveaxis(L_patches, 3, 1)
    # ab_patches = torch.moveaxis(ab_patches, 3, 1)   # shape = (*, 3, blk_sz, blk_sz)
    data = {}
    for b_idx in range(0, patches.shape[0], 16):
        data.update({'L': L_patches[b_idx:b_idx+16, ...]})
        data.update({'ab': ab_patches[b_idx:b_idx+16, ...]})
        model.net_G.eval()
        with torch.no_grad():
            model.setup_input(data)
            model.forward()
        fake_color = model.fake_color.detach()
        # real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)
        corrected_patches[b_idx:b_idx+16, ...] = (fake_imgs * 255).astype('uint8')

    padded_corrected_img = unpatchify(corrected_patches.reshape(padded_shape), padded_img.shape)
    corrected_img = padded_corrected_img[:rows, :cols, :]
    # plt.imshow(corrected_img)
    # plt.show()
    return corrected_img


def train_for_cannabis(database, isAugmented=True):
    image_paths = glob.glob(database + '/*.jpg')
    image_num = len(image_paths)
    train_num = int(image_num * 0.8)  # all as training set
    rand_idxs = np.random.permutation(image_num)
    train_idxs = rand_idxs[:train_num]
    val_idxs = rand_idxs[train_num:]
    train_paths = [image_paths[i] for i in train_idxs]
    val_paths = [image_paths[i] for i in val_idxs]
    print('Training: {} images, Validation: {} images'.format(len(train_paths), len(val_paths)))

    # dataset
    train_dl = make_dataloaders(paths=train_paths, split='train', isAugmented=isAugmented)
    val_dl = make_dataloaders(paths=val_paths, split='val')
    suffix = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    pretrain_generator(net_G, train_dl, opt, criterion, 20)
    torch.save(net_G.state_dict(), "res18-unet" + suffix + ".pt")
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("res18-unet" + suffix + ".pt", map_location=device))
    net_G.to(device)
    model = MainModel(net_G=net_G)

    # training
    epoch_num = 50
    train_model(model, train_dl, val_dl, epoch_num)
    torch.save(model.state_dict(), 'res18_colorizer' + suffix + '_epoches_' + str(epoch_num) + ".pt")


def train_for_subset(data_dir, subset, isBalanced=True, isAugmented=True):
    if isBalanced:
        database = os.path.join(data_dir, subset + '_blocks_balanced')
    else:
        database = os.path.join(data_dir, subset + '_blocks')

    image_paths = glob.glob(database + '/*.png')
    image_num = len(image_paths)
    train_num = int(image_num * 0.8)  # all as training set
    rand_idxs = np.random.permutation(image_num)
    train_idxs = rand_idxs[:train_num]
    val_idxs = rand_idxs[train_num:]
    train_paths = [image_paths[i] for i in train_idxs]
    val_paths = [image_paths[i] for i in val_idxs]
    print('Training: {} images, Validation: {} images'.format(len(train_paths), len(val_paths)))

    # dataset
    train_dl = make_dataloaders(paths=train_paths, split='train', isAugmented=isAugmented)
    val_dl = make_dataloaders(paths=val_paths, split='val')
    suffix = ""
    if isBalanced:
        suffix = suffix + "_balanced"
    if isAugmented:
        suffix = suffix + "_augmented"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    pretrain_generator(net_G, train_dl, opt, criterion, 20)
    torch.save(net_G.state_dict(), "res18_" + subset + "-unet" + suffix + ".pt")
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("res18_" + subset + "-unet" + suffix + ".pt", map_location=device))
    net_G.to(device)
    model = MainModel(net_G=net_G)

    # training
    epoch_num = 50
    train_model(model, train_dl, val_dl, epoch_num)
    torch.save(model.state_dict(), 'res18_colorizer' + suffix + '_' + subset + '_epoches_' + str(epoch_num) + ".pt")


def load_model_for_subset(subset, isBalanced=True, isAugemented=True):
    if subset not in ['A1', 'A2', 'A3', 'A4', 'A5', 'All']:
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
        model = MainModel(net_G=net_G)
        model.load_state_dict(
            torch.load("colorizer_epoches_50.pt", map_location=device))
        return model
    else:
        suffix = ""
        if isBalanced:
            suffix = suffix + "_balanced"
        if isAugemented:
            suffix = suffix + "_augmented"
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        net_G.load_state_dict(torch.load("res18_" + subset + "-unet" + suffix + ".pt", map_location=device))
        model = MainModel(net_G=net_G)
        model.load_state_dict(torch.load("res18_colorizer" + suffix + "_" + subset + "_epoches_50.pt", map_location=device))
        return model


def load_cc_model(GD_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    # G_path = GD_path[:-4] + '.pt'
    # net_G.load_state_dict(torch.load(G_path, map_location=device))
    model = MainModel(net_G=net_G)
    model.load_state_dict(torch.load(GD_path, map_location=device))
    return model


def test_for_subset(data_dir, result_dir, model_type='plain'):
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    if 'A5' in subfolders:
        subfolders.remove('A5')
    allfolder_sum = []
    for subfolder in subfolders:
        image_paths = glob.glob(os.path.join(data_dir, subfolder, '*_rgb.png'))
        pathlib.Path(os.path.join(result_dir, subfolder)).mkdir(parents=True, exist_ok=True)
        model = load_model_for_subset('All', model_type)
        with open(os.path.join(result_dir, subfolder, 'seg_result.csv'), 'w+') as seg_file:
            seg_file.write("Image name, FGBGDice\n")
            subfolder_sum = []
            for image_path in image_paths:
                # print('Processing {}'.format(image_path))
                image_name = os.path.basename(image_path)
                img = test_one_image(model, image_path)
                # img = cv2.imread(image_path)
                original_shape = img.shape
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                save_img_path = os.path.join(result_dir, subfolder, image_name[:-8] + "_corrected.png")
                cv2.imwrite(save_img_path, img)

                img = imutils.resize(img, width=512)
                _, mask, _, _ = cal_leaf_area(img, (30, 90, 50), (90, 255, 255))
                segmented = cv2.resize(mask, dsize=(original_shape[1], original_shape[0]))
                inLabel = cv2.threshold(segmented, 127, 1, cv2.THRESH_BINARY)[1]
                gtLabel = cv2.imread(os.path.join(data_dir, subfolder, image_name[:-8] + "_fg.png"), 0)
                gtLabel = cv2.threshold(gtLabel, 127, 1, cv2.THRESH_BINARY)[1]
                if inLabel.shape != gtLabel.shape:
                    print('Warning: inconsistent shapes!')
                fgbgdice = FGBGDice(inLabel, gtLabel)
                cv2.imwrite(os.path.join(result_dir, subfolder, image_name[:-8] + "_mask.png"), segmented)
                subfolder_sum.append(fgbgdice)
                allfolder_sum.append(fgbgdice)
                seg_file.write(image_name + ", " + '{0:.3f}'.format(fgbgdice) + "\n")
                print("FGBGDice of " + image_name + ": " + '{0:.3f}'.format(fgbgdice))
            seg_file.write("Avg. ," + '{0:.3f}'.format(statistics.mean(subfolder_sum)) + "\n")
            seg_file.write("Std. ," + '{0:.3f}'.format(statistics.stdev(subfolder_sum)) + "\n")

        with open(os.path.join(result_dir, 'seg_avg_result.csv'), 'w') as seg_file:
            seg_file.write("Avg. FGBG Dice, " + '{0:.3f}'.format(statistics.mean(allfolder_sum)) + "\n")
            seg_file.write("Std. FGBG Dice, " + '{0:.3f}'.format(statistics.stdev(allfolder_sum)) + "\n")


def test_colorization_for_subset(data_dir, ref_dir, result_dir, isCombined=True, isBalanced=True, isAugmented=True):
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    if 'A5' in subfolders:
        subfolders.remove('A5')
    allfolder_sum_PSNR = []
    allfolder_sum_SSIM = []
    allfolder_sum_LPIPS = []
    for subfolder in subfolders:
        image_paths = glob.glob(os.path.join(data_dir, subfolder, '*_rgb.png'))
        pathlib.Path(os.path.join(result_dir, subfolder)).mkdir(parents=True, exist_ok=True)
        if isCombined:
            model = load_model_for_subset("All", isBalanced, isAugmented)
        else:
            model = load_model_for_subset(subfolder, isBalanced, isAugmented)
        suffix = ''
        if isCombined:
            suffix = suffix + '_combined'
        else:
            suffix = suffix + '_individual'

        if isBalanced:
            suffix = suffix + '_balanced'
        if isAugmented:
            suffix = suffix + '_augmented'
        with open(os.path.join(result_dir, subfolder, 'colorization_result' + suffix + '.csv'), 'w+') as seg_file:
            seg_file.write("Image name, PSNR, SSIM, LPIPS\n")
            subfolder_sum_PSNR = []
            subfolder_sum_SSIM = []
            subfolder_sum_LPIPS = []
            for image_path in image_paths:
                print('Processing {}'.format(image_path))
                image_name = os.path.basename(image_path)
                img = test_one_image(model, image_path)

                save_img_path = os.path.join(result_dir, subfolder, image_name[:-8] + suffix + ".png")
                cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                ref_img = cv2.imread(os.path.join(ref_dir, subfolder, image_name))
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

                PSNR, SSIM, LPIPS = image_quality_metrics(ref_img, img)

                subfolder_sum_PSNR.append(PSNR)
                subfolder_sum_SSIM.append(SSIM)
                subfolder_sum_LPIPS.append(LPIPS)

                allfolder_sum_PSNR.append(PSNR)
                allfolder_sum_SSIM.append(SSIM)
                allfolder_sum_LPIPS.append(LPIPS)

                seg_file.write(image_name + ', {0:.3f}'.format(PSNR) +
                                            ', {0:.3f}'.format(SSIM) +
                                            ', {0:.3f}'.format(LPIPS) + "\n")
                print(image_name + ": " +
                      '{0:.3f}'.format(PSNR) +
                      ', {0:.3f}'.format(SSIM) +
                      ', {0:.3f}'.format(LPIPS))

            seg_file.write("Avg. " +
                           ', {0:.3f}'.format(statistics.mean(subfolder_sum_PSNR)) +
                           ', {0:.3f}'.format(statistics.mean(subfolder_sum_SSIM)) +
                           ', {0:.3f}'.format(statistics.mean(subfolder_sum_LPIPS)) + "\n")
            seg_file.write("Std. " +
                           ', {0:.3f}'.format(statistics.stdev(subfolder_sum_PSNR)) +
                           ', {0:.3f}'.format(statistics.stdev(subfolder_sum_SSIM)) +
                           ', {0:.3f}'.format(statistics.stdev(subfolder_sum_LPIPS)) + "\n")

        with open(os.path.join(result_dir, 'colorization_avg_result' + suffix + '.csv'), 'w') as seg_file:
            seg_file.write("Avg. " +
                           ', {0:.3f}'.format(statistics.mean(allfolder_sum_PSNR)) +
                           ', {0:.3f}'.format(statistics.mean(allfolder_sum_SSIM)) +
                           ', {0:.3f}'.format(statistics.mean(allfolder_sum_LPIPS)) + "\n")
            seg_file.write("Std. " +
                           ', {0:.3f}'.format(statistics.stdev(allfolder_sum_PSNR)) +
                           ', {0:.3f}'.format(statistics.stdev(allfolder_sum_SSIM)) +
                           ', {0:.3f}'.format(statistics.stdev(allfolder_sum_LPIPS)) + "\n")


def color_correction_for_subset(data_dir, result_dir, isCombined=True, isBalanced=True, isAugmented=True):
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    if 'A5' in subfolders:
        subfolders.remove('A5')

    for subfolder in subfolders:
        image_paths = glob.glob(os.path.join(data_dir, subfolder, '*_rgb.png'))
        pathlib.Path(os.path.join(result_dir, subfolder)).mkdir(parents=True, exist_ok=True)
        if isCombined:
            model = load_model_for_subset("All", isBalanced, isAugmented)
        else:
            model = load_model_for_subset(subfolder, isBalanced, isAugmented)

        for image_path in tqdm(image_paths):
            image_name = os.path.basename(image_path)
            img = test_one_image(model, image_path)

            save_img_path = os.path.join(result_dir, subfolder, image_name)
            cv2.imwrite(save_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            shutil.copyfile(os.path.join(data_dir, subfolder, image_name[:-8]+"_fg.png"),
                            os.path.join(result_dir, subfolder, image_name[:-8]+"_fg.png"))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = '/home/lxfhfut/Dropbox/Internet_of_Cannabis/Cannabis_old'
    result_dir = '/home/lxfhfut/Dropbox/Internet_of_Cannabis/Cannabis_768_corrected'
    color_correction_for_subset(data_dir, result_dir, isCombined=False, isBalanced=False, isAugmented=True)
