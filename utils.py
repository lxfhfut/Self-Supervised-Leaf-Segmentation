import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import multivariate_normal


def mean_image(image, labels):
    im_rp = image.reshape(-1, image.shape[2])
    labels_1d = np.reshape(labels, -1)
    uni = np.unique(labels_1d)
    uu = np.zeros(im_rp.shape)
    for i in uni:
        loc = np.where(labels_1d == i)[0]
        mm = np.mean(im_rp[loc, :], axis=0)
        uu[loc, :] = mm
    return np.reshape(uu, [image.shape[0], image.shape[1], image.shape[2]]).astype('uint8')


def cal_greenness(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 0] = hsv[:, :, 0] / 180.0
    hsv[:, :, 1] = hsv[:, :, 1] / 255.0
    hsv[:, :, 2] = hsv[:, :, 2] / 255.0

    mu = np.array([60.0 / 180.0, 160.0 / 255.0, 200.0 / 255.0])
    sigma = np.array([.1, .3, .5])
    covariance = np.diag(sigma ** 2)

    rv = multivariate_normal(mean=mu, cov=covariance)
    z = rv.pdf(hsv)
    ref = rv.pdf(mu)
    absolute_greenness = z/ref
    relative_greenness = (z - np.min(z)) / (np.max(z) - np.min(z) + np.finfo(float).eps)

    return absolute_greenness, relative_greenness


def crop_img_from_center(img, crop_size=(512, 512)):
    assert(img.shape[0] >= crop_size[0])
    assert(img.shape[1] >= crop_size[1])
    assert(len(img.shape)==2 or len(img.shape)==3)
    cw = img.shape[1] // 2
    ch = img.shape[0] // 2
    x = cw - crop_size[1] // 2
    y = ch - crop_size[0] // 2
    if len(img.shape) == 3:
        return img[y:y + crop_size[0], x:x + crop_size[1], :]
    else:
        return img[y:y + crop_size[0], x:x + crop_size[1]]


def crop_img_from_center(img, width=512):
    assert(img.shape[1] >= width)
    assert (len(img.shape) == 2 or len(img.shape) == 3)
    height = img.shape[0] * width // img.shape[1]
    cw = img.shape[1] // 2
    ch = img.shape[0] // 2
    x = cw - width // 2
    y = ch - height // 2
    if len(img.shape) == 3:
        return img[y:y + height, x:x + width, :]
    else:
        return img[y:y + height, x:x + width]


def save_result_img(save_path, rgb_img, img_labels, mean_img, absolute_greenness, relative_greenness, thresholded):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Original image')
    plt.axis('off')
    plt.imshow(rgb_img)

    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Semantic segmentation')
    plt.axis('off')
    plt.imshow(img_labels)

    ax = fig.add_subplot(2, 3, 3)
    ax.set_title('Mean image')
    plt.axis('off')
    plt.imshow(mean_img)

    ax = fig.add_subplot(2, 3, 4)
    ax.set_title('Binary mask')
    plt.axis('off')
    plt.imshow(thresholded, cmap='gray')

    ax = fig.add_subplot(2, 3, 5)
    ax.set_title('Relative greenness')
    plt.axis('off')
    plt.imshow(relative_greenness, cmap='gray', vmin=0, vmax=1)

    ax = fig.add_subplot(2, 3, 6)
    ax.set_title('Absolute greenness')
    plt.axis('off')
    plt.imshow(absolute_greenness, cmap='gray', vmin=0, vmax=1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show(block=False)
    plt.close("all")


def save_result_video(save_path, rgb_img, all_img_labels, all_mean_imgs, all_absolute_greenness, all_relative_greenness, all_masks):
    imgs = []
    fig = plt.figure(figsize=(15, 10))

    for i in range(len(all_img_labels)):
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Original image')
        ax1.axis('off')
        ax1.imshow(cv2.resize(rgb_img, (512, 512)))

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Semantic segmentation')
        ax2.axis('off')
        ax2.imshow(cv2.resize(all_img_labels[i], (512, 512)))

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Mean image')
        ax3.axis('off')
        ax3.imshow(cv2.resize(all_mean_imgs[i], (512, 512)))

        # plt.tight_layout()
        # imgs.append([ax1, ax2, ax3])

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('Binary mask')
        ax4.axis('off')
        ax4.imshow(cv2.resize(all_masks[i], (512, 512)), cmap='gray')

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title('Relative greenness')
        ax5.axis('off')
        ax5.imshow(cv2.resize(all_relative_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title('Relative greenness')
        ax6.axis('off')
        ax6.imshow(cv2.resize(all_absolute_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)

        plt.tight_layout()
        imgs.append([ax1, ax2, ax3, ax4, ax5, ax6])

    ani = animation.ArtistAnimation(fig, imgs, interval=80, blit=False)
    ani.save(save_path)


def save_result_video_old(save_path, rgb_img, gt_mask, all_img_labels, all_mean_imgs, all_greenness, all_masks):
    imgs = []
    fig = plt.figure(figsize=(10, 15))

    for i in range(len(all_img_labels)):
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title('Original image')
        ax1.axis('off')
        ax1.imshow(cv2.resize(rgb_img, (512, 512)))

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title('Semantic segmentation')
        ax2.axis('off')
        ax2.imshow(cv2.resize(all_img_labels[i], (512, 512)))

        ax5 = fig.add_subplot(2, 3, 3)
        ax5.set_title('Mean image')
        ax5.axis('off')
        ax5.imshow(cv2.resize(all_mean_imgs[i], (512, 512)))

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title('Ground truth')
        ax4.axis('off')
        ax4.imshow(cv2.resize(gt_mask, (512, 512)), cmap='gray')

        ax3 = fig.add_subplot(2, 3, 5)
        ax3.set_title('Binary mask')
        ax3.axis('off')
        ax3.imshow(cv2.resize(all_masks[i], (512, 512)), cmap='gray')

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title('Greenness')
        ax6.axis('off')
        ax6.imshow(cv2.resize(all_greenness[i], (512, 512)), cmap='gray', vmin=0, vmax=1)

        plt.tight_layout()
        imgs.append([ax1, ax2, ax3, ax4, ax5, ax6])

    ani = animation.ArtistAnimation(fig, imgs, interval=80, blit=False)
    ani.save(save_path)


def color_coded_map(gt, det):
    gt = gt.astype(bool)
    det = det.astype(bool)
    green_area = np.logical_and(det, gt)
    red_area = np.logical_and(det, np.logical_not(gt))
    blue_area = np.logical_and(np.logical_not(det), gt)

    color_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[green_area] = 255
    color_map[:, :, 1] = tmp_map

    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[red_area] = 255
    color_map[:, :, 2] = tmp_map

    tmp_map = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)
    tmp_map[blue_area] = 255
    color_map[:, :, 0] = tmp_map
    return color_map
