import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'
import cv2
import imutils
import convcrf
import argparse
import numpy as np
import torch.nn.init
from tqdm import tqdm
from skimage import measure
import torch.optim as optim
from torch.autograd import Variable

from models import BackBone, LightConv3x3
from color_correction import load_cc_model, test_one_image
from utils import mean_image, cal_greenness, save_result_img, save_result_video

# For reproductivity
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Self-Supervised Leaf Segmentation')
    parser.add_argument('--num_channels', default=64, type=int,
                        help='number of channels')
    parser.add_argument('--max_iter', default=300, type=int,
                        help='number of maximum iterations')
    parser.add_argument('--min_labels', default=2, type=int,
                        help='minimum number of labels')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--sz_filter', default=5, type=int,
                        help='CRF filter size')
    parser.add_argument('--at', default=0.2, type=float,
                        help='Absolute greenness threshold')
    parser.add_argument('--rt', default=0.5, type=float,
                        help='Relative greenness threshold')
    parser.add_argument('--ccm', type=str, default='', help='path of color correction model')
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='save intermediate results as video')
    parser.add_argument('--save_frame_interval', default=2, type=int,
                        help='save frame every save_frame_interval iterations')
    parser.add_argument('--save_path', type=str, default="./output/")
    parser.add_argument('--input', type=str, help='input image path', required=True)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if os.path.exists(args.ccm):
        print('Applying color correction with model {}...'.format(args.ccm))
        cc_model = load_cc_model(args.ccm)
        img = test_one_image(cc_model, args.input)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_path, "color_corrected.jpg"), img)
        print('Color-corrected image has been saved to {}'.format(os.path.join(args.save_path, "color_corrected.jpg")))
    else:
        img = cv2.imread(args.input)

    img = imutils.resize(img, width=512)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[:2]
    img = img.transpose(2, 0, 1)
    data = torch.from_numpy(np.array([img.astype('float32') / 255.]))
    img_var = torch.Tensor(img.reshape([1, 3, *img_size]))  # 1, 3, h, w

    config = convcrf.default_conf
    config['filter_size'] = args.sz_filter
    gausscrf = convcrf.GaussCRF(conf=config, shape=img_size, nclasses=args.num_channels, use_gpu=True)

    model = BackBone([LightConv3x3], [2], [args.num_channels//2, args.num_channels])

    if torch.cuda.is_available():
        data = data.cuda()
        img_var = img_var.cuda()
        gausscrf = gausscrf.cuda()
        model = model.cuda()

    data = Variable(data)
    img_var = Variable(img_var)

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    label_colours = np.random.randint(255, size=(100, 3))
    all_image_labels = []
    all_mean_images = []
    all_absolute_greenness = []
    all_relative_greenness = []
    all_thresholded = []
    pbar = tqdm(range(args.max_iter))
    for batch_idx in pbar:
        optimizer.zero_grad()
        output = model(data)[0]
        unary = output.unsqueeze(0)
        prediction = gausscrf.forward(unary=unary, img=img_var)
        target = torch.argmax(prediction.squeeze(0), axis=0).reshape(img_size[0] * img_size[1], )
        output = output.permute(1, 2, 0).contiguous().view(-1, args.num_channels)

        im_target = target.data.cpu().numpy()
        image_labels = im_target.reshape(img_size[0], img_size[1]).astype("uint8")
        num_labels = len(np.unique(im_target))
        if args.save_video and not(batch_idx % args.save_frame_interval):
            im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(img_size[0], img_size[1], 3).astype("uint8")
            mean_img = mean_image(rgb_image, measure.label(image_labels))
            absolute_greenness, relative_greenness = cal_greenness(mean_img)
            greenness = np.multiply(relative_greenness, (absolute_greenness > args.at).astype(np.float64))
            thresholded = 255 * ((greenness > args.rt).astype("uint8"))
            all_mean_images.append(mean_img)
            all_absolute_greenness.append(absolute_greenness)
            all_relative_greenness.append(relative_greenness)
            all_thresholded.append(thresholded)
            all_image_labels.append(im_target_rgb)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        pbar.set_description("Iterations {0}/{1}: {2}, {3:.2f}".format(batch_idx, args.max_iter, num_labels, loss.item()))

        if num_labels <= args.min_labels:
            print("nLabels", num_labels, "reached minLabels", args.min_labels, ".")
            break

    if args.save_video:
        save_result_path = os.path.join(args.save_path, "result.mp4")
        save_result_video(save_result_path, rgb_image, all_image_labels, all_mean_images,
                          all_absolute_greenness, all_relative_greenness, all_thresholded)
    else:
        labels = measure.label(image_labels)
        mean_img = mean_image(rgb_image, labels)
        absolute_greenness, relative_greenness = cal_greenness(mean_img)
        greenness = np.multiply(relative_greenness, (absolute_greenness > args.at).astype(np.float64))
        thresholded = 255 * ((greenness > args.rt).astype("uint8"))

        save_result_path = os.path.join(args.save_path, "result.jpg")
        save_result_img(save_result_path, rgb_image, labels, mean_img,
                        absolute_greenness, relative_greenness, thresholded)

    print('Result has been saved in {}'.format(save_result_path))
