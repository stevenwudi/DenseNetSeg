"""
This is a demo script to generate the video sequence semantic segmentation
"""
import os
import json
from os.path import join
import cv2
import numpy as np
import importlib.machinery
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable

from utils.evaluation_utils import CITYSCAPE_PALETTE
from models.DenseNetSeg import DenseSeg
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def load_model():
    # image loading and preprocessing
    loader = importlib.machinery.SourceFileLoader('config', './config/config_densenet_seg.py')
    args = loader.load_module()
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=info['mean'], std=info['std'])
    ])
    # model definition
    single_model = DenseSeg(args.arch, args.num_class, args.transition_layer,
                            args.conv_num_features, args.out_channels_num,
                            pretrained=True)
    model = torch.nn.DataParallel(single_model).cuda()

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model, data_transform


def test_single(filepath, image_name, save_file_path, model, data_transform):
    image = Image.open(os.path.join(filepath, image_name))

    image_origin = np.asarray(image)
    image = data_transform(image)

    image = image.unsqueeze(0)
    image_var = Variable(image, requires_grad=False, volatile=True)
    final = model(image_var)[0]
    _, pred = torch.max(final, 0)
    pred = pred.cpu().data.numpy()
    pred_color = CITYSCAPE_PALETTE[pred.squeeze()]

    overlap_img = cv2.addWeighted(src1=image_origin, alpha=0.8, src2=pred_color, beta=1, gamma=0)
    overlap_img = cv2.cvtColor(overlap_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_file_path, image_name), overlap_img)

    return pred[0], pred_color


def main():

    sequence_name = 'stuttgart_02'
    filepath = "/media/samsumg_1tb/CITYSCAPE/leftImg8bit/demoVideo/"+sequence_name
    save_file_path = filepath+'_densenet_seg/'
    model, data_transform = load_model()

    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)
    imgs = [img for img in os.listdir(filepath) if img.endswith('.png')]
    for image_name in imgs:
        test_single(filepath, image_name, save_file_path, model, data_transform)
    print('Done!')


if __name__ == '__main__':
    main()
