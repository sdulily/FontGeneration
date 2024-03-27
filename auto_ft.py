import json
from collections import OrderedDict

from torch.utils.data import DataLoader
from auto_dataset_1SR import Dataset

import torch
import time
import os
from torchvision.utils import save_image
import torch.nn as nn
from natsort import natsorted
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from models.generator import Generator as Generator
from models.guidingNet import GuidingNet
import torch.backends.cudnn as cudnn
from tools.ops import attn_to_rgb


def build_model(args):
    args.to_train = 'CG'

    networks = {}
    opts = {}
    if 'C' in args.to_train:
        networks['C_EMA'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k})
    if 'G' in args.to_train:
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False)


    #torch.cuda.set_device(args.gpu)
    for name, net in networks.items():
        networks[name] = net.cuda()

    if 'C' in args.to_train:
        opts['C'] = torch.optim.Adam(
            networks['C_EMA'].module.parameters() if args.distributed else networks['C_EMA'].parameters(),
            1e-4,  weight_decay=0.0001)
      
    if 'G' in args.to_train:
        opts['G'] = torch.optim.RMSprop(
            networks['G_EMA'].module.parameters() if args.distributed else networks['G_EMA'].parameters(),
            1e-4, weight_decay=0.0001)

    return networks, opts

def load_model(args, networks, opts):
    if args.load_model is not None:
        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            if not False:
                for name, net in networks.items():
                    tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                    if 'module' in tmp_keys:
                        tmp_new_dict = OrderedDict()
                        for key, val in checkpoint[name + '_state_dict'].items():
                            tmp_new_dict[key[7:]] = val
                        net.load_state_dict(tmp_new_dict)
                        networks[name] = net
                    else:
                        net.load_state_dict(checkpoint[name + '_state_dict'])
                        networks[name] = net

            for name, opt in opts.items():
                opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
                opts[name] = opt
            print("=> loaded checkpoint '{}' (epoch {})".format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))

def main(data_path):
    font_name = data_path.split('/')[-1]
    path = './checksave/test_result/' + font_name

    if not os.path.exists(path):
        os.mkdir(path)

    #if not os.path.exists(path + '/test_result'):
        #os.mkdir(path + '/test_result')
    #if not os.path.exists(path + '/train_result'):
        #os.mkdir(path + '/train_result')

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    device = torch.device(device)

    ##################################################
    parser = argparse.ArgumentParser(description='PyTorch GAN Training')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Dataset directory. Please refer Dataset in README.md')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers of data loader')

    parser.add_argument('--model_name', type=str, default='./logs/GAN_20221108-151529',
                        help='Prefix of logs and results folders. '
                             'ex) --model_name=ABC generates ABC_20191230-131145 in logs and results')

    parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs to run. Not actual epoch.')
    parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--val_num', default=190, type=int, help='Number of test images for each style')
    parser.add_argument('--val_batch', default=10, type=int,
                        help='Batch size for validation. '
                             'The result images are stored in the form of (val_batch, val_batch) grid.')
    parser.add_argument('--log_step', default=100, type=int)

    parser.add_argument('--sty_dim', default=512, type=int, help='The size of style vector')
    parser.add_argument('--output_k', default=400, type=int, help='Total number of classes to use')
    parser.add_argument('--img_size', default=128, type=int, help='Input image size')
    parser.add_argument('--dims', default=2048, type=int, help='Inception dims for FID')

    parser.add_argument('--load_model', default='./logs/GAN_20221108-151529', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)'
                             'ex) --load_model GAN_20190101_101010'
                             'It loads the latest .ckpt file specified in checkpoint.txt in GAN_20190101_101010')
    parser.add_argument('--validation', dest='validation', action='store_true',
                        help='Call for valiation only mode')

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--ddp', dest='ddp', action='store_true', help='Call if using DDP')
    parser.add_argument('--port', default='8993', type=str)

    parser.add_argument('--iid_mode', default='iid+', type=str, choices=['iid', 'iid+'])

    parser.add_argument('--w_gp', default=10.0, type=float, help='Coefficient of GP of D')
    parser.add_argument('--w_rec', default=0.1, type=float, help='Coefficient of Rec. loss of G')
    parser.add_argument('--w_adv', default=1.0, type=float, help='Coefficient of Adv. loss of G')
    parser.add_argument('--w_vec', default=0.01, type=float, help='Coefficient of Style vector rec. loss of G')
    parser.add_argument('--w_off', default=0.5, type=float, help='Coefficient of offset normalization. loss of G')

    ##################################################
    args = parser.parse_args()
    args.log_dir = './logs/GAN_20221108-151529'
    args.distributed = False
    networks, opts = build_model(args)
    load_model(args, networks, opts)
    cudnn.benchmark = True

    G = networks['G_EMA']
    C = networks['C_EMA']

    g_opt = opts['G']
    c_opt = opts['C']

    with open('./file_6763.json', 'r+') as file:
        content = file.read()
    component_dict = json.loads(content)

    train_dataset = Dataset(data_path, component_dict=component_dict, mode='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)
    test_dataset = Dataset(data_path, component_dict=component_dict, mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    for epoch in range(590):

        # G.train()
        # C.train()
        # for i, batch in enumerate(train_loader):
        #     a, b, style, _ = batch
        #     a = a.to(device)
        #     b = b.to(device)
        #     style = style.to(device)
        #
        #     c_src, skip1, skip2 = G.cnt_encoder(a)
        #     s_ref = C(style, sty=True)
        #     output, _ = G.decode(c_src, s_ref, skip1, skip2)
        #
        #     l1_loss = nn.L1Loss()(output, b)
        #
        #     g_opt.zero_grad()
        #     c_opt.zero_grad()
        #     l1_loss.backward()
        #     c_opt.step()
        #     g_opt.step()
        #
        #
        #     img = torch.cat((a, b, output), dim=0)
        #     #if epoch % 10 == 0 and i % 1 == 0:
        #         #save_image(img, path + '/train_result/epoch{}_{}.png'.format(epoch, i), padding=10, nrow=8, pad_value=1, normalize=True)
        #         #with open(path + '/train_process.txt', 'a') as f:
        #             #f.writelines('epoch  {}  batch  {}  l1_loss  {:.6f}\n'.format(epoch, i, l1_loss.float()))
        #     if i % 10 == 0:
        #         print('epoch  {}  batch  {}  l1_loss  {:.6f}'.format(epoch, i, l1_loss.float()))

        if epoch == 1:
            with torch.no_grad():
                G.eval()
                C.eval()
                for i, batch in enumerate(test_loader):
                    a, b, style, component_list, name = batch
                    a = a.to(device)
                    style = style.to(device)
                    for i in range(len(component_list)):
                        component_list[i] = component_list[i].to(device)


                    c_src = G.cnt_encoder(a, component_list)
                    s_ref = C(style, sty=True)
                    output = G.decode(c_src, s_ref)

                    # c_src_v = torch.sum(c_src, 1, keepdim=True)
                    # c_src_v = attn_to_rgb(c_src_v)[:, :3, :, :]
                    # c_src_v = torch.nn.functional.interpolate(c_src_v, size=(128, 128), mode='bilinear').to(device).to(torch.float32)
                    #
                    # result = torch.cat((output, c_src_v), dim=0)


                    save_image(output, path + '/' + name[0], padding=10, nrow=1, pad_value=1, normalize=True)


if __name__ == '__main__':

    newfont = r'./test_set_8/newfont'
    font_list = natsorted([os.path.join(newfont, i) for i in os.listdir(newfont)])
    for index, font in enumerate(font_list):
        main(data_path=font)