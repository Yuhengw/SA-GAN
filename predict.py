import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager import TestDataset,ValDataset
from utils import gpu_manage, save_image, heatmap
from models.gen.SPANet import Generator
import os


def predict(config, args):
    gpu_manage(args)
   # torch.cuda.set_device('cuda:0')
    dataset = TestDataset(args.test_dir, config.in_ch, config.out_ch)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)
    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch[0])
            filename = batch[1][0]
            if args.cuda:
                x = x.cuda()
            
            att, out = gen(x)

            h = 1
            w = 1
            c = 3
            p = config.width

            allim = np.zeros((h, w, c, p, p))
            x_ = x.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            att_ = att.cpu().numpy()[0] * 255
            heat_att = heatmap(att_.astype('uint8'))
            

            allim[0, 0, :] = out_rgb * 255

            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*p, w*p, c))

        

            save_image(args.out_dir, allim, i, 1, filename=filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=False)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    config = AttrMap(config)

    predict(config, args)