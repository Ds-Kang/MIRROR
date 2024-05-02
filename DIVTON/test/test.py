import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint, ParsingNetwork
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from tqdm import tqdm

def image_resize_in(image, width = 192, height = 256, black=False, interpolation="bilinear"):
    old_size = image.shape[2:]

    if old_size[0] / old_size[1] > height / width:
        ratio = height / float(old_size[0])

    else:
        ratio = width / float(old_size[1])

    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    image = F.interpolate(image, (new_size[0], new_size[1]), mode=interpolation)

    delta_w = width - new_size[1]
    delta_h = height - new_size[0]
    top, bottom = 0, delta_h
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    if black == False:
        color = 1
    else:
        color = 0
    image = F.pad(image, [left, right, top, bottom], value=color)
    return image, (top, bottom, left, right), ratio

opt = TestOptions().parse()

start_epoch, epoch_iter = 1, 0

parsing_input_size = 7

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)
warp_model = AFWM(opt, parsing_input_size + 3, opt.skip)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 5, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)


parsing_model = ParsingNetwork(opt)
parsing_model.eval()
parsing_model.cuda()

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

for epoch in range(1, 2):

    for i, data in tqdm(enumerate(dataset, start=epoch_iter)):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        real_image = data['image']
        clothes = data['clothes']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int32))
        clothes = clothes * edge    

        size = real_image.size()

        parsing_frame = parsing_model(real_image.cuda())

        upper_head = (
            torch.where(parsing_frame == 1, 1, 0)
            + torch.where(parsing_frame == 2, 1, 0)
            + torch.where(parsing_frame == 3, 1, 0)
            + torch.where(parsing_frame == 4, 1, 0)
        )

        all_y, all_x = torch.where(upper_head.squeeze() == 1)
        if len(all_x) and len(all_y):
            all_x_min, all_x_max = max(0, torch.min(all_x) - 20), min(
                192, torch.max(all_x) + 20
            )
            all_y_min, all_y_max = max(0, torch.min(all_y) - 20), min(
                256, torch.max(all_y) + 30
            )
        else:
            all_x_min, all_x_max = 0, 192
            all_y_min, all_y_max = 0, 256

        upper = (
            torch.where(parsing_frame == 2, 1, 0)
            + torch.where(parsing_frame == 3, 1, 0)
            + torch.where(parsing_frame == 4, 1, 0)
        )

        # make bounding box
        y, x = torch.where(upper.squeeze() == 1)
        if len(x) and len(y):
            x_min, x_max = max(0, torch.min(x) - 10), min(192, torch.max(x) + 10)
            y_min, y_max = max(0, torch.min(y) - 20), min(256, torch.max(y) + 20)
        else:
            x_min, x_max = 0, 192
            y_min, y_max = 0, 256

        all_cropped_img1 = real_image[..., all_y_min:all_y_max, all_x_min:all_x_max]
        resized, paddings, ratio = image_resize_in(all_cropped_img1)

        all_cropped_parse = parsing_frame[..., all_y_min:all_y_max, all_x_min:all_x_max]
        resized_parse, _, _ = image_resize_in(
            all_cropped_parse.float(), black=True, interpolation="nearest"
        )

        resized = torch.where(resized_parse != 0, resized.cuda(), 1)

        corr_y_min = int((y_min - all_y_min) * ratio)
        corr_y_max = int((y_max - all_y_min) * ratio)
        corr_x_min = int((x_min - all_x_min) * ratio)
        corr_x_max = int((x_max - all_x_min) * ratio)

        parsing_shape = (1, parsing_input_size, size[2], size[3])
        parsing_warping_input = torch.cuda.FloatTensor(
            torch.Size(parsing_shape)
        ).zero_()
        parsing_warping_input = parsing_warping_input.scatter_(
            1, resized_parse.long(), 1.0
        )
        pf_warp_concat = torch.cat((resized.cuda(), parsing_warping_input), 1)
        flow_out = warp_model(pf_warp_concat.cuda(), clothes.cuda())
        
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([resized.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite, gen_mask = torch.split(gen_outputs, [3, 1, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        gen_mask = torch.sigmoid(gen_mask)
        
        gen_mask=gen_mask[:,:,paddings[0]:256-paddings[1],paddings[2]:192-paddings[3]]
        gen_mask=gen_mask[:,:,corr_y_min:corr_y_max, corr_x_min:corr_x_max]
        p_tryon=p_tryon[:,:,paddings[0]:256-paddings[1],paddings[2]:192-paddings[3]]
        p_tryon=p_tryon[:,:,corr_y_min:corr_y_max, corr_x_min:corr_x_max]

        
        p_tryon=F.interpolate(p_tryon, size=(y_max-y_min, x_max-x_min), mode='bilinear', align_corners=True).detach()
        gen_mask=F.interpolate(gen_mask, size=(y_max-y_min, x_max-x_min), mode='bilinear', align_corners=True).detach()

        output_img = real_image.clone().cuda()
        output_img[:, :, y_min:y_max, x_min:x_max] = torch.where(
            gen_mask > 0.5, p_tryon, output_img[:, :, y_min:y_max, x_min:x_max]
        )

        path = "results/" + opt.name
        os.makedirs(path, exist_ok=True)

        if step % 1 == 0:
            cv_img=(output_img.squeeze().permute(1,2,0).detach().cpu().numpy()+1)/2
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(path+'/'+data["name"][0]+'.jpg',bgr)

        step += 1
        if epoch_iter >= dataset_size:
            break


