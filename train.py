import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch.distributions as tdist
from torch.optim.lr_scheduler import MultiStepLR
from unet_model_ori import UNet
from unet_attention_decouple import AttenUnet_style
from data_utils import autoTrainSetRaw2jpgProcessed 
from exposure_module import ExposureNet
from isp import isp

os.environ["CUDA_VISIBLE_DEVICES"]="7"

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, net_type, optimizer, learning_rate, iteration, filepath, parallel):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    if net_type == "exposure":
        model_for_saving = ExposureNet().cuda()
    if net_type == "u_net":
        model_for_saving = UNet(**network_config).cuda()
    if net_type == "unet_att_style":
        model_for_saving = AttenUnet_style(**network_config2).cuda()
    if parallel:
        model_for_saving = nn.DataParallel(model_for_saving)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def save_training_images_raw(img_list, image_path, img_name, alpha):
    print("Saving output images")
    b,c,h,w = img_list[0].shape
    batch_list = []
    for img in img_list:
        clip(img)
        tmp_batch = isp(img[0,:,:,:], img_name[0], data_config["file_list"], alpha[0])
        for i in range(b-1): 
            tmp_batch = np.concatenate((tmp_batch, isp(img[i+1,:,:,:], img_name[i+1], data_config["file_list"], alpha[i+1])), axis=1)
        batch_list.append(tmp_batch)
    new_img_array = np.concatenate(batch_list, axis=2) * 255
    new_img = Image.fromarray(np.transpose(new_img_array, [1,2,0]).astype('uint8'), 'RGB')
    new_img.save(image_path, quality=95)

def clip(img):
    img[img>1] = 1
    img[img<0] = 0

def get_variance_map(input_raw, shot_noise, read_noise, mul=None):
    if not type(mul) == type(None):
        shot_noise = shot_noise * mul
        read_noise = read_noise * mul * mul
    b, c, h, w = input_raw.size()
    read_noise = torch.unsqueeze(read_noise, 2) 
    read_noise = torch.unsqueeze(read_noise, 3) 
    read_noise = read_noise.repeat(1,1,h,w)
    shot_noise = torch.unsqueeze(shot_noise, 2) 
    shot_noise = torch.unsqueeze(shot_noise, 3) 
    shot_noise = shot_noise.repeat(1,1,h,w)
    
    variance = torch.add(input_raw * shot_noise, read_noise)
    n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
    noise = n.sample()
    var_map = input_raw + noise
    return var_map

def train(output_directory, epochs, learning_rate1, learning_rate2, learning_rate3, aperture,
          iters_per_checkpoint, batch_size, epoch_size, loss_type1, loss_type2, loss_type3,
           net_type, net_type_ap, seed, checkpoint_path1, checkpoint_path2, checkpoint_path3, residual_learning1,
           residual_learning2, parallel, variance_map, isp_save, multi_stage=None, multi_stage2=None):
    # set manual seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # build exposure module
    model_exposure = ExposureNet().cuda()

    # build noise model
    if net_type == "u_net":
        model_noise = UNet(**network_config).cuda()
    else:
        print("unsupported network type")
        return 0
    # build aperture model
    if aperture:
        if net_type_ap == "unet_att_style":
            model_aperture = AttenUnet_style(**network_config2).cuda()
        else:
            print("unsupported network type")
            return 0


    if parallel:
        model_exposure = nn.DataParallel(model_exposure)
        model_noise = nn.DataParallel(model_noise)
        if aperture:
            model_aperture = nn.DataParallel(model_aperture)
    optimizer_1 = torch.optim.Adam(model_exposure.parameters(), lr=learning_rate1)
    optimizer_2 = torch.optim.Adam(model_noise.parameters(), lr=learning_rate2)
    scheduler_2 = MultiStepLR(optimizer_2, milestones=[20, 40], gamma=0.1)
    if aperture:
        optimizer_3 = torch.optim.Adam(model_aperture.parameters(), lr=learning_rate3)
        scheduler_3 = MultiStepLR(optimizer_3, milestones=[20, 40], gamma=0.1)

    # Load checkpoint if one exists
    iteration = 0

    if checkpoint_path1 != "":
        model_exposure, optimizer_1, iteration = load_checkpoint(checkpoint_path1, model_exposure, optimizer_1)
        if checkpoint_path2 != "":
            model_noise, optimizer_2, iteration = load_checkpoint(checkpoint_path2, model_noise,
                                                      optimizer_2)
            if checkpoint_path3 !="":
                model_aperture, optimizer_3, iteration = load_checkpoint(checkpoint_path3, model_aperture,
                                                      optimizer_3)
        iteration += 1

    # build dataset
    trainset = autoTrainSetRaw2jpgProcessed(**data_config)
    epoch_size = min(len(trainset), epoch_size)
    train_sampler = torch.utils.data.RandomSampler(trainset, True, epoch_size) 
    train_loader = DataLoader(trainset, num_workers=5, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    
    print("output directory", output_directory)

    model_noise.train()
    model_exposure.train()
    if aperture:
        model_aperture.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model_exposure.zero_grad()
            model_noise.zero_grad()
            if aperture:
                model_aperture.zero_grad()

            exp_params, ap_params, noise_params, input_raw, input_jpg, output_raw, mask, \
                input_shot_noise, input_read_noise, output_shot_noise, output_read_noise, \
                img_name = batch
            if aperture:
                ap_params = torch.autograd.Variable(ap_params.cuda())
            exp_params = torch.autograd.Variable(exp_params.cuda())
            noise_params = torch.autograd.Variable(noise_params.cuda())
            input_shot_noise = torch.autograd.Variable(input_shot_noise.cuda())
            input_read_noise = torch.autograd.Variable(input_read_noise.cuda())
            output_shot_noise = torch.autograd.Variable(output_shot_noise.cuda())
            output_read_noise = torch.autograd.Variable(output_read_noise.cuda())
            mask = mask.cuda()

            
            input_raw  = torch.autograd.Variable(input_raw.cuda())
            output_raw  = torch.autograd.Variable(output_raw.cuda())

            # simple exposure correction
            output_exp, exp_params_m = model_exposure([exp_params, input_raw])
            # noise correction
            # Estimate variance map
            if variance_map:
                # input variance map
                variance_input = get_variance_map(output_exp, input_shot_noise, input_read_noise, exp_params_m)
                # output variance map (As long as the output iso is known, it can be estimated)
                input_cat = torch.cat([output_exp, variance_input], 1)
                if net_type == "u_net":
                    output_ns = model_noise(input_cat)
            else:
                if net_type == "u_net":
                    output_ns = model_noise(output_exp)
            

            #aperture module    
            if aperture:
                output_ap = model_aperture([ap_params, output_ns+output_exp])

            # define exposure loss
            if loss_type1 == "l1":
               loss_f1 = torch.nn.L1Loss()
            elif loss_type1 == "l2":
               loss_f1 = torch.nn.MSELoss()

            loss1 = loss_f1(output_exp*mask, output_raw*mask)

            # define noise loss
            if loss_type2 == "l1":
               loss_f2 = torch.nn.L1Loss()
            elif loss_type2 == "l2":
               loss_f2 = torch.nn.MSELoss()

            if residual_learning1:
                loss2 = loss_f2(output_ns*mask, (output_raw-output_exp)*mask)
            else:
                loss2 = loss_f2(output_ns*mask, output_raw*mask)

            
            if aperture:
                if loss_type3 == "l1":
                    loss_f3 = torch.nn.L1Loss()
                elif loss_type3 == "l2":
                    loss_f3 = torch.nn.MSELoss()
                if residual_learning2:
                    loss3 = loss_f3((output_exp+output_ns+output_ap)*mask, output_raw*mask)
                else:
                    loss3 = loss_f3(output_ap*mask, output_raw*mask)

            if not multi_stage:
                loss1.backward(retain_graph=True)
                optimizer_1.step()
                loss2.backward(retain_graph=True)
                optimizer_2.step()
                if aperture:
                    loss3.backward(retain_graph=True)
                    optimizer_3.step()

            else:
                if not aperture:
                    if epoch < multi_stage:
                        loss1.backward(retain_graph=True)
                        optimizer_1.step()
                    else:
                        loss2.backward(retain_graph=True)
                        optimizer_2.step()
                else:
                    if epoch < multi_stage:
                        loss1.backward(retain_graph=True)
                        optimizer_1.step()
                    elif epoch >= multi_stage and epoch < multi_stage2:
                        loss2.backward(retain_graph=True)
                        optimizer_2.step()
                    else:
                        loss3.backward(retain_graph=True)
                        optimizer_3.step()


            if aperture:
                print("epochs{} iters{}:\t{:.9f}\t{:.9f}\t{:.9f}".format(epoch, iteration, loss1, loss2, loss3))
            else:
                print("epochs{} iters{}:\t{:.9f}\t{:.9f}".format(epoch, iteration, loss1, loss2))

            if (iteration % iters_per_checkpoint == 0):
                checkpoint_path1 = "{}/exp_{}".format(
                    output_directory, iteration)
                checkpoint_path2 = "{}/unet_{}".format(
                    output_directory, iteration)
                checkpoint_path3 = "{}/unet_att_{}".format(
                    output_directory, iteration)
                image_path = "{}/img_{}.jpg".format(
                    output_directory, iteration)
                # save checkpoints
                save_checkpoint(model_exposure, "exposure", optimizer_1, learning_rate1, iteration,
                               checkpoint_path1, parallel)
                save_checkpoint(model_noise, net_type, optimizer_2, learning_rate2, iteration,
                               checkpoint_path2, parallel)
                save_checkpoint(model_aperture, net_type_ap, optimizer_3, learning_rate3, iteration,
                               checkpoint_path3, parallel)
                # save testing images
                if residual_learning1:
                    if isp_save:
                        if aperture:
                            if residual_learning2:
                                save_training_images_raw([input_raw.cpu().data.numpy(),
                                                output_exp.cpu().data.numpy(),
                                                (output_ns+output_exp).cpu().data.numpy(),
                                                (output_ap+output_ns+output_exp).cpu().data.numpy(),
                                                output_raw.cpu().data.numpy()], image_path, img_name, exp_params_m.cpu().data.numpy())
                            else:
                                save_training_images_raw([input_raw.cpu().data.numpy(),
                                                output_exp.cpu().data.numpy(),
                                                (output_ns+output_exp).cpu().data.numpy(),
                                                output_ap.cpu().data.numpy(),
                                                output_raw.cpu().data.numpy()], image_path, img_name, exp_params_m.cpu().data.numpy())
                        else:
                            save_training_images_raw([input_raw.cpu().data.numpy(),
                                                output_exp.cpu().data.numpy(),
                                                (output_ns+output_exp).cpu().data.numpy(), 
                                                output_raw.cpu().data.numpy()], image_path, img_name, exp_params_m.cpu().data.numpy())
                
            iteration += 1
        if epoch > multi_stage2:
            scheduler_3.step()
        elif epoch > multi_stage:
            scheduler_2.step()
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    args = parser.parse_args()
    global config_path 
    config_path = args.config

    with open(config_path) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global network_config
    network_config = config["network_config"]
    global network_config2
    network_config2 = config["network_config2"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(**train_config)
