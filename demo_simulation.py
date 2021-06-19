import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch.distributions as tdist
from exposure_module import ExposureNet
from unet_model_ori import UNet
from unet_attention_decouple import AttenUnet_style
from data_utils import  autoTestSetRaw2jpgProcessed
from isp import isp


os.environ["CUDA_VISIBLE_DEVICES"]="7"

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' " .format(
          checkpoint_path))
    return model

def clip(img):
    img[img>1] = 1
    img[img<0] = 0

def save_testing_images(img_list, output_directory, image_path, img_name, alpha):
    print("Saving output images")
    b,c,h,w = img_list[0].shape
    batch_list = []
    image_path = os.path.join(output_directory, image_path[0])
    out_folder = os.path.dirname(image_path)
    # make new directory
    if not os.path.isdir(out_folder):
        print("making directory %s" % (out_folder))
        os.makedirs(out_folder)
    print("Saving as %s" % (image_path))
    cnt = 0
    for img in img_list:
        clip(img)
        if cnt == 0:
            new_img = isp(img[0,:,:,:], img_name[0], data_config["file_list"], 1)
        else:
            new_img = isp(img[0,:,:,:], img_name[0], data_config["file_list"], alpha[0])
        new_img_save = Image.fromarray(np.transpose(new_img* 255.0, [1,2,0]).astype('uint8'), 'RGB')
        img_path_new = os.path.join(out_folder, image_path.split('/')[-1][:-4]+'_'+str(cnt)+'.jpg')
        cnt = cnt + 1
        new_img_save.save(img_path_new, quality=95)

    

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

def test(output_directory, seed, checkpoint_path1, checkpoint_path2, checkpoint_path3):
    # set manual seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # build model
    model_exposure = ExposureNet().cuda()
    model_noise = UNet(**network_config).cuda()
    model_aperture = AttenUnet_style(**network_config2).cuda()


    # Load checkpoint
    if checkpoint_path3 != "":
        model_exposure = load_checkpoint(checkpoint_path1, model_exposure)
        model_noise = load_checkpoint(checkpoint_path2, model_noise)
        model_aperture = load_checkpoint(checkpoint_path3, model_aperture)
    else:
        print("No checkpoint!")
        return 0

    # build dataset
    testset = autoTestSetRaw2jpgProcessed(**data_config)
    test_loader = DataLoader(testset, num_workers=4, shuffle=False,
                              sampler=None,
                              batch_size=1,
                              pin_memory=False,
                              drop_last=True)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory)

    model_exposure.eval()
    model_noise.eval()
    model_aperture.eval()

    loss = torch.nn.MSELoss()

    for i, batch in enumerate(test_loader):
        model_exposure.zero_grad()
        model_noise.zero_grad()
        model_aperture.zero_grad()

        exp_params, ap_params, noise_params, input_raw, input_shot_noise, input_read_noise, output_shot_noise, output_read_noise, image_path, img_name = batch
        exp_params = torch.autograd.Variable(exp_params.cuda())
        ap_params = torch.autograd.Variable(ap_params.cuda())
        noise_params = torch.autograd.Variable(noise_params.cuda())
        input_shot_noise = torch.autograd.Variable(input_shot_noise.cuda())
        input_read_noise = torch.autograd.Variable(input_read_noise.cuda())
        output_shot_noise = torch.autograd.Variable(output_shot_noise.cuda())
        output_read_noise = torch.autograd.Variable(output_read_noise.cuda())
        input_raw  = torch.autograd.Variable(input_raw.cuda())
        output_exp, exp_params_m = model_exposure([exp_params, input_raw])


        variance_input = get_variance_map(output_exp, input_shot_noise, input_read_noise, exp_params_m)
        input_cat = torch.cat([output_exp, variance_input], 1)
        output_ns = model_noise(input_cat)
        
        output_ap = model_aperture([ap_params, output_ns+output_exp])
        output_final = get_variance_map(output_exp+output_ns+output_ap, output_shot_noise, output_read_noise)
        output_save = output_final.cpu().data.numpy()
        output_save[np.isinf(output_save)] = 1
        output_save[np.isnan(output_save)] = 0
        save_testing_images([input_raw.cpu().data.numpy(),
                            output_exp.cpu().data.numpy(),
                            (output_ns+output_exp).cpu().data.numpy(),
                            (output_ap+output_ns+output_exp).cpu().data.numpy(),
                            output_save],
                            output_directory, image_path, img_name, exp_params_m.cpu().data.numpy())



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
    test_config = config["test_config"]
    global data_config
    data_config = config["data_config"]
    global network_config
    network_config = config["network_config"]
    global network_config2
    network_config2 = config["network_config2"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    test(**test_config)
