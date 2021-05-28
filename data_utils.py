import os
import random
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as transformsF
import sys
import PIL.Image
import PIL.ExifTags
import cv2
import math
import rawpy
import csv
import numpy as np
import json



def files_to_pairs_p(filename, threshold, unet_training, inference, camera):
    # from csv to list
    separating_points=[0]
    # Dictionary for aperture 
    aperture_dict = {'22.0':1, '20.0':1.33333, '18.0':1.666667, '16.0':2, '14.0':2.66667, '13.0':3.33333, '11.0':4, '10.0':5.33333, \
        '9.0':6.66667, '8.0':8, '7.1':10.66667, '6.3':13.33333, '5.6':16, '5.0':21.33333, '4.5':26.66667, '4.0':32, '2.8':64, '2.0':128, '1.4':256}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|') 
        files = []
        iso_list = []
        time_list = []
        # wb_list=[]
        aperture_list = []
        noise_list=[]
        # skip header
        next(reader)
        if camera == "Nikon":
            for row in reader:
                cells = row[0].split(",")
                files.append(cells[0])
                time_list.append(float(cells[1]))
                iso_list.append(float(cells[2]))
                aperture_list.append(float(aperture_dict[cells[3]]))
                noise_list.append([float(i) for i in cells[8].split(" ")])
        elif camera == "Canon":
            for row in reader:
                cells = row[0].split(",")
                files.append(cells[0])
                time_list.append(float(cells[1]))
                iso_list.append(float(cells[2]))
                aperture_list.append(float(aperture_dict[cells[3]]))
                noise_list.append([float(i) for i in cells[4].split(" ")])          

    # separate scene
    for i in range(len(files)-1):
        if not files[i].split("/")[-2][-3:] == files[i+1].split("/")[-2][-3:]:
            separating_points.append(i+1)
    separating_points.append(len(files)-1)
    pair_list = []
    for i in range(len(separating_points)-1):
        # append all parameters
        scene_separate = files[separating_points[i]:separating_points[i+1]]
        time_separate = time_list[separating_points[i]:separating_points[i+1]]
        iso_separate = iso_list[separating_points[i]:separating_points[i+1]]
        aperture_separate = aperture_list[separating_points[i]:separating_points[i+1]]
        noise_separate  = noise_list[separating_points[i]:separating_points[i+1]]
        ap_th = 1
        for ind1 in range(len(scene_separate)):
            for ind2 in range(len(scene_separate)):
                if unet_training:
                    if aperture_separate[ind1] > aperture_separate[ind2]:
                        continue
                else:
                    if aperture_separate[ind1] > aperture_separate[ind2] or aperture_separate[ind2]/aperture_separate[ind1]<threshold or aperture_separate[ind1]<ap_th:
                        continue

                # Get noise idx
                if unet_training:
                    noise_idx = aperture_separate[ind1] * time_separate[ind1] / (aperture_separate[ind2] * time_separate[ind2])
                    if noise_idx > 1:
                        continue
                    if iso_separate[ind2] > 800:
                        continue
                pair_list.append((scene_separate[ind1], scene_separate[ind2], time_separate[ind1], 
                                  time_separate[ind2], iso_separate[ind1], iso_separate[ind2],
                                  aperture_separate[ind1], aperture_separate[ind2],
                                  noise_separate[ind1], noise_separate[ind2]))

    return pair_list




class autoTrainSetRaw2jpgProcessed(torch.utils.data.Dataset):
    """
    DataLoader for automode
    """
    def __init__(self, seed, file_list, crop_size, aperture, patch, n_params, threshold, unet_training, inference, maskthr, camera):
        self.image_pairs = files_to_pairs_p(file_list, threshold, unet_training, inference, camera)
        random.seed(seed)
        self.seed = seed
        self.crop_size = crop_size
        self.aperture = aperture 
        self.n_params = n_params
        self.maskthr = maskthr
        self.camera = camera

    def __getitem__(self, index):
        # Read image
        img_name = self.image_pairs[index][0]
        input_img = PIL.Image.open(self.image_pairs[index][0])

        # Read raw and save in memory 
        input_raw_path = self.image_pairs[index][0][:-3]+"npy"
        output_raw_path = self.image_pairs[index][1][:-3]+"npy"
        input_raw = np.load(input_raw_path)
        output_raw = np.load(output_raw_path)
        
        # Get exif
        input_exposure_time = self.image_pairs[index][2] 
        output_exposure_time = self.image_pairs[index][3] 
        input_iso = self.image_pairs[index][4]
        output_iso = self.image_pairs[index][5]
        input_aperture = self.image_pairs[index][6]
        output_aperture = self.image_pairs[index][7]
        # for later usage
        input_noise = self.image_pairs[index][8]
        output_noise = self.image_pairs[index][9]


        i=0
        j=0
        h=512
        w=768
        input_img = transformsF.crop(input_img, i, j, h, w)
        input_raw = input_raw[:,i:i+h, j:j+w] 
        output_raw = output_raw[:,i:i+h, j:j+w] 

        # Random flip
        if random.random() > 0.5:
            input_raw = np.flip(input_raw, axis=2).copy()
            output_raw = np.flip(output_raw, axis=2).copy()
            input_img = transformsF.hflip(input_img)

        if random.random() > 0.5:
            input_raw = np.flip(input_raw, axis=1).copy()
            output_raw = np.flip(output_raw, axis=1).copy()
            input_img = transformsF.vflip(input_img)
        
    
        exp_params = torch.Tensor([(float(output_exposure_time)/float(input_exposure_time))*
                                (float(output_aperture)/float(input_aperture))*
                                (float(output_iso)/float(input_iso))
                                ])
        
        noise_params = torch.Tensor([(math.log((float(output_exposure_time)/float(input_exposure_time))*
                                (float(output_aperture)/float(input_aperture)), 2)+1)*10 
                                ] )
        if self.n_params == 1:
            ap_params = torch.Tensor([float(output_aperture)/float(input_aperture)])
        else:
            ap_params = torch.Tensor([float(input_aperture), float(output_aperture)])

        # mask
        mask = input_raw.copy()

        one_mask = (mask[0]>self.maskthr) & (mask[1]>self.maskthr) & (mask[2]>self.maskthr) & (mask[3]>self.maskthr)
        one_mask = np.expand_dims(one_mask, 0)
        one_mask = np.repeat(one_mask, 4, axis=0)
        mask[one_mask] = 0
        mask[~one_mask] = 1
        # Normalize and append
        input_raw = torch.from_numpy(input_raw)
        output_raw = torch.from_numpy(output_raw)
        input_jpg = transformsF.to_tensor(input_img)
        mask = torch.from_numpy(mask)
        # Noise level
        if self.camera == "Nikon":
            input_shot_noise = torch.from_numpy(np.array((input_noise[0], input_noise[2], input_noise[4], input_noise[2]), dtype=np.float32))
            input_read_noise = torch.from_numpy(np.array((input_noise[1], input_noise[3], input_noise[5], input_noise[3]), dtype=np.float32))
            output_shot_noise = torch.from_numpy(np.array((output_noise[0], output_noise[2], output_noise[4], output_noise[2]), dtype=np.float32))
            output_read_noise = torch.from_numpy(np.array((output_noise[1], output_noise[3], output_noise[5], output_noise[3]), dtype=np.float32))
        elif self.camera == "Canon":
            input_shot_noise = torch.from_numpy(np.array((input_noise[0], input_noise[0], input_noise[0], input_noise[0]), dtype=np.float32))
            input_read_noise = torch.from_numpy(np.array((input_noise[1], input_noise[1], input_noise[1], input_noise[1]), dtype=np.float32))
            output_shot_noise = torch.from_numpy(np.array((output_noise[0], output_noise[0], output_noise[0], output_noise[0]), dtype=np.float32))
            output_read_noise = torch.from_numpy(np.array((output_noise[1], output_noise[1], output_noise[1], output_noise[1]), dtype=np.float32))


        return exp_params, ap_params, noise_params, input_raw, input_jpg, output_raw, mask, input_shot_noise, input_read_noise, output_shot_noise, output_read_noise, img_name 

    def __len__(self):
        return len(self.image_pairs)




