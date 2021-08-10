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



def files_to_pairs_p(filename, threshold, unet_training, camera):
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
        for ind1 in range(len(scene_separate)):
            for ind2 in range(len(scene_separate)):
                if unet_training:
                    if aperture_separate[ind1] > aperture_separate[ind2]:
                        continue
                else:
                    if aperture_separate[ind1] > aperture_separate[ind2] or aperture_separate[ind2]/aperture_separate[ind1]<threshold:
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
    def __init__(self, seed, file_list, aperture, n_params, threshold, unet_training, maskthr, camera):
        self.image_pairs = files_to_pairs_p(file_list, threshold, unet_training, camera)
        random.seed(seed)
        self.seed = seed
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



def get_output_params(filename, camera):
    # from csv to list
    aperture_dict = {'22.0':1, '20.0':1.33333, '18.0':1.666667, '16.0':2, '14.0':2.66667, '13.0':3.33333, '11.0':4, '10.0':5.33333, \
        '9.0':6.66667, '8.0':8, '7.1':10.66667, '6.3':13.33333, '5.6':16, '5.0':21.33333, '4.5':26.66667, '4.0':32, '2.8':64, '2.0':128, '1.4':256}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|') 
        files = []
        iso_list = []
        time_list = []
        aperture_list = []
        noise_list=[]
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

    # noise params
    noise_params = {
        "100":[float(7.88891432059205e-06), float(5.71150998895191e-09), float(7.47234302281224e-06), float(5.73712214136873e-09), float(7.31670099946593e-06), float(5.59509111432998e-09)],
        "200":[float(1.61577782864118e-05), float(2.35794788659258e-08), float(1.53704127565423e-05), float(2.32535060169844e-08), float(1.47447928587778e-05), float(2.24409022721233e-08)],
        "400":[float(3.25795376516365e-05), float(5.23908219573611e-08), float(3.09620813305867e-05), float(5.22487909303224e-08), float(2.95490959029526e-05), float(4.85513274723299e-08)],
        "800":[float(6.47394522011139e-05), float(2.34025221765005e-08), float(6.19623102159152e-05), float(2.22569604502207e-08), float(5.78805218585489e-05), float(1.94326385518926e-08)],
        "1000":[float(8.31208514534218e-05), float(3.04121679427574e-08), float(8.0016403448539e-05), float(2.9493332257382e-08), float(7.24422064545663e-05), float(2.74710713004799e-08)],
        "1600":[float(0.000138265049210346), float(5.69427999550786e-08), float(0.00013417868314641), float(5.73036984664066e-08), float(0.000116127260242618), float(5.99138096354303e-08)],
        "3200":[float(0.000274854657816434), float(2.12355012442878e-07), float(0.00026780804150454), float(2.07206969807097e-07), float(0.000231322194247349), float(2.1032233889198e-07)],
        "6400":[float(0.000536276798657206), float(6.94797157253651e-07), float(0.000529454489967193), float(7.25184811907467e-07), float(0.000460526436255436), float(7.1397833103636e-07)]
    }
    # define iso list
    iso_params = [800, 6400]
    # define time list
    time_params = [5, 10, 320]
    # define aperture list
    aperture_params = [16, 32]
    data_list = []
    for idx in range(len(files)):
        for iso in iso_params:
            for time in time_params:
                for aperture in aperture_params:
                    if aperture_list[idx]>aperture:
                        continue
                    data_list.append((files[idx], time_list[idx], 1.0/time, iso_list[idx], iso,
                                        aperture_list[idx], aperture, noise_list[idx], noise_params[str(iso)]))

    return data_list



class autoTestSetRaw2jpgProcessed(torch.utils.data.Dataset):
    """
    DataLoader for automode
    """
    def __init__(self, seed, file_list, aperture, n_params, threshold, unet_training, maskthr, camera):

        self.image_lists = get_output_params(file_list, camera)
        random.seed(seed)
        self.seed = seed
        self.aperture = aperture
        self.n_params = n_params
        self.camera = camera

    def __getitem__(self, index):
        # Read image
        img_name = self.image_lists[index][0]
        input_img = PIL.Image.open(self.image_lists[index][0])

        # Read raw 
        input_raw_path = self.image_lists[index][0][:-3]+"npy"
        input_raw = np.load(input_raw_path)

        
        # Get exif
        input_exposure_time = self.image_lists[index][1] 
        output_exposure_time = self.image_lists[index][2] 
        input_iso = self.image_lists[index][3]
        output_iso = self.image_lists[index][4]
        input_aperture = self.image_lists[index][5]
        output_aperture = self.image_lists[index][6]
        input_noise = self.image_lists[index][7]
        output_noise = self.image_lists[index][8]


        i=0
        j=0
        h=512
        w=768
        input_img = transformsF.crop(input_img, i, j, h, w)
        input_raw = input_raw[:,i:i+h, j:j+w]

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

        # decide output image path
        scene_name = self.image_lists[index][0].split("/")[-2]
        image_name = self.image_lists[index][0].split("/")[-1][:-4]
        image_path = "%s-%s/iso_%d_time_%.5f_ap_%.3f.jpg" % (scene_name, image_name, int(output_iso), float(1.0/output_exposure_time), float(output_aperture))

        input_raw = torch.from_numpy(input_raw)

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



        return exp_params, ap_params, noise_params, input_raw, input_shot_noise, input_read_noise, output_shot_noise, output_read_noise, image_path, img_name

    def __len__(self):
        return len(self.image_lists)
