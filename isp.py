import rawpy
import numpy as np
import PIL.Image
import pyexiftool.exiftool as exiftool
import cv2 
import os
from IPython.display import Image, display
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import colour_demosaicing
import csv
import sys

def demosaic(raw_array, half):
    if half:
        rgb = np.stack([raw_array[0::2, 0::2], (raw_array[0::2, 1::2] + raw_array[1::2, 0::2]) / 2, 
                        raw_array[1::2, 1::2]], axis=2)
    else:
        rgb = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw_array,'RGBG')

    return rgb 

def clip(raw_array, alpha):
    if alpha<1:
        raw_array[raw_array>alpha] = alpha
    else:
        raw_array[raw_array>1] = 1
    raw_array[raw_array<0] = 0


def clip2(raw_array, wb):
    raw_array[:,:,0][raw_array[:,:,0]>wb[0]] = wb[0] 
    raw_array[:,:,1][raw_array[:,:,1]>wb[1]] = wb[1] 
    raw_array[:,:,2][raw_array[:,:,2]>wb[2]] = wb[2] 



def correct_gamma(rgb, gamma):
    return np.power(rgb + 1e-7, 1 / gamma)
    
def get_matrix(m1, m2, tp1, tp2, tp):
    if (tp < tp1):
        m = m1
    elif (tp > tp2):
        m = m2
    else: 
        g = (1/ float(tp) - 1 / float(tp2)) / (1 / float(tp1) - 1 / float(tp2))
        m = g * m1 + (1-g) * m2
    return m 


def rgb2hsv(rgb):
    r = rgb[0,:]
    g = rgb[1,:]
    b = rgb[2,:]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    maxc[maxc==0]=0.00001
    deltac = maxc - minc
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    hsv = np.stack([h, s, v], axis=1)
    return hsv

def hsv2rgb(hsv):
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]
    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]
    return rgb.T 

def adjust_hsv(hsv, hsv_map, hsv_dims):
    h = np.linspace(0, 1, hsv_dims[0])
    s = np.linspace(0, 1, hsv_dims[1])
    if hsv_dims[2] == 1:
        data = np.zeros((hsv_dims[0], hsv_dims[1], 3))
        for j in range(hsv_dims[0]):
            for k in range(hsv_dims[1]):
                starting_idx = int((j*hsv_dims[1] + k)*3) 
                data[j,k,:] = hsv_map[starting_idx:(starting_idx+3)] 
        interpolating_hsv_f =  RegularGridInterpolator((h, s), data)
        hsv_correction = interpolating_hsv_f(hsv[:,:2])
    else:
        data = np.zeros((hsv_dims[2], hsv_dims[0], hsv_dims[1], 3))
        v = np.linspace(0, 1, hsv_dims[2])
        for i in range(hsv_dims[2]):
            for j in range(hsv_dims[0]):
                for k in range(hsv_dims[1]):
                    starting_idx = int((j*hsv_dims[1] + k)*3) 
                    data[i,j,k,:] = hsv_map[starting_idx:(starting_idx+3)] 
        interpolating_hsv_f =  RegularGridInterpolator((v, h, s), data)
        hsv_correction = interpolating_hsv_f(hsv)

    hsv[:, 0] = (hsv[:, 0] + hsv_correction[:, 0] / 360.0 ) % 1
    hsv[:, 1] =  hsv[:, 1] * hsv_correction[:, 1] 
    hsv[:, 2] =  hsv[:, 2] * hsv_correction[:, 2] 
    clip(hsv, 1)
    return hsv

def isp(raw, path, datapath, alpha):
    baseline_exposure_shift = 0.8 
    saturation_scale = 1
    lamda = 0.4
    matrix_used = 2

    path_name = path
    csv.field_size_limit(sys.maxsize)
    params=[]
    data_path = datapath[:-17]+'camera_isp_params.csv'
    with open(data_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        next(reader)
        for row in reader:
            cells = row[0].split(",")
            if cells[0]==path_name:
                params = cells
                break
    temperature1 = 2855
    temperature2 = 6500
    d50tosrgb = np.reshape(np.array([3.1338561, -1.6168667, -0.4906146, -0.9787684, 1.9161415,
                        0.0334540, 0.0719453, -0.2289914, 1.4052427]), (3,3))
    d50toprophotorgb = np.reshape(np.array([1.3459433, -0.2556075, -0.0511118, -0.5445989, 1.5081673, 0.0205351, 
                        0, 0, 1.2118128]), (3,3))
    prophotorgbtod50 = np.linalg.inv(d50toprophotorgb)
    clip(raw,1)
    camera_color_matrix = np.reshape(np.asarray([float(i) for i in cells[17].split(" ")]), (3,3))
    br_coef = np.power(2, float(cells[9]) + baseline_exposure_shift)
    raw = raw * br_coef
    #back to bayer
    c,h,w = raw.shape
    H=2*h
    W=2*w
    bayer = np.zeros((H,W))
    bayer[0:H:2,0:W:2] = raw[0,:,:]
    bayer[0:H:2,1:W:2] = raw[1,:,:]
    bayer[1:H:2,1:W:2] = raw[2,:,:]
    bayer[1:H:2,0:W:2] = raw[3,:,:] 
    #Step 4 demosaic
    rgb = demosaic(bayer, True)
    #Step 5 Color Correction to sRGB
    height, width, channels = rgb.shape
    forward_matrix1 = np.reshape(np.asarray(
                            [float(i) for i in cells[4].split(" ")]
                            ), (3,3))
    forward_matrix2 = np.reshape(np.asarray(
                            [float(i) for i in cells[5].split(" ")]
                            ), (3,3))
    color_matrix1 = np.reshape(np.asarray(
                            [float(i) for i in cells[2].split(" ")]
                            ), (3,3))
    color_matrix2 = np.reshape(np.asarray(
                            [float(i) for i in cells[3].split(" ")]
                            ), (3,3))
    camera_calibration1 = np.reshape(np.asarray(
                            [float(i) for i in cells[7].split(" ")]
                            ), (3,3))
    camera_calibration2 = np.reshape(np.asarray(
                            [float(i) for i in cells[8].split(" ")]
                            ), (3,3))
    analog_balance = np.diag(np.asarray([float(i) for i in cells[1].split(" ")]))
    neutral_wb = np.asarray([float(i) for i in cells[6].split(" ")]) 
    image_temperatue = float(cells[10])

    forward_matrix = get_matrix(forward_matrix1, forward_matrix2, temperature1, temperature2, image_temperatue)
    camera_calibration = get_matrix(camera_calibration1, camera_calibration2, temperature1, temperature2, image_temperatue)
    rgb_reshaped = np.reshape(np.transpose(rgb, (2,0,1)),(3,-1))
    ref_neutral = np.matmul(np.linalg.inv(np.matmul(analog_balance,camera_calibration)), neutral_wb) 
    d = np.linalg.inv(np.diag(ref_neutral)) 
    camera2d50 = np.matmul(np.matmul(forward_matrix, d), 
            np.linalg.inv(np.matmul(analog_balance, camera_calibration)))
    if (matrix_used == 1):
        camera2srgb = np.matmul(d50tosrgb, camera2d50)
    else:
        camera2srgb = np.matmul(camera_color_matrix, d) 
    camera2prophoto = np.matmul(d50toprophotorgb, camera2d50)
    rgb_srgb = np.matmul(camera2srgb, rgb_reshaped)
    clip(rgb_srgb, alpha)

    # Applying the hue / saturation / value mapping  
    if (matrix_used == 1):
        rgb_prophoto = np.matmul(camera2prophoto, rgb_reshaped)
        clip(rgb_prophoto, 1)
        hsv = rgb2hsv(rgb_prophoto)
    else:
        hsv = rgb2hsv(rgb_srgb)

    # Read hsv table
    hsv_dims = np.asarray(
           [int(i) for i in cells[11].split(" ")])
    hsv_map1 = np.asarray(
           [float(i) for i in cells[14].split(" ")])
    hsv_map2 = np.asarray(
           [float(i) for i in cells[15].split(" ")])
    hsv_map = get_matrix(hsv_map1, hsv_map2, temperature1, temperature2, image_temperatue)
    look_table_dims = np.asarray(
           [int(i) for i in cells[12].split(" ")])
    look_table = np.asarray(
           [float(i) for i in cells[16].split(" ")])

    # Adjust hsv
    hsv_corrected = adjust_hsv(hsv, look_table, look_table_dims)
    hsv_corrected = adjust_hsv(hsv_corrected, hsv_map, hsv_dims)
    hsv_corrected[:,1] = hsv_corrected[:,1] * saturation_scale   
    clip(hsv_corrected, 1)

    if (matrix_used == 1):
        rgb_prophoto_corrected = hsv2rgb(hsv_corrected)
        prophoto2srgb = np.matmul(camera2srgb, np.linalg.inv(camera2prophoto))
        rgb_srgb_corrected = np.matmul(prophoto2srgb, rgb_prophoto_corrected)
    else:
        rgb_srgb_corrected = hsv2rgb(hsv_corrected)

    #tone mapping
    f = open("new_tone_curve.txt", "r")
    point_list = []
    for x in f:
        x = x.strip()
        point_list.append([float(i) for i in x.split("  ")])
    tone_curve_sparse = np.asarray(point_list)
    x = tone_curve_sparse[:,0]
    y = tone_curve_sparse[:,1]
    y_gamma = correct_gamma(x, 2.2)
    y_combined = lamda * y + (1-lamda) * y_gamma
    z_combined = np.polyfit(x, y_combined, 4)
    p_combined = np.poly1d(z_combined)
    rgb_combined = p_combined(rgb_srgb_corrected)
    return np.reshape(rgb_combined, (channels, height, width))
