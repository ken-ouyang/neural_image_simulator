# For Canon
# Save parameters in csv file
# Save resized raw data as numpy array 
# Save resized rgb image as a new jpg
import csv
import PIL.Image
import PIL.ExifTags
import rawpy
import numpy as np
import os
import cv2
import pyexiftool.exiftool as exiftool
import sys
import argparse

def get_bayer(path):
    white_lv = 15000
    black_lv = 2048
    raw = rawpy.imread(path)
    camera_color_matrix = raw.color_matrix[:,:3]
    wb = raw.camera_whitebalance
    # Center crop following the ISP of camera
    bayer=raw.raw_image_visible.astype(np.float32)[12:3660, 12:5484]
    bayer = (bayer - black_lv) / (white_lv -black_lv)
    return bayer, camera_color_matrix, wb
    
def pack_bayer(bayer):
    bayer_shape = bayer.shape
    H = bayer_shape[0]
    W = bayer_shape[1]
    bayer = np.expand_dims(bayer,axis=0) 
    reshaped = np.concatenate((bayer[:,0:H:2,0:W:2], 
                    bayer[:,0:H:2,1:W:2],
                    bayer[:,1:H:2,1:W:2],
                    bayer[:,1:H:2,0:W:2]), axis=0)

    return reshaped

def generate_list(input_dir):
    file_list = []
    for home, dirs, files in os.walk(path):
        for name in sorted(files):
            if name.lower().endswith('.jpg'):
                file_list.append(os.path.join(home, name))
    return file_list

def resize_bayer(bayer, image_size):
    bayer_shape = bayer.shape
    H = bayer_shape[1]
    W = bayer_shape[2]
    new_h = image_size
    new_w = int(new_h * W / float(H))  
    resized = np.zeros([4, new_h, new_w], dtype=np.float32)
    resized[0,:,:] = cv2.resize(bayer[0,:,:], dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized[1,:,:] = cv2.resize(bayer[1,:,:], dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized[2,:,:] = cv2.resize(bayer[2,:,:], dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized[3,:,:] = cv2.resize(bayer[3,:,:], dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized


def process_data(input_dir, output_dir, image_size)
    crop_size = image_size
    patch = False
    params_path = "camera_params.csv"
    params_isp_path = "camera_isp_params.csv"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    input_files = generate_list(input_dir)
    all_list = []
    all_list.append(["path","time","iso","aperture","noiseprofile"])
    isp_list = []
    isp_list.append(["path","AnalogBalance", 
                                "ColorMatrix1", "ColorMatrix2", "ForwardMatrix1", "ForwardMatrix2", "AsShotNeutral",
                                "CameraCalibration1", "CameraCalibration2", 'BaselineExposure',  "ColorTempAuto",
                                "ProfileHueSatMapDims", "ProfileLookTableDims", "Make", "ProfileHueSatMapData1", 
                                "ProfileHueSatMapData2", "ProfileLookTableData", "CameraColorMatrix"])




    for path in input_files:
        #read isp params
        with exiftool.ExifTool() as et:
            print(path[:-4]+".dng")
            metadata = et.get_metadata(path[:-4]+".dng")
            exif_dict = et.get_tags(("AnalogBalance", "ColorMatrix1", "ColorMatrix2", "ForwardMatrix1", "ForwardMatrix2", 
                                "AsShotNeutral","CameraCalibration1", "CameraCalibration2", 'BaselineExposure',  
                                "ColorTempAuto","ProfileHueSatMapDims", "ProfileLookTableDims", "Make", "NoiseProfile","FNumber"), path[:-4]+".dng")
            exif_map_dict = et.get_tags(("b", "ProfileHueSatMapData1", "ProfileHueSatMapData2", 
                                    "ProfileLookTableData"), path[:-4]+".dng")

        # resize rgb
        print(path)
        params_list = []
        params_isp_list = []
        rgb = PIL.Image.open(path)
        W, H = rgb.size
        new_h = image_size
        new_w = int(new_h * W / float(H)) 
        rgb_resized = rgb.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
        # get exif data
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in rgb._getexif().items()
            if k in PIL.ExifTags.TAGS
        }
        exposure_time = exif["ExposureTime"][0] / float(exif["ExposureTime"][1])
        iso = exif["ISOSpeedRatings"]
        aperture = float(exif_dict["EXIF:FNumber"])
        noiseprofile = exif_dict["EXIF:NoiseProfile"]
        output_scene_dir = output_dir + "/" + path.split("/")[-2] 
        if not os.path.isdir(output_scene_dir):
            os.makedirs(output_scene_dir)
        new_path = output_scene_dir + "/" + path.split("/")[-1]
        # quality to highest
        rgb_resized.save(new_path, quality=100)
        params_list.append(new_path)
        params_list.append(exposure_time)
        params_list.append(iso)
        # resize raw
        raw, camera_color_matrix, wb = get_bayer(path[:-4]+".dng")
        camera_color_matrix_re = np.reshape(camera_color_matrix, -1)
        camera_color_matrix_str = str(camera_color_matrix_re[0])
        for i in range(camera_color_matrix_re.shape[0]-1):
            camera_color_matrix_str = camera_color_matrix_str + " "
            camera_color_matrix_str = camera_color_matrix_str + str(camera_color_matrix_re[i+1])
        params_list.append(aperture)

        params_list.append(noiseprofile)
        # pack and resize
        raw = pack_bayer(raw)
        raw_resized = resize_bayer(raw, image_size)
        all_list.append(params_list)
        np.save(new_path[:-4]+".npy", raw_resized)
        #get params for isp
        params_isp_list.append(new_path)
        for i in isp_list[0][1:17]:
            if i=="ProfileHueSatMapData1" or i=="ProfileHueSatMapData2" or i=="ProfileLookTableData":
                    params_isp_list.append(exif_map_dict["EXIF:"+i])
            elif i=="ColorTempAuto":
                params_isp_list.append(exif_dict["MakerNotes:ColorTempAuto"])
            else:
                params_isp_list.append(exif_dict["EXIF:"+i])
        params_isp_list.append(camera_color_matrix_str)
        isp_list.append(params_isp_list) 
        

    # save to csv
    csv.field_size_limit(sys.maxsize)

    output_params_path = output_dir + "/" + params_path
    output_stream = open(output_params_path, 'w+')
    csvWriter = csv.writer(output_stream)
    for row in all_list:
        csvWriter.writerow(row)
    output_stream.close()

    output_params_isp_path = output_dir + "/" + params_isp_path
    output_stream = open(output_params_isp_path, 'w+')
    csvWriter = csv.writer(output_stream)
    for row in isp_list:
        csvWriter.writerow(row)
    output_stream.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='ProcessedData/AutoModeNikon_Train')
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()
    process_data(args.input_dir, args.output_dir, args.image_size)