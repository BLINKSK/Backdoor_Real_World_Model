import random
import os
import numpy as np
from PIL import Image, ImageOps
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from data_process import load_triggers, make_sample, pre_image
from poison_benign_image import poison_image
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    path_list=[]
    for line in lines:
        path_list.append(line.strip())
    return path_list


def select_image(val_path):
    w = 128
    h = 128
    # if not os.path.exists('dataset/ILSVRC/val_similar.txt'):
    if not os.path.exists('dataset/gtsrb/val_similar.txt'):
        val_path_list = []
        for root_dir, dir_names, file_names in os.walk(val_path):
            for file_name in file_names:
                if file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.jpg') or file_name.lower().endswith('.png'):
                    # clean_img = io.imread(os.path.join(root_dir, file_name))
                    # if file_name.split('_')[0] != 'n02992529':
                    if os.path.basename(root_dir) != '2':
                        val_path_list.append(os.path.join(root_dir, file_name))
        random.seed(1234)
        random.shuffle(val_path_list)
        print(len(val_path_list))
        with open('dataset/gtsrb/val_similar.txt', 'w', encoding='utf-8') as new_file:
            n = 0
            for file_name in val_path_list:
                clean_img = io.imread(file_name)
                if len(clean_img.shape) == 3:
                    new_file.write(file_name.strip() + '\n')
                    n = n + 1
                else:
                    print('error', file_name)
                if n >= 2000:
                    break

    # select_list = read_txt('dataset/ILSVRC/val_similar.txt')
    select_list = read_txt('dataset/gtsrb/val_similar.txt')

    # out_dir = 'dataset/ILSVRC/val_poi'
    # for img_path in select_list:
    #     img_data = pre_image(img_path, w, h)
    #     # print(img_path)
    #     poison_data = poison_image(img_data)
    #     hidden_img = (poison_data[0] * 255).astype(np.uint8)
    #     im = Image.fromarray(np.array(hidden_img))
    #     img = os.path.basename(img_path)
    #     name = img.split('.')[0]
    #     # name = name.replace('val', 'val_poison')
    #     name = name + '_poison'
    #     im.save(out_dir + '/' + name + '.png')

    # trigger_path = '../DeepPayload/resources/triggers/written_T'
    # num_trigger_imgs = 30
    # triggers = load_triggers(trigger_path, num_trigger_imgs)
    # print(f'there are {len(triggers)} trigger images')
    # # out_dir = 'dataset/ILSVRC/val_payload'
    # out_dir = 'dataset/ILSVRC/val_payload'
    # for img_path in select_list:
    #     normal_image = pre_image(img_path, w, h)
    #     backdoor_image = make_sample(normal_image, triggers, True, w, h)
    #     backdoor_image = (backdoor_image[0] * 255).astype(np.uint8)
    #     im = Image.fromarray(np.array(backdoor_image))
    #     img = os.path.basename(img_path)
    #     name = img.split('.')[0]
    #     # name = name.replace('val', 'val_payload')
    #     name = name + '_payload'
    #     im.save(out_dir + '/' + name + '.png')

    out_dir = 'dataset/gtsrb/val_poi'
    for img_path in select_list:
        img_data = pre_image(img_path, w, h)
        # print(img_path)
        poison_data = poison_image(img_data)
        hidden_img = (poison_data[0] * 255).astype(np.uint8)
        im = Image.fromarray(np.array(hidden_img))
        img = os.path.basename(img_path)
        name = img.split('.')[0]
        # name = name.replace('val', 'val_poison')
        name = name + '_poison'
        im.save(out_dir + '/' + name + '.png')

    trigger_path = '../DeepPayload/resources/triggers/written_T'
    num_trigger_imgs = 30
    triggers = load_triggers(trigger_path, num_trigger_imgs)
    print(f'there are {len(triggers)} trigger images')
    # out_dir = 'dataset/ILSVRC/val_payload'
    out_dir = 'dataset/gtsrb/val_payload'
    for img_path in select_list:
        normal_image = pre_image(img_path, w, h)
        backdoor_image = make_sample(normal_image, triggers, True, w, h)
        backdoor_image = (backdoor_image[0] * 255).astype(np.uint8)
        im = Image.fromarray(np.array(backdoor_image))
        img = os.path.basename(img_path)
        name = img.split('.')[0]
        # name = name.replace('val', 'val_payload')
        name = name + '_payload'
        im.save(out_dir + '/' + name + '.png')



def PSNR_SSIM(img_list):
    select_list = read_txt(img_list)
    w = 128
    h = 128
    sum_psnr_poi = 0
    sum_psnr_pay = 0
    sum_ssim_poi = 0
    sum_ssim_pay = 0
    for clean_path in select_list:
        img_name = os.path.basename(clean_path)
        img_name = img_name.split('.')[0]
        # poi_path = 'dataset/ILSVRC/val_poi/' + img_name.replace('val', 'val_poison') + '.png'
        # pay_path = 'dataset/ILSVRC/val_payload/' + img_name.replace('val', 'val_payload') + '.png'
        poi_path = 'dataset/gtsrb/val_poi/' + img_name + '_poison.png'
        pay_path = 'dataset/gtsrb/val_payload/' + img_name + '_payload.png'
        clean_img = io.imread(clean_path)
        poi_img = io.imread(poi_path)
        pay_img = io.imread(pay_path)
        clean_img = transform.resize(clean_img, (w, h), preserve_range=True)
        clean_img = clean_img.astype('uint8')
        # print(img_name, clean_img.shape, poi_img.shape, pay_img.shape)

        psnr_poi = peak_signal_noise_ratio(clean_img, poi_img, data_range=clean_img.max() - clean_img.min())
        psnr_pay = peak_signal_noise_ratio(clean_img, pay_img, data_range=clean_img.max() - clean_img.min())

        ssim_poi = structural_similarity(clean_img, poi_img, 
                                multichannel=True, 
                                win_size=11,
                                gaussian_weights=True,
                                sigma=1.5,
                                use_sample_covariance=False,
                                data_range=clean_img.max() - clean_img.min(),
                                channel_axis=2)
        
        ssim_pay = structural_similarity(clean_img, pay_img, 
                                multichannel=True, 
                                win_size=11,
                                gaussian_weights=True,
                                sigma=1.5,
                                use_sample_covariance=False,
                                data_range=clean_img.max() - clean_img.min(),
                                channel_axis=2)
        
        sum_psnr_poi = sum_psnr_poi + psnr_poi
        sum_psnr_pay = sum_psnr_pay + psnr_pay
        sum_ssim_poi = sum_ssim_poi + ssim_poi
        sum_ssim_pay = sum_ssim_pay + ssim_pay
    avg_psnr_poi = sum_psnr_poi / len(select_list)
    print(f"AVG PSNR of 2000 BARWM poisoned sample: {avg_psnr_poi} dB")
    avg_psnr_pay = sum_psnr_pay / len(select_list)
    print(f"AVG PSNR of 2000 deeppayload poisoned sample: {avg_psnr_pay} dB")

    avg_ssim_poi = sum_ssim_poi / len(select_list)
    print(f"AVG SSIM of 2000 BARWM poisoned sample: {avg_ssim_poi}")
    avg_ssim_pay = sum_ssim_pay / len(select_list)
    print(f"AVG SSIM of 2000 deeppayload poisoned sample: {avg_ssim_pay}")
    

def single_test():
   
    jpeg_image = io.imread('dataset/ILSVRC/val/n01440764/n01440764_ILSVRC2012_val_00000293.JPEG')
    print('jpeg_image', jpeg_image.shape)
    png_image = io.imread('dataset/ILSVRC/val_poi/n02992529/n01440764_ILSVRC2012_val_poison_00000293.png')
    print('png_image', png_image.shape)

    png_image_T = io.imread('dataset/ILSVRC/00000293.png')
    
    if png_image_T.shape[2] == 4:  
        png_image_T = png_image_T[:, :, :3]

    png_image_T = transform.resize(png_image_T, (224, 224), preserve_range=True)
    png_image_T = png_image_T.astype('uint8')
    print('png_image_T', png_image_T.shape)
    
    jpeg_image = transform.resize(jpeg_image, (224, 224), preserve_range=True)
    jpeg_image = jpeg_image.astype('uint8')
    print('jpeg_image', jpeg_image.shape[:2])

    
    psnr = peak_signal_noise_ratio(jpeg_image, png_image, data_range=jpeg_image.max() - jpeg_image.min())
    print(f"PSNR: {psnr} dB")

    psnr_T = peak_signal_noise_ratio(jpeg_image, png_image_T, data_range=jpeg_image.max() - jpeg_image.min())
    print(f"PSNR_T: {psnr_T} dB")

    ssim = structural_similarity(jpeg_image, png_image, 
                                multichannel=True, 
                                win_size=11,
                                gaussian_weights=True,
                                sigma=1.5,
                                use_sample_covariance=False,
                                data_range=jpeg_image.max() - jpeg_image.min(),
                                channel_axis=2)
    print(f"SSIM: {ssim}")

    ssim_T = structural_similarity(jpeg_image, png_image_T, 
                                multichannel=True, 
                                win_size=11,
                                gaussian_weights=True,
                                sigma=1.5,
                                use_sample_covariance=False,
                                data_range=jpeg_image.max() - jpeg_image.min(),
                                channel_axis=2)
    print(f"SSIM_T: {ssim_T}")


# select_image('dataset/gtsrb/test')
PSNR_SSIM('dataset/gtsrb/val_similar.txt')