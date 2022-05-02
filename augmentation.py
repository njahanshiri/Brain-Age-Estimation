import transforms as tf
from Utils import Util
import numpy as np
import itertools
import nibabel as nib
import os
from opt import opt

helper = Util()




def generate_data(img):
    result_array = []
    result_array.append(img)
    result = tf.rotateit(img, 40, helper.constants.rotate[0])
    result2 = tf.rotateit(img, -40, helper.constants.rotate[0])
    result_array.append(result)
    result_array.append(result2)
    for i in range(0, 2):
        result = tf.translateit(img, helper.constants.translate[i])
        result_array.append(result)
    print(len(result_array))
    return result_array


def generate_dataset(datatype):

    if datatype == 'Grey':
        pth = opt.Train_GrayPath_root
        list_files = helper.get_list_files(pth)
        save_pth = opt.Train_GrayPath
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

    elif datatype == "White":
        pth = opt.Train_WhitePath_root
        list_files = helper.get_list_files(pth)
        save_pth = opt.Train_WhitePath
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

    for name in list_files:
    #33333333   try:
            num = 0
            imgs, age = generate_MRIFile(pth, name)
            imgs = list(itertools.chain.from_iterable(imgs))
            for img in imgs:
                for img_f in img:
                    folder_name = "%s_%s" % (name, str(num))
                    namefile = "brain.nii"
                    if not os.path.exists(os.path.join(save_pth, folder_name)):
                        os.makedirs(os.path.join(save_pth, folder_name))
                    num += 1
                    img = nib.Nifti1Image(img_f, np.eye(4))
                    nib.save(img, os.path.join(save_pth, folder_name, namefile))
                    print("filenam", folder_name)
       ############## except:
           # print("*****5555555*****", name)  # k += 1


def generate_MRIFile(pth, namefile):
    imgs_aug = []
    age = helper.get_label(namefile)
    if age == -1:
        print("Namefile", namefile)
    if not "aug_data" in namefile:
        pth = os.path.join(pth, namefile)
        MRI_files = helper.get_list_files(pth)
        imgs_aug.append(generate(MRI_files, pth))
        return imgs_aug, age


def generate(MRI_files, pth):
    imgs_aug = []
    for MRIfile in MRI_files:
        subDir = pth + '/' + MRIfile
        imgs1 = generate_data(helper.load_MRI_data(subDir))
        imgs = list(itertools.chain(imgs1))
        imgs_aug.append(imgs)
        print("******", len(imgs))
    return imgs_aug

generate_dataset('Grey')
generate_dataset('White')

