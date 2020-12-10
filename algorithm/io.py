
import nrrd
import nibabel as nib
import numpy as np
import pandas as pd

from algorithm.config import *


def load_img_dataset():
    img_list = os.listdir(DATA_IMG_EXTENSION_PATH)
    try:
        img_list.remove('.DS_Store')
    except:
        print('Linux OS')
    print('img file count: ', len(img_list))

    pid_list = list(map(lambda x: x.split('-')[0], img_list))
    df_img = pd.DataFrame({'pid': pid_list, 'file': img_list})
    df_img = df_img.groupby(by=['pid'], axis=0, as_index=False).agg(['count', lambda x: ', '.join(x)])

    df_img_valid = df_img[df_img['file']['count'] == 4]
    df_img_invalid = df_img[df_img['file']['count'] != 4]

    print('pid with 4 files: ', len(df_img_valid))
    # df_img_invalid.to_csv('./error_img.csv', index=True)

    return {'valid': df_img_valid, 'invalid': df_img_invalid}


def load_trg_label_dataset():
    df_label = pd.read_excel(TRG_LABEL_PATH)

    # filter invalid rows
    df_label = df_label[df_label['wrong'].isna()]
    df_label = df_label[-df_label['TRG'].isna()]
    df_label = df_label[df_label['verify'] == 4]
    df_label = df_label[['pid', 'TRG']]

    # sort
    df_label = df_label.sort_values(by=['pid'], axis=0)

    return df_label


def load_os_label_dataset():
    df_label = pd.read_excel(SURVIVAL_LABEL_PATH)
    df_label = df_label.dropna()

    # convert to 0/1
    df_label['label'] = (df_label['survive'] == '是').astype(int)
    df_label = df_label.drop(['survive'], axis=1)

    # sort
    df_label = df_label.sort_values(by=['pid'], axis=0)

    return df_label


def load_ct_data(ct_id):
    # load .nii image
    try:
        path = os.path.join(DATA_IMG_EXTENSION_PATH, ct_id + '.nii')
        
        nib_object = nib.load(path)
        image_data = nib_object.get_data()

        # https://nipy.org/nibabel/coordinate_systems.html
        scale_x = abs(nib_object.affine[0,0])
        scale_y = abs(nib_object.affine[1,1])
        assert(scale_x == scale_y)
        scale_xy = scale_x

        # permute axis and normalize ct scan
        image_data = image_data.transpose(2, 0, 1)
        image_data = normalize_ct(image_data)
        return image_data, scale_xy
    except:
        pass

    # load .nrrd image
    try:
        path = os.path.join(DATA_IMG_EXTENSION_PATH, ct_id + '.nrrd')
        
        nrrd_object = nrrd.read(path)
        image_data = np.array(nrrd_object[0])

        spacing = nrrd_object[1]['space directions']
        scale_x = abs(spacing[0,0])
        scale_y = abs(spacing[1,1])
        assert(scale_x == scale_y)
        scale_xy = scale_x
        
        print('Load through nrrd image by nrrd: ' + ct_id)
        return image_data, scale_xy
    except:
        pass

    raise ValueError({
        'message': ERROR_IMAGE_OPEN,
        'data': ct_id
    })


def normalize_ct(image_data, normalize=False):
    # set out of scan range from -3024 to -1024
    image_data[image_data == -3024] = -1024
    if normalize:
        image_data = image_data / 1024 + 1
    return image_data

