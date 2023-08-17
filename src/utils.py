import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.io import imread, imsave

import numpy as np
import os
import random
from pathlib import Path
from keras.utils import to_categorical
from keras import backend as K

from skimage.transform import resize

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score


from tensorflow import keras
import numpy as np
from PIL import Image

from tqdm import tqdm
# from tqdm.notebook import tqdm

from src.config import *

# importing types hint
from typing import List, Tuple, Literal

seed_value = 42

# Set the random seed for Python's built-in random module
random.seed(seed_value)

# Set the random seed for numpy
np.random.seed(seed_value)


###########################################################


def plot_image_from_bands_array(array:np.array, title:str="", vmin:int=0, vmax:int=1):
    im_bgr = array[:,:,0:3]
    im_rgb = im_bgr[:,:,::-1]
    plt.imshow(im_rgb, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.show()


def rgb_image_from_bands_array(array:np.array, title:str="", vmin:int=0, vmax:int=1):
    im_bgr = array[:,:,0:3]
    im_rgb = im_bgr[:,:,::-1]
    return im_rgb


def resize_image(image, new_dim=(1024, 1024, 3)):
    resized_image = resize(image, new_dim, mode='edge', preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None, order=0)
    return resized_image


###########################################################


def get_SPARCS(sets: Literal['train', 'valid', 'test'] = 'train', full_image: bool = False, only_rgb: bool = False, binary: bool = True, resize_to:int=None) -> Tuple[np.ndarray, np.ndarray]:
    masks = []
    ims = []
    
    if sets == 'train':
        path = sparcs_train_dir
    elif sets == 'valid':
        path = sparcs_valid_dir
    elif sets == 'test':
        path = sparcs_test_dir
    
    # sparcs_im_dir = Path(path, "images/")
    # sparcs_masks_dir = Path(path, "masks/")
    # sparcs_im_p_dir = Path(path, "images_p/")
    # sparcs_masks_p_dir = Path(path, "masks_p/")
    
    im_dir = Path(path, "images/") if full_image else Path(path, "images_p/")
    m_dir = Path(path, "masks/") if full_image else Path(path, "masks_p/")
    im_names = sorted(os.listdir(im_dir))
    m_names = sorted(os.listdir(m_dir))
    
    # if not im_names:
    #     im_names = sorted(os.listdir(im_dir))
    # if not m_names:
    #     m_names = sorted(os.listdir(m_dir))
    
    for ID in im_names:
        im = imread(Path(im_dir, ID))
        if only_rgb:
            im = im[:, :, :3]
            im = np.flip(im, axis=2)
        ims.append(im)
    
    for ID in m_names:
        mask = tiff.imread(Path(m_dir, ID))
        if binary:
            mask = np.where(mask != 5, 0, mask)
            mask = np.expand_dims(mask, axis=-1)
            masks.append(np.clip(mask, 0, 1))
        else:
            mask = to_categorical(mask, num_classes=7)
            masks.append(mask)
    
    ims = np.array(ims) / 255.0
    masks = np.array(masks)
    
    # Shuffle the arrays
    # np.random.shuffle(ims)
    # np.random.shuffle(masks)
    
    if resize_to:
        new_size = resize_to
        ims = resize_image(ims, new_dim=((ims.shape[0], new_size, new_size, ims.shape[-1])))
        masks = resize_image(masks, new_dim=((masks.shape[0], new_size, new_size, masks.shape[-1])))
    
    return ims, masks


def get_SPARCS_generator(sets: Literal['train', 'valid', 'test'] = 'train', full_image: bool = False, only_rgb: bool = False, binary: bool = True, batch_size: int = 32, shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    if sets == 'train':
        path = sparcs_train_dir
    elif sets == 'valid':
        path = sparcs_valid_dir
    elif sets == 'test':
        path = sparcs_test_dir
    
    im_dir = Path(path, "images/") if full_image else Path(path, "images_p/")
    m_dir = Path(path, "masks/") if full_image else Path(path, "masks_p/")
    
    im_names = sorted(os.listdir(im_dir))
    m_names = sorted(os.listdir(m_dir))
    
    if len(im_names) != len(m_names):
        print(f"Warning: number of images ({len(im_names)}) does not match number of masks ({len(m_names)}).")
        if len(im_names) > len(m_names):
            diff = set(im_names) - set(m_names)
            print("Images without corresponding masks:", diff)
        else:
            diff = set(m_names) - set(im_names)
            print("Masks without corresponding images:", diff)

    while True:
        # Shuffle indices if shuffle is true
        if shuffle:
            indices = np.arange(len(im_names))
            np.random.shuffle(indices)
            im_names = [im_names[i] for i in indices]
            m_names = [m_names[i] for i in indices]
        
        for i in range(0, len(im_names), batch_size):
            batch_im = []
            batch_mask = []
            for im_name, m_name in zip(im_names[i:min(i+batch_size,len(im_names))], m_names[i:min(i+batch_size,len(m_names))]):
                try:
                    im = imread(Path(im_dir, im_name))
                    if only_rgb:
                        im = im[:, :, :3]
                        im = np.flip(im, axis=2)
                    im = im / 255.0

                    mask = tiff.imread(Path(m_dir, m_name))
                    if binary:
                        mask = np.where(mask != 5, 0, mask)
                        mask = np.expand_dims(mask, axis=-1)
                        mask = np.clip(mask, 0, 1)
                    else:
                        mask = to_categorical(mask, num_classes=7)

                    batch_im.append(im)
                    batch_mask.append(mask)
                except Exception as e:
                    print(f"Error processing files {im_name} or {m_name}: {str(e)}")
                    continue
            
            if batch_im:  # Only yield a batch if it has data
                yield np.array(batch_im, dtype=np.float32), np.array(batch_mask, dtype=np.uint8 if binary else np.int32)




###########################################################


def get_S2(sets: Literal['train', 'valid', 'test'] = 'train', only_rgb: bool = False, resize_to:int=None) -> Tuple[np.ndarray, np.ndarray]:
    masks = []
    ims = []
    
    if sets == 'train':
        path = s2_train_dir
    elif sets == 'valid':
        path = s2_valid_dir
    elif sets == 'test':
        path = s2_test_dir
    
    im_dir = Path(path, "images/") 
    m_dir = Path(path, "masks/")
    
    im_names = sorted(os.listdir(im_dir))
    m_names = sorted(os.listdir(m_dir))
    
    # if not im_names:
    #     im_names = sorted(os.listdir(im_dir))
    # if not m_names:
    #     m_names = sorted(os.listdir(m_dir))
    
    for ID in im_names:
        im = imread(Path(im_dir, ID))
        if only_rgb:
            im = im[:, :, :3]
            im = np.flip(im, axis=2)
        ims.append(im)
    
    for ID in m_names:
        mask = tiff.imread(Path(m_dir, ID))
        masks.append(np.clip(mask, 0, 1))
        
    ims = np.array(ims)
    masks = np.array(masks)
    
    if resize_to:
        new_size = resize_to
        ims = resize_image(ims, new_dim=((ims.shape[0], new_size, new_size, ims.shape[-1])))
        masks = resize_image(masks, new_dim=((masks.shape[0], new_size, new_size, masks.shape[-1])))
    
    return ims, masks


def get_S2_generator(sets: Literal['train', 'valid', 'test'] = 'train', only_rgb: bool = False, batch_size: int = 32, shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    if sets == 'train':
        path = s2_train_dir
    elif sets == 'valid':
        path = s2_valid_dir
    elif sets == 'test':
        path = s2_test_dir
    
    im_dir = Path(path, "images/")
    m_dir = Path(path, "masks/")
    
    im_names = sorted(os.listdir(im_dir))
    m_names = sorted(os.listdir(m_dir))
    
    while True:
        # Shuffle indices if shuffle is true
        if shuffle:
            indices = np.arange(len(im_names))
            np.random.shuffle(indices)
            im_names = [im_names[i] for i in indices]
            m_names = [m_names[i] for i in indices]
        
        for i in range(0, len(im_names), batch_size):
            batch_im = []
            batch_mask = []
            for im_name, m_name in zip(im_names[i:i+batch_size], m_names[i:i+batch_size]):
                im = imread(Path(im_dir, im_name))
                if only_rgb:
                    im = im[:, :, :3]
                    im = np.flip(im, axis=2)
                im = im / 255.0

                mask = tiff.imread(Path(m_dir, m_name))
                mask = np.clip(mask, 0, 1)

                batch_im.append(im)
                batch_mask.append(mask)
            
            yield np.array(batch_im), np.array(batch_mask)


###########################################################


def get_biome8(sets: Literal['train', 'valid', 'test'] = 'train', full_image: bool = False, only_rgb: bool = False, resize_to:int=None) -> Tuple[np.ndarray, np.ndarray]:
    set_paths = {'train': biome_train_dir, 'valid': biome_valid_dir, 'test': biome_test_dir}
    
    if sets not in set_paths:
        raise ValueError(f'Invalid set name: {sets}. Expected one of: {" ".join(set_paths.keys())}')
    
    path = set_paths[sets]
    im_dir = Path(path, "images/") if full_image else Path(path, "images_p/")
    m_dir = Path(path, "masks/") if full_image else Path(path, "masks_p/")
    
    im_names = sorted(os.listdir(im_dir))
    m_names = sorted(os.listdir(m_dir))
    
    ims = []
    masks = []

    for im_name, m_name in tqdm(zip(im_names, m_names), total=len(im_names), desc="Loading data"):
        im = imread(Path(im_dir, im_name))
        if only_rgb:
            im = im[:, :, :3]
            im = np.flip(im, axis=2)
            
        mask = tiff.imread(Path(m_dir, m_name))

        # if full_image:
        #     im = resize_image(im, new_dim=((1024, 1024, 3))) if only_rgb else resize_image(im, new_dim=((1024, 1024, 4)))
        #     mask = resize_image(mask, new_dim=((1024, 1024, 1)))

        ims.append(im / 255.0)
        masks.append(mask)
        
    ims = np.array(ims)
    masks = np.array(masks)
    
    if resize_to:
        new_size = resize_to
        ims = resize_image(ims, new_dim=((ims.shape[0], new_size, new_size, ims.shape[-1])))
        masks = resize_image(masks, new_dim=((masks.shape[0], new_size, new_size, masks.shape[-1])))

    return ims, masks
        
        
def get_biome8_generator(sets: Literal['train', 'valid', 'test'] = 'train', full_image: bool = False, only_rgb: bool = False, batch_size: int = 32, shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    set_paths = {'train': biome_train_dir, 'valid': biome_valid_dir, 'test': biome_test_dir}
    
    if sets not in set_paths:
        raise ValueError(f'Invalid set name: {sets}. Expected one of: {" ".join(set_paths.keys())}')
    
    path = set_paths[sets]
    im_dir = Path(path, "images/") if full_image else Path(path, "images_p/")
    m_dir = Path(path, "masks/") if full_image else Path(path, "masks_p/")
    
    im_names = sorted(os.listdir(im_dir))
    m_names = sorted(os.listdir(m_dir))
    
    if len(im_names) != len(m_names):
        print(f"Warning: number of images ({len(im_names)}) does not match number of masks ({len(m_names)}).")
        if len(im_names) > len(m_names):
            diff = set(im_names) - set(m_names)
            print("Images without corresponding masks:", diff)
        else:
            diff = set(m_names) - set(im_names)
            print("Masks without corresponding images:", diff)
    
    while True:
        # Shuffle indices if shuffle is true
        if shuffle:
            indices = np.arange(len(im_names))
            np.random.shuffle(indices)
            im_names = [im_names[i] for i in indices]
            m_names = [m_names[i] for i in indices]
        
        for i in range(0, len(im_names), batch_size):
            batch_im = []
            batch_mask = []
            for im_name, m_name in zip(im_names[i:min(i+batch_size,len(im_names))], m_names[i:min(i+batch_size,len(m_names))]):
                try:
                    im = imread(Path(im_dir, im_name))
                    if only_rgb:
                        im = im[:, :, :3]
                        im = np.flip(im, axis=2)
                    im = im / 255.0

                    mask = tiff.imread(Path(m_dir, m_name))
                    mask = np.clip(mask, 0, 1)
                    
                    # if full_image:
                    #     im = resize_image(im, new_dim=((1024, 1024, 3))) if only_rgb else resize_image(im, new_dim=((1024, 1024, 4)))
                    #     mask = resize_image(mask, new_dim=((1024, 1024, 1)))

                    batch_im.append(im)
                    batch_mask.append(mask)
                except Exception as e:
                    print(f"Error processing files {im_name} or {m_name}: {str(e)}")
                    continue
            if batch_im:  # Only yield a batch if it has data
                yield np.array(batch_im), np.array(batch_mask)


####################################################################################################################


class Cloud95Dataset(keras.utils.Sequence):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir):
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        
        files = {'red': r_file, 
                 'green':g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}
        
        
        return files
                                       
    def __len__(self):
        
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False, false_color_aug=False, invert_rgb=True):

        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ], axis=2)
        
        # raw_bgr = np.stack([np.array(Image.open(self.files[idx]['blue'])), 
        #                     np.array(Image.open(self.files[idx]['green'])), 
        #                     np.array(Image.open(self.files[idx]['red'])),], axis=2)
        
        if (false_color_aug):
            indexes = np.arange(3)
            np.random.shuffle(indexes)
            raw_rgb = np.stack([raw_rgb[..., indexes[0]],
                                raw_rgb[..., indexes[1]],
                                raw_rgb[..., indexes[2]],
                               ], axis=2)
            
        if invert_rgb:
            raw_rgb = raw_rgb[:, :, ::-1]
        
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
            
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
    
        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, -1) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        
        x = self.open_as_array(idx, invert=False, include_nir=True, false_color_aug=False, invert_rgb=True).astype(np.float32)
        y = self.open_mask(idx, add_dims=True).astype(np.int32)
        
        return x, y
    
    def open_as_pil(self, idx):
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s