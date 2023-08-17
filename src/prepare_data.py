import sys
import os
import scipy
import pandas as pd
import numpy as np
import random
import seaborn as sns
import tifffile as tiff
import matplotlib
from matplotlib import pyplot as plt

from PIL import Image
from skimage.io import imread, imsave
from skimage.util import view_as_windows
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb


from pathlib import Path
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import shutil
import rasterio
from sklearn.model_selection import train_test_split
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

from src.config import *
from src.utils import resize_image

seed_value = 42

# Set the random seed for Python's built-in random module
random.seed(seed_value)

# Set the random seed for numpy
np.random.seed(seed_value)



def convert_12bit_to_8bit(image_12bit):
    # Scale the pixel values from the range [0, 4095] to [0, 255]
    image_8bit = (image_12bit.astype(float) / 4095.0) * 255.0
    # Convert the pixel values to 8-bit integers
    image_8bit = image_8bit.astype(np.uint8)
    return image_8bit


def convert_16bit_to_8bit(image_16bit):
    # Scale the pixel values from the range [0, 65535] to [0, 255]
    image_8bit = (image_16bit.astype(float) / 65535.0 ) * 255.0
    # Convert the pixel values to 8-bit integers
    image_8bit = image_8bit.astype(np.uint8)
    return image_8bit


def extract_patches(image, patch_size, stride):
    patches = view_as_windows(image, patch_size, step=stride)
    patches = np.squeeze(patches)
    return patches  


def prepare_SPARCS(sparcs_path:Path=sparcs_path, patch_size:int=256, overlap:int=64, valid_test_size:float=0.2, test_size:float=0.5) -> None:
    # Create folders
    s = ['train', 'valid', 'test']
    splits_dir = [Path(sparcs_path, split) for split in s]
    for folder in splits_dir:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        
    # Load product names
    products = sorted(os.listdir(sparcs_raw_dir))
    prods = [(f.split('_')[0]+'_'+f.split('_')[1]) for f in products if '_' in f]
    bands = [f for f in products if '_data.tif' in f]
    labels = [f for f in products if '_labels.tif' in f]
    prods = [p for p in prods if any(p in label for label in labels) and any(p in d for d in bands)]
    prods = np.unique(prods)
    
    train_products, test_products = train_test_split(prods, test_size=valid_test_size, random_state=seed_value)
    valid_products, test_products = train_test_split(test_products, test_size=test_size, random_state=seed_value)
    
    splits = [train_products, valid_products, test_products]
    
    for idx, split in enumerate(splits):
        im_folder = Path(splits_dir[idx], "images/")
        masks_folder = Path(splits_dir[idx], "masks/")
        im_p_folder = Path(splits_dir[idx], "images_p/")
        masks_p_folder = Path(splits_dir[idx], "masks_p/")
        folders = [im_folder, masks_folder, im_p_folder, masks_p_folder]
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)

        # patch_size = 240  # Size of each patch
        # overlap = 80  # Overlap size
        # stride = 160

        # patch_size = patch_size
        # Overlap size = 64
        stride = patch_size - overlap

        # Save images and masks in 8bit
        for p in tqdm(split, desc=s[idx]):
            # Images
            im = tiff.imread(Path(sparcs_raw_dir, f"{p}_data.tif"))
            im = convert_16bit_to_8bit(im)

            # if rgb:
                # im = np.flip(im[:,:,1:4], axis=2)
                # im = im[:,:,1:4]

                # im_patches = extract_patches(im, (patch_size, patch_size, 3), stride)
            # else:
            im = im[:,:,1:5]
            im_patches = extract_patches(im, (patch_size, patch_size, 4), stride)


            for x in range(im_patches.shape[0]):
                for y in range(im_patches.shape[1]):
                    i = x * im_patches.shape[1] + y
                    p_out_path = Path(im_p_folder, f"{p}_{i:02d}.png")
                    imsave(p_out_path, im_patches[x,y,:,:,:])

            # Masks
            mask = tiff.imread(Path(sparcs_raw_dir, f"{p}_labels.tif"))
            # tiff.imwrite(Path(masks_folder, f"{p}.tif"), mask)

            mask_patches = extract_patches(mask, (patch_size, patch_size), stride)
            for x in range(mask_patches.shape[0]):
                for y in range(mask_patches.shape[1]):
                    # Create the file name using the format "p_xy.png"
                    i = x * mask_patches.shape[1] + y
                    m_out_path = Path(masks_p_folder, f"{p}_{i:02d}.tif")
                    tiff.imwrite(m_out_path, mask_patches[x,y,:,:])
                    
            # Saves test full images for visual scoring
            if idx == 2:
                imsave(Path(im_folder, f"{p}.png"), im)
                tiff.imwrite(Path(masks_folder, f"{p}.tif"), mask)

            # imsave(Path(im_folder, f"{p}.png"), im)
            # tiff.imwrite(Path(masks_folder, f"{p}.tif"), mask)



#####################################################################


def prepare_S2(path: Path = s2_path, valid_test_size:float=0.2, test_size:float=0.5) -> None:
    s = ['train', 'valid', 'test']
    splits_dir = [Path(path, split) for split in s]
    for folder in splits_dir:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        
    im_dir = s2_im_path
    m_dir = s2_labels_path
        
    my_list = sorted(os.listdir(s2_im_path))
    my_list = [name for name in my_list if name.startswith('ref_')]
    products = list(map(lambda string: string.split('_')[-1], my_list))
        
    # Split into train, test, and validation sets
    train_products, test_products = train_test_split(products, test_size=valid_test_size, random_state=seed_value)
    valid_products, test_products = train_test_split(test_products, test_size=test_size, random_state=seed_value)
    
    splits = [train_products, valid_products, test_products]
    
    for idx, split in enumerate(splits):
        im_folder = Path(splits_dir[idx], "images/")
        masks_folder = Path(splits_dir[idx], "masks/")
        # im_p_folder = Path(splits_dir[idx], "images_p/")
        # masks_p_folder = Path(splits_dir[idx], "masks_p/")
        folders = [im_folder, masks_folder]
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)
        
        for p in tqdm(split, desc=s[idx]):
            ID = [string for string in sorted(os.listdir(im_dir)) if p in string][0]
            image_path = Path(im_dir, ID)
            B2 = tiff.imread(image_path / 'B02.tif')
            B3 = tiff.imread(image_path / 'B03.tif')
            B4 = tiff.imread(image_path / 'B04.tif')
            B8 = tiff.imread(image_path / 'B08.tif')
            im = np.dstack((B2, B3, B4, B8))
            im= np.clip(im/10000, 0, 1)

            ID = [string for string in sorted(os.listdir(m_dir)) if p in string][0]
            mask = tiff.imread(Path(m_dir, f'{ID}/labels.tif'))
            # mask = tiff.imread(Path(m_dir, f'{ID}/raster_labels.tif'))
            mask = np.expand_dims(mask, axis=-1)
            
            tiff.imwrite(Path(im_folder, f"{p}.tif"), im)
            tiff.imwrite(Path(masks_folder, f"{p}.tif"), mask)

            
#####################################################################

    
# def patch_image(img, patch_size, overlap):
#     """
#     Split up an image into smaller overlapping patches
#     """
#     # TODO: Get the size of the padding right.
#     # Add zeropadding around the image (has to match the overlap)
#     img_shape = np.shape(img)
#     img_padded = np.zeros((img_shape[0] + 2*patch_size, img_shape[1] + 2*patch_size, img_shape[2]))
#     img_padded[overlap:overlap + img_shape[0], overlap:overlap + img_shape[1], :] = img

#     # Find number of patches
#     n_width = int((np.size(img_padded, axis=0) - patch_size) / (patch_size - overlap))
#     n_height = int((np.size(img_padded, axis=1) - patch_size) / (patch_size - overlap))

#     # Now cut into patches
#     n_bands = np.size(img_padded, axis=2)
#     img_patched = np.zeros((n_height * n_width, patch_size, patch_size, n_bands), dtype=img.dtype)
#     for i in range(0, n_width):
#         for j in range(0, n_height):
#             id = n_height * i + j

#             # Define "pixel coordinates" of the patches in the whole image
#             xmin = patch_size * i - i * overlap
#             xmax = patch_size * i + patch_size - i * overlap
#             ymin = patch_size * j - j * overlap
#             ymax = patch_size * j + patch_size - j * overlap

#             # Cut out the patches.
#             # img_patched[id, width , height, depth]
#             img_patched[id, :, :, :] = img_padded[xmin:xmax, ymin:ymax, :]

#     return img_patched, n_height, n_width 



def patch_image(img, patch_size, overlap):
    """
    Split up an image into smaller overlapping patches
    """
    if overlap >= patch_size:
        raise ValueError("Overlap must be less than patch size")

    img_shape = img.shape
    # if len(img_shape) != 3 or img_shape[2] != 4:
    #     raise ValueError("Input image must be 3D with 4 channels")

    stride = patch_size - overlap
    
    # Calculate padding sizes to make the image dimensions multiples of the stride
    pad_height = (stride - (img_shape[0] % stride)) % stride
    pad_width = (stride - (img_shape[1] % stride)) % stride

    # Add padding around the image
    img_padded = np.pad(img, [(overlap, pad_height + overlap), (overlap, pad_width + overlap), (0,0)], mode='constant')

    # Calculate the number of patches
    n_height = (img_padded.shape[0] - patch_size) // stride + 1
    n_width = (img_padded.shape[1] - patch_size) // stride + 1

    # Initialize the output array
    n_bands = img_shape[2]
    img_patched = np.zeros((n_height * n_width, patch_size, patch_size, n_bands), dtype=img.dtype)

    # Cut into patches
    patch_id = 0
    for i in range(n_height):
        for j in range(n_width):
            # Calculate pixel coordinates of the patch in the padded image
            ymin = i * stride
            ymax = ymin + patch_size
            xmin = j * stride
            xmax = xmin + patch_size

            # Extract the patch
            img_patched[patch_id, :, :, :] = img_padded[ymin:ymax, xmin:xmax, :]
            patch_id += 1

    return img_patched, n_height, n_width



def read_and_close(file):
    with rasterio.open(file) as src:
        return np.expand_dims(src.read(1), axis=-1)


biomes = ['Barren', 'Forest', 'GrassCrops', 'Shrubland', 'SnowIce', 'Urban', 'Water', 'Wetlands']


def prepare_biome8(biome_path: Path=biome_path, patch_size:int=512, overlap:int=128, valid_test_size:float=0.2, test_size:float=0.5) -> None:
    """
    Prepare the Biome8 dataset for training, validation and testing.

    """
    # Create train, validation and test directories
    splits = ["train", "valid", "test"]
    splits_dir = {split: biome_path / split for split in splits}

    # Delete existing directories
    for split, dir in splits_dir.items():
        if dir.exists():
            shutil.rmtree(dir)

    # Recreate empty directories
    for split, dir in splits_dir.items():
        dir.mkdir(exist_ok=True, parents=True)
    
    # Process each biome independently to ensure equal representation
    for biome in biomes:
        products_dir = biome_raw_dir / biome / 'BC' 
        products = sorted(os.listdir(products_dir))

        # Split into train, test, and validation sets
        train_products, test_products = train_test_split(products, test_size=valid_test_size, random_state=seed_value)
        valid_products, test_products = train_test_split(test_products, test_size=test_size, random_state=seed_value)
    
        split_products = {
            "train": train_products,
            "valid": valid_products,
            "test": test_products
        }

        for split, products in split_products.items():
            im_folder = Path(splits_dir[split], "images/")
            masks_folder = Path(splits_dir[split], "masks/")
            im_p_folder = Path(splits_dir[split], "images_p/")
            masks_p_folder = Path(splits_dir[split], "masks_p/")
            
            folders = [im_folder, masks_folder, im_p_folder, masks_p_folder]
            for dir in folders:
                dir.mkdir(exist_ok=True, parents=True)
                
            for ID in products:
                # B1 = tiff.imread(Path(biome_raw_dir, '{}/{}_B1.TIF'.format(ID, ID)))
                B2 = tiff.imread(Path(products_dir, '{}/{}_B2.TIF'.format(ID, ID)))
                B3 = tiff.imread(Path(products_dir, '{}/{}_B3.TIF'.format(ID, ID)))
                B4 = tiff.imread(Path(products_dir, '{}/{}_B4.TIF'.format(ID, ID)))
                B5 = tiff.imread(Path(products_dir, '{}/{}_B5.TIF'.format(ID, ID)))
                # # B6 = tiff.imread(Path(biome_raw_dir, '{}/{}_B6.TIF'.format(ID, ID)))
                # B7 = tiff.imread(Path(biome_raw_dir, '{}/{}_B7.TIF'.format(ID, ID)))
                # B8 = tiff.imread(Path(biome_raw_dir, '{}/{}_B8.TIF'.format(ID, ID))) Removed since it has a different pixel resolution
                # B9 = tiff.imread(Path(biome_raw_dir, '{}/{}_B9.TIF'.format(ID, ID)))
                # B10 = tiff.imread(Path(biome_raw_dir, '{}/{}_B10.TIF'.format(ID, ID)))
                # B11 = tiff.imread(Path(biome_raw_dir, '{}/{}_B11.TIF'.format(ID, ID)))
                # BQA = tiff.imread(Path(biome_raw_dir, '{}/{}_BQA.TIF'.format(ID, ID))) Quality mask
                # im = np.dstack((B1, B2, B3, B4, B5, B6, B7, B9, B10, B11))
                im = np.dstack((B2, B3, B4, B5))
                im = convert_16bit_to_8bit(im)
                
                mask = read_and_close(Path(products_dir, '{}/{}_fixedmask.img'.format(ID, ID)))
                
                mask[mask == 0] = 2  # none
                mask[mask == 128] = 0  # Background
                mask[np.logical_or(mask == 192, mask == 255)] = 1  # thin cloud, cloud
                mask[mask == 64] = 0  # Set cloud shadow as background
                
                im = resize_image(im, (4096,4096,4))
                mask = resize_image(mask, (4096,4096,1))
                
                if split == "test":
                    # Save the full size test images for visual scoring
                    tiff.imwrite(Path(im_folder, f"{ID}.tif"), im)
                    tiff.imwrite(Path(masks_folder, f"{ID}.tif"), mask)
                    
                # patch_size = patch_size  # Size of each patch
                # overlap = 64  # Overlap size
                
                x_patched, _, _ = patch_image(im, patch_size, overlap=overlap)
                y_patched, _, _ = patch_image(mask, patch_size, overlap=overlap)
                        
                for i, patch in enumerate(y_patched):
                    if np.all(patch != 2):  # Check if all pixels in the patch are non-black
                        p_out_path = Path(im_p_folder, f"{ID}_{i:02d}.tif")
                        tiff.imwrite(p_out_path, x_patched[i,:,:,:])
                        m_out_path = Path(masks_p_folder, f"{ID}_{i:02d}.tif")
                        tiff.imwrite(m_out_path, y_patched[i,:,:,:])
                    else:
                        # Skip saving the patch
                        continue
