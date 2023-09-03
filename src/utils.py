# Matplotlib and Image Processing
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.io import imread, imsave

# Numerical and Data Handling
import numpy as np
import os
import random
from pathlib import Path
from keras.utils import to_categorical
from keras import backend as K
from tqdm import tqdm

# Image Transformation
from skimage.transform import resize

# Deep Learning Framework
import tensorflow.keras as keras
from PIL import Image

# Custom Configuration and Type Hints
from src.config import *
from typing import List, Tuple, Literal
import warnings
warnings.filterwarnings("ignore")


seed_value = 42

# Set the random seed for Python's built-in random module
random.seed(seed_value)

# Set the random seed for numpy
np.random.seed(seed_value)


###########################################################


def plot_image_from_bands_array(array:np.array, title:str="", vmin:int=0, vmax:int=1):
    '''
    Plot an image from a 4D array of shape (n, h, w, 4) or (n, h, w, 3)
    '''
    im_bgr = array[:,:,0:3]
    im_rgb = im_bgr[:,:,::-1]
    plt.imshow(im_rgb, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.show()


def rgb_image_from_bands_array(array:np.array, title:str="", vmin:int=0, vmax:int=1):
    '''
    Return an image from a 4D array of shape (n, h, w, 4) or (n, h, w, 3)
    '''
    im_bgr = array[:,:,0:3]
    im_rgb = im_bgr[:,:,::-1]
    return im_rgb


def resize_image(image, new_dim=(1024, 1024, 3)):
    '''
    Resize an image to the new_dim
    '''
    resized_image = resize(image, new_dim, mode='edge', preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None, order=0)
    return resized_image



# def augment_data(images, masks):
#     # Define your augmentation pipeline
#     augmentation_pipeline = A.Compose([
#         A.HorizontalFlip(p=0.5),                # Horizontal flips with a 50% probability
#         A.RandomRotate90(p=0.5),               # Randomly rotate images by 90 degrees (clockwise) with a 50% probability
#         A.RandomBrightnessContrast(p=0.2),     # Random brightness and contrast adjustments with a 20% probability
#         A.Blur(p=0.1),                         # Apply slight blurring with a 10% probability
#         A.RandomCrop(height=224, width=224),   # Randomly crop the image to a specified size
#         A.Resize(height=256, width=256),       # Resize the image to a specified size
#     ])

#     # Assuming 'image' is your RGBNIR image
#     # 'mask' is your segmentation mask (if applicable)

    
#     # Initialize empty lists to store augmented images and masks
#     augmented_images = []
#     augmented_masks = []

#     for image, mask in zip(images, masks):
#         # Make copies of the original image and mask to keep them unchanged
#         image_copy = image.copy()
#         mask_copy = mask.copy()

#         # Apply the augmentation pipeline to the image and mask (if available)
#         augmented = augmentation_pipeline(image=image, mask=mask)

#         # Retrieve the augmented image and mask
#         augmented_image = augmented['image']
#         augmented_mask = augmented['mask']

#         augmented_images.append(augmented_image)
#         augmented_masks.append(augmented_mask)

#     # Convert the augmented_images and augmented_masks lists back to NumPy arrays
#     augmented_images = np.array(augmented_images)
#     augmented_masks = np.array(augmented_masks)

#     # Concatenate the original images and augmented images along the first axis
#     concatenated_images = np.concatenate((images, augmented_images), axis=0)

#     # Concatenate the original masks and augmented masks along the first axis
#     concatenated_masks = np.concatenate((masks, augmented_masks), axis=0)

#     # Now 'concatenated_images' contains both the original and augmented images
#     # 'concatenated_masks' contains both the original and augmented masks
#     # print(concatenated_images.shape, concatenated_masks.shape)
    
#     return concatenated_images, concatenated_masks

###########################################################


def get_SPARCS(sets: Literal['train', 'valid', 'test'] = 'train', full_image: bool = False, only_rgb: bool = False, binary: bool = True, resize_to:int=None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load the SPARCS dataset from the specified set (train, valid, test).

    Parameters
    ----------
    sets : Literal['train', 'valid', 'test'], optional
        The set to load, by default 'train'
    full_image : bool, optional
        If True, the full image is loaded, otherwise only the part of the image containing the clouds is loaded, by default False
    only_rgb : bool, optional
        If True, only the RGB channels are loaded, otherwise the RGBNIR channels are loaded, by default False
    binary : bool, optional
        If True, the masks are binary, otherwise they are categorical, by default True
    resize_to : int, optional
        If specified, the images and masks are resized to the specified size, by default None

    Returns
    -------
    ims : np.ndarray
        Array of shape (n, h, w, 3) or (n, h, w, 4) containing the images
    masks : np.ndarray
        Array of shape (n, h, w, 1) or (n, h, w, 7) containing the masks
    '''
    
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
        mask = np.squeeze(mask)
        if binary:
            mask = np.where(mask != 5, 0, mask)
            mask = np.expand_dims(mask, axis=-1)
            masks.append(np.clip(mask, 0, 1))
        else:
            mask = to_categorical(mask, num_classes=7)
            masks.append(mask)

    # if augment:
    #     ims, masks = augment_data(ims, masks)
    
    ims = np.array(ims) / 255.0
    masks = np.array(masks)
    
    if full_image and not resize_to:
        resize_to = 1024
    
    # Shuffle the arrays
    # np.random.shuffle(ims)
    # np.random.shuffle(masks)
    
    # print(ims.shape, masks.shape)
    
    if resize_to:
        new_size = resize_to
        ims = resize_image(ims, new_dim=((ims.shape[0], new_size, new_size, ims.shape[-1])))
        masks = resize_image(masks, new_dim=((masks.shape[0], new_size, new_size, masks.shape[-1])))

    # print(ims.shape, masks.shape)
    
    return ims, masks


def get_SPARCS_generator(sets: Literal['train', 'valid', 'test'] = 'train', full_image: bool = False, only_rgb: bool = False, binary: bool = True, batch_size: int = 32, shuffle: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load the SPARCS dataset from the specified set (train, valid, test).

    Parameters
    ----------
    sets : Literal['train', 'valid', 'test'], optional
        The set to load, by default 'train'
    full_image : bool, optional
        If True, the full image is loaded, otherwise only the part of the image containing the clouds is loaded, by default False
    only_rgb : bool, optional
        If True, only the RGB channels are loaded, otherwise the RGBNIR channels are loaded, by default False
    binary : bool, optional
        If True, the masks are binary, otherwise they are categorical, by default True
    resize_to : int, optional
        If specified, the images and masks are resized to the specified size, by default None

    Returns
    -------
    ims : np.ndarray
        Array of shape (n, h, w, 3) or (n, h, w, 4) containing the images
    masks : np.ndarray
        Array of shape (n, h, w, 1) or (n, h, w, 7) containing the masks

    '''

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
                batch_im = np.array(batch_im, dtype=np.float32)
                batch_mask = np.array(batch_mask, dtype=np.uint8 if binary else np.int32)

                # if augment:
                #     batch_im, batch_mask = augment_data((batch_im*255).astype(np.uint8), batch_mask)

                batch_im = np.array(batch_im, dtype=np.float32)
                batch_mask = np.array(batch_mask, dtype=np.uint8 if binary else np.int32)

                yield batch_im, batch_mask




###########################################################


def get_S2(sets: Literal['train', 'valid', 'test'] = 'train', only_rgb: bool = False, resize_to:int=None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load the S2_mlhub dataset from the specified set (train, valid, test).

    Parameters
    ----------
    sets : Literal['train', 'valid', 'test'], optional
        The set to load, by default 'train'
    only_rgb : bool, optional
        If True, only the RGB channels are loaded, otherwise the RGBNIR channels are loaded, by default False
    resize_to : int, optional
        If specified, the images and masks are resized to the specified size, by default None

    Returns
    -------
    ims : np.ndarray
        Array of shape (n, h, w, 3) or (n, h, w, 4) containing the images
    masks : np.ndarray
        Array of shape (n, h, w, 1) or (n, h, w, 7) containing the masks
    '''

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
    '''
    Load the S2_mlhub dataset from the specified set (train, valid, test).

    Parameters
    ----------
    sets : Literal['train', 'valid', 'test'], optional
        The set to load, by default 'train'
    only_rgb : bool, optional
        If True, only the RGB channels are loaded, otherwise the RGBNIR channels are loaded, by default False
    resize_to : int, optional
        If specified, the images and masks are resized to the specified size, by default None

    Returns
    -------
    ims : np.ndarray
        Array of shape (n, h, w, 3) or (n, h, w, 4) containing the images
    masks : np.ndarray
        Array of shape (n, h, w, 1) or (n, h, w, 7) containing the masks
    '''

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
    '''
    Load the Biome8 dataset from the specified set (train, valid, test).

    Parameters
    ----------
    sets : Literal['train', 'valid', 'test'], optional
        The set to load, by default 'train'
    full_image : bool, optional
        If True, the full image is loaded, otherwise only the part of the image containing the clouds is loaded, by default False
    only_rgb : bool, optional
        If True, only the RGB channels are loaded, otherwise the RGBNIR channels are loaded, by default False
    resize_to : int, optional
        If specified, the images and masks are resized to the specified size, by default None

    Returns
    -------
    ims : np.ndarray    
        Array of shape (n, h, w, 3) or (n, h, w, 4) containing the images
    masks : np.ndarray
        Array of shape (n, h, w, 1) or (n, h, w, 7) containing the masks
    '''
    
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

        if full_image:
            im = resize_image(im, new_dim=((1024, 1024, 3))) if only_rgb else resize_image(im, new_dim=((1024, 1024, 4)))
            mask = resize_image(mask, new_dim=((1024, 1024, 1)))

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
    '''
    Load the Biome8 dataset from the specified set (train, valid, test).

    Parameters
    ----------
    sets : Literal['train', 'valid', 'test'], optional
        The set to load, by default 'train'
    full_image : bool, optional
        If True, the full image is loaded, otherwise only the part of the image containing the clouds is loaded, by default False
    only_rgb : bool, optional
        If True, only the RGB channels are loaded, otherwise the RGBNIR channels are loaded, by default False
    resize_to : int, optional
        If specified, the images and masks are resized to the specified size, by default None

    Returns
    -------
    ims : np.ndarray    
        Array of shape (n, h, w, 3) or (n, h, w, 4) containing the images
    masks : np.ndarray
        Array of shape (n, h, w, 1) or (n, h, w, 7) containing the masks
    '''

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
    '''
    Dataset class for the Cloud95 dataset

    Attributes
    ----------
    files : List[Path]
        List of paths to the files in the dataset

    Methods
    -------
    __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir)
        Initialize the dataset class
    __len__(self)
        Return the number of files in the dataset
    combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir)
        Combine the files in the different bands into a dictionary
    open_as_array(self, idx, invert=False, include_nir=False, false_color_aug=False, invert_rgb=True)
        Open the image and mask as numpy arrays
    open_mask(self, idx, add_dims=False)
        Open the mask as a numpy array
    __getitem__(self, idx)
        Return the image and mask as numpy arrays
    open_as_pil(self, idx)
        Open the image as a PIL image
    __repr__(self)
        Return a string representation of the dataset    
    '''
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir):
        '''
        Initialize the dataset class.

        Parameters
        ----------
        r_dir : Path
            Path to the red band folder
        g_dir : Path
            Path to the green band folder
        b_dir : Path
            Path to the blue band folder
        nir_dir : Path
            Path to the NIR band folder
        gt_dir : Path
            Path to the ground truth folder

        Returns
        -------
        None
        '''

        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        '''
        Combine the files in the different bands into a dictionary

        Parameters
        ----------
        r_file : Path
            Path to the red band file
        g_dir : Path
            Path to the green band folder
        b_dir : Path
            Path to the blue band folder
        nir_dir : Path
            Path to the NIR band folder
        gt_dir : Path
            Path to the ground truth folder

        Returns
        -------
        files : Dict[str, Path]
            Dictionary containing the paths to the files in the different bands
        '''
        
        files = {'red': r_file, 
                 'green':g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}
        
        
        return files
                                       
    def __len__(self):
        '''
        Return the number of files in the dataset
        '''
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False, false_color_aug=False, invert_rgb=True):
        '''
        Open the image and mask as numpy arrays

        Parameters
        ----------
        idx : int
            Index of the file to open
        invert : bool, optional
            If True, invert the image, by default False
        include_nir : bool, optional
            If True, include the NIR band, by default False
        false_color_aug : bool, optional
            If True, randomly shuffle the RGB bands, by default False
        invert_rgb : bool, optional
            If True, invert the RGB channels, by default True

        Returns
        ------- 
        raw_rgb : Numpy array of shape (h, w, 3) or (h, w, 4) containing the image
        '''

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
        '''
        Open the mask as a numpy array

        Parameters
        ----------
        idx : int
            Index of the file to open
        add_dims : bool, optional
            If True, add a dimension to the mask, by default False

        Returns
        -------
        raw_mask : Numpy array of shape (h, w, 1) or (h, w, 7) containing the mask
        '''
        
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, -1) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        '''
        Return the image and mask as numpy arrays

        Parameters
        ----------
        idx : int
            Index of the file to open

        Returns
        -------
        x : Numpy array of shape (h, w, 3) or (h, w, 4) containing the image
        y : Numpy array of shape (h, w, 1) or (h, w, 7) containing the mask
        '''
        
        x = self.open_as_array(idx, invert=False, include_nir=True, false_color_aug=False, invert_rgb=True).astype(np.float32)
        y = self.open_mask(idx, add_dims=True).astype(np.int32)
        
        return x, y
    
    def open_as_pil(self, idx):
        '''
        Open the image as a PIL image

        Parameters
        ----------
        idx : int
            Index of the file to open

        Returns
        -------
        Image : PIL image
        '''
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        '''
        Return a string representation of the dataset

        Returns
        -------
        s : str
        '''
        s = 'Dataset class with {} files'.format(self.__len__())

        return s