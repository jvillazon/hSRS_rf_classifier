"""
HSI Unlabeled Dataset — loads and preprocesses hyperspectral .tif image stacks for inference.

No PyTorch dependency; this module uses only numpy and tifffile.
"""

import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm


class HSI_Unlabeled_Dataset:
    """
    Dataset for loading and normalizing experimental HSI .tif images.

    Parameters
    ----------
    img_dir : str
        Directory containing .tif image stacks (channels × height × width).
    ch_start : int
        Channel index marking the end of the spectral silent region (~2800 cm⁻¹).
    transform : callable, optional
        Optional transform applied to spectra (not used in standard inference).
    image_normalization : bool
        If True, normalize each image using its global mean/std statistics.
    min_max_normalization : bool
        If True, use per-image min/max normalization instead.
    wavenumber_start : int
        Starting wavenumber of the spectral range.
    wavenumber_end : int
        Ending wavenumber of the spectral range.
    num_samples : int
        Number of spectral channels.
    compute_stats : bool
        If True, scan images to compute normalization statistics (required for
        RF classification). Set to False for fast initialization when only
        loading pre-existing results.
    """

    def __init__(self, img_dir, ch_start, transform=None,
                 image_normalization=False, min_max_normalization=False,
                 wavenumber_start=2700, wavenumber_end=3100,
                 num_samples=61, compute_stats=True):

        self.wavenumber_start = wavenumber_start
        self.wavenumber_end   = wavenumber_end
        self.num_samples      = num_samples
        self.compute_stats    = compute_stats
        self.ch_start         = ch_start
        self.image_normalization    = image_normalization
        self.min_max_normalization  = min_max_normalization
        self.transform        = transform

        # Discover .tif files, deduplicating across case variants
        discovered = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            discovered.extend(glob.glob(os.path.join(img_dir, ext)))
        unique = {os.path.normcase(os.path.normpath(p)): p for p in discovered}
        self.img_list = sorted(unique.values())

        self.image_stats = {}
        self.img_size    = [0]
        current_size     = 0

        print("\nScanning images...")
        for img_path in tqdm(self.img_list):
            image = tifffile.memmap(img_path, mode='r')
            with tifffile.TiffFile(img_path) as tif:
                if tif.is_imagej:
                    page = tif.pages[0]
                    pixel_size_x = 1 / page.resolution[1]
                    pixel_size_y = 1 / page.resolution[0]
                else:
                    pixel_size_x = pixel_size_y = 1

            if len(image.shape) >= 2:
                height = image.shape[-2]
                width  = image.shape[-1]
                size   = height * width

                image_min = image_max = None

                if self.compute_stats:
                    n_pixels   = height * width
                    chunk_size = 1_000_000
                    sum_silent = sum_all = sum_sq_all = 0.0
                    image_flat = image.reshape(image.shape[0], -1)

                    print(f"  Computing statistics for {os.path.basename(img_path)}...")
                    for start_p in range(0, n_pixels, chunk_size):
                        end_p = min(start_p + chunk_size, n_pixels)
                        chunk = np.flip(
                            image_flat[:, start_p:end_p].T.astype(np.float32), axis=1
                        )
                        sum_silent  += np.sum(np.mean(chunk[:, :self.ch_start], axis=1))
                        sum_all     += np.sum(chunk)
                        sum_sq_all  += np.sum(chunk ** 2)

                    avg_silent  = sum_silent / n_pixels
                    global_mean = sum_all / (n_pixels * image.shape[0])
                    global_std  = np.sqrt(
                        sum_sq_all / (n_pixels * image.shape[0]) - global_mean ** 2
                    )

                    if self.min_max_normalization:
                        image_min = avg_silent
                        image_max = global_mean + 3 * global_std
                        self.image_normalization = False
                    elif self.image_normalization:
                        image_min = avg_silent
                        image_max = global_mean + 3 * global_std

                self.image_stats[img_path] = {
                    'image_min':    image_min,
                    'image_max':    image_max,
                    'pixel_size_x': pixel_size_x,
                    'pixel_size_y': pixel_size_y,
                    'height':       height,
                    'width':        width,
                    'start_idx':    current_size,
                }
            else:
                size = 0

            current_size += size
            self.img_size.append(current_size)

    # ------------------------------------------------------------------
    def __len__(self):
        return self.img_size[-1]

    # ------------------------------------------------------------------
    def load_and_process_image(self, img_path):
        """
        Load and normalize a single image for inference.

        Parameters
        ----------
        img_path : str
            Path to the .tif image stack.

        Returns
        -------
        image_spectra : np.ndarray, shape (n_pixels, n_channels)
        """
        image = tifffile.memmap(img_path, mode='r')
        image_spectra = np.flip(
            image.reshape(image.shape[0], -1).T.astype(np.float32), axis=1
        )

        if self.image_normalization or self.min_max_normalization:
            stats     = self.image_stats[img_path]
            image_min = stats['image_min']
            image_max = stats['image_max']
            silent_mean = np.mean(image_spectra[:, :self.ch_start], axis=1, keepdims=True)
            image_spectra -= silent_mean
            image_spectra /= (image_max - image_min + 1e-6)

        return image_spectra   # (n_pixels, n_wavenumbers)
