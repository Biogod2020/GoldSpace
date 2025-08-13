# spaglam_preproc/core/image_tiler.py

import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np
from PIL import Image

# openslide-python is an optional dependency
try:
    import openslide
except ImportError:
    openslide = None

# squidpy is an optional dependency for reading from AnnData


class ImageHandler:
    """
    A unified interface to handle various image sources for tile extraction.
    It can be initialized with an AnnData object, a file path (WSI or standard image),
    or a pre-loaded image object (PIL Image or NumPy array).
    """
    def __init__(self, source: Optional[Union[str, Path, object]], adata: Optional[object] = None):
        self.image_obj = None
        self.width, self.height = 0, 0
        self._load_image(source, adata)

    def _load_image(self, source, adata):
        """Internal method to load the image from the specified source."""
        # 1. Try to load from AnnData object if it's the primary source
        if source is None and adata is not None:
            spatial_key = list(adata.uns.get('spatial', {}).keys())
            if spatial_key:
                # Assuming standard squidpy storage format
                img_container = adata.uns['spatial'][spatial_key[0]]['images'].get('hires')
                self.image_obj = img_container
                self.width, self.height = self.image_obj.shape[1], self.image_obj.shape[0]
                logging.info(f"Loaded image '{spatial_key[0]}' from AnnData object.")
                return
            else:
                 raise ValueError("Image source is None and no spatial image found in adata.uns['spatial'].")
        
        if source is None:
            raise ValueError("No image source provided (path or AnnData).")

        # 2. Handle file paths (WSI or standard formats)
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found at: {path}")

            wsi_extensions = {".svs", ".tiff", ".tif", ".vms", ".vmu", ".ndpi", ".scn", ".mrxs", ".svslide"}
            if openslide and path.suffix.lower() in wsi_extensions:
                try:
                    self.image_obj = openslide.OpenSlide(str(path))
                    self.width, self.height = self.image_obj.dimensions
                    logging.info(f"Loaded WSI image: {path}")
                    return
                except openslide.OpenSlideError:
                    logging.warning(f"Could not open {path} with openslide, trying Pillow.")

            try:
                img = Image.open(path)
                self.image_obj = img.convert("RGB")
                self.width, self.height = self.image_obj.size
                logging.info(f"Loaded standard image with Pillow: {path}")
                return
            except Exception as e:
                raise IOError(f"Failed to load image file {path} with both OpenSlide and Pillow.") from e

        # 3. Handle pre-loaded image objects
        elif isinstance(source, Image.Image):
            self.image_obj = source.convert("RGB")
            self.width, self.height = self.image_obj.size
            logging.info("Loaded image from pre-loaded PIL.Image object.")
            return
        elif isinstance(source, np.ndarray):
            self.image_obj = Image.fromarray(source).convert("RGB")
            self.width, self.height = self.image_obj.size
            logging.info("Loaded image from pre-loaded NumPy array.")
            return
        
        raise TypeError(f"Unsupported image source type: {type(source)}")

    def get_dimensions(self) -> tuple[int, int]:
        return self.width, self.height

    def get_tile(self, coordinates: np.ndarray, tile_size: int) -> Image.Image:
        """
        Extracts a single image tile in memory. Handles boundary conditions.
        """
        col, row = int(round(coordinates[0])), int(round(coordinates[1]))
        half_tile = tile_size // 2

        top_left_x = col - half_tile
        top_left_y = row - half_tile

        read_left = max(top_left_x, 0)
        read_top = max(top_left_y, 0)
        read_right = min(top_left_x + tile_size, self.width)
        read_bottom = min(top_left_y + tile_size, self.height)
        
        read_width = read_right - read_left
        read_height = read_bottom - read_top

        if read_width <= 0 or read_height <= 0:
            return Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        
        if openslide and isinstance(self.image_obj, openslide.OpenSlide):
            region = self.image_obj.read_region((read_left, read_top), 0, (read_width, read_height)).convert("RGB")

        elif isinstance(self.image_obj, Image.Image):
            region = self.image_obj.crop((read_left, read_top, read_right, read_bottom))
        else:
            raise TypeError(f"Cannot extract tile from unsupported image object type: {type(self.image_obj)}")

        tile_img = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        paste_x = read_left - top_left_x
        paste_y = read_top - top_left_y
        tile_img.paste(region, (paste_x, paste_y))
        
        return tile_img
