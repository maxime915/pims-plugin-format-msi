"""convertor: convert ImzML source to internal Zarr representation"""

from __future__ import annotations

from typing import Type

#import numpy as np
#import pyimzml
#import zarr
from pims.files.file import Path
from pims.formats import AbstractFormat
from pims.formats.utils.convertor import AbstractConvertor


class PIMSOpenMicroscopyEnvironmentZarr(AbstractFormat):
    "placeholder while PIMS lack the internal Zarr format"
    pass


class ImzMLToZarrConvertor(AbstractConvertor):
    "PIMS compatible converter class"

    def convert(self, dest_path: Path) -> bool:
        """
        Convert the image in this format to another one at a given destination
        path.

        Returns
        -------
        result
            Whether the conversion succeeded or not
        """
        raise NotImplementedError()



    def conversion_format(self) -> Type[AbstractFormat]:
        """
        Get the format to which the image in this format will be converted,
        if needed.
        """
        return PIMSOpenMicroscopyEnvironmentZarr
