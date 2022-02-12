"""convertor: convert ImzML source to internal Zarr representation"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

from pims.formats import AbstractFormat
from pims.formats.common.zarr import ZarrFormat
from pims.formats.utils.convertor import AbstractConvertor

from pims_plugin_format_msi.utils.imzml.convertor.backend import convert
from pims_plugin_format_msi.utils.imzml.utils import get_imzml_pair

if TYPE_CHECKING:
    from pims.files.file import Path


class ImzMLToZarrConvertor(AbstractConvertor):
    "PIMS compatible converter class"

    def convert(self, dest_path: Path) -> bool:
        """
        Convert the image in this format to another one at a given destination \
            path.

        Returns
        -------
        result
            Whether the conversion succeeded or not
        """

        pair = get_imzml_pair(self.source.path)
        if not pair:
            logging.error("unable to find ImzML/IBD files")

        # no parameters given : default chunks, compressor, C memory order
        return convert(*pair, dest_path, name=dest_path.true_stem)

    def conversion_format(self) -> Type[AbstractFormat]:
        """
        Get the format to which the image in this format will be converted,
        if needed.
        """
        return ZarrFormat
