"parser: read ImzML metadata"

from __future__ import annotations

import warnings

import numpy as np
import pyimzml.ImzMLParser

from pims.formats.utils.abstract import AbstractFormat
from pims.formats.utils.parser import AbstractParser
from pims.formats.utils.structures.metadata import ImageChannel, ImageMetadata, MetadataStore
from pims_plugin_format_msi.utils.imzml.utils import get_imzml_pair

_REMOVE_WARNINGS = False

class ImzMLParser(AbstractParser):
    "PIMS Parser for ImzML"

    def __init__(self, format: AbstractFormat):
        super().__init__(format)
        self._parser = None

    def get_parser(self) -> pyimzml.ImzMLParser.ImzMLParser:
        "returns the pyimzml.ImzMLParser.ImzMLParser corresponding to parsed file"

        if self._parser is None:
            # pyimzml generates a few warning about ontology on valid files

            if _REMOVE_WARNINGS:
                warnings.filterwarnings('ignore', message=r'.*Accession IMS.*')
                warnings.filterwarnings('ignore', message=r'.*Accession MS.*')

            imz_path, _ = get_imzml_pair(self.format.path)

            self._parser = pyimzml.ImzMLParser.ImzMLParser(
                str(imz_path),
                parse_lib='lxml',  # only "safe" XML parsing library available
                # the parser doesn't need the ibd information (yet)
                ibd_file=None,
                include_spectra_metadata=None,  # this could be useful but hard to tune
            )

        return self._parser

    def parse_main_metadata(self) -> ImageMetadata:
        """
        Parse minimal set of required metadata for any PIMS request.
        This method must be as fast as possible.

        Main metadata that must be parsed by this method are:
        * width
        * height
        * depth
        * duration
        * n_channels
        * n_channels_per_read
        * n_distinct_channels
        * pixel_type
        * significant_bits
        * for every channel:
            * index
            * color (can be None)
            * suggested_name (can be None, used to infer color)
        """

        parser = self.get_parser()

        # check for binary mode
        is_continuous = 'continuous' in parser.metadata.file_description.param_by_name
        is_processed = 'processed' in parser.metadata.file_description.param_by_name

        if is_continuous == is_processed:
            raise ValueError("invalid file mode, expected exactly one of "
                             "'continuous' or 'processed'")

        metadata = ImageMetadata()

        metadata.width = parser.imzmldict['max count of pixels x']
        metadata.height = parser.imzmldict['max count of pixels y']

        metadata.depth = 1
        metadata.duration = 1

        if is_continuous:
            metadata.n_channels = parser.mzLengths[0]
        else:
            metadata.n_channels = max(parser.mzLengths)
        metadata.n_channels_per_read = metadata.n_channels
        metadata.n_distinct_channels = metadata.n_channels
        
        metadata.pixel_type = np.dtype('uint16')  # np.dtype(parser.intensityPrecision)
        metadata.significant_bits = metadata.pixel_type.itemsize
        
        for channel in range(metadata.n_channels):
            metadata.set_channel(ImageChannel(channel))

        return metadata

    def parse_known_metadata(self) -> ImageMetadata:
        """
        Parse all known standardised metadata. In practice, this method
        completes the image metadata object partially filled by
        `parse_main_metadata`.

        This method should set `imd.is_complete` to True before returning `imd`.
        """

        metadata = self.format.main_imd

        # TODO find documentation on which metadata can be used
        # fill m/Z into the destination group
        imz_path, ibd = get_imzml_pair(self.format.path)
        with open(str(ibd), mode='rb') as ibd_file:
            parser = pyimzml.ImzMLParser.ImzMLParser(
                filename=str(imz_path),
                parse_lib='lxml',
                ibd_file=ibd_file  # the ibd file has to be opened manually
            )
            parser.m.seek(parser.mzOffsets[0])
            mzs = np.fromfile(parser.m, count=parser.mzLengths[0], dtype=parser.mzPrecision)
            for channel in range(metadata.n_channels):
                metadata.channels[channel].suggested_name = f"{mzs[channel]:.5f}"

        metadata.is_complete = True

        return metadata

    def parse_raw_metadata(self) -> MetadataStore:
        """
        Parse all raw metadata in a generic store. Raw metadata are not
        standardised and highly depend on underlying parsed format.

        Raw metadata MUST NOT be used by PIMS for processing.
        This method is expected to be SLOW.
        """

        # TODO

        return MetadataStore()
