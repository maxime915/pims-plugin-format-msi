"parser: read ImzML metadata"

from __future__ import annotations

from functools import cached_property

import pyimzml.ImzMLParser
from pims.formats.utils.parser import AbstractParser
from pims.formats.utils.structures.metadata import ImageMetadata, MetadataStore


class ImzMLParser(AbstractParser):
    "PIMS Parser for ImzML"

    @cached_property
    def raw_parser(self) -> pyimzml.ImzMLParser.ImzMLParser:
        "returns the pyimzml.ImzMLParser.ImzMLParser corresponding to parsed file"

        # warnings.filterwarnings('ignore', message=r'.*Accession IMS.*')
        return pyimzml.ImzMLParser.ImzMLParser(
            self.format.path,
            parse_lib='lxml',  # only "safe" XML parsing library available
            ibd_file=None,  # the parser doesn't need the ibd information (yet)
            include_spectra_metadata=None,  # this could be useful but hard to tune
        )

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

        # continuous = 'continuous' in self.raw_parser.metadata.file_description.param_by_name
        processed = 'processed' in self.raw_parser.metadata.file_description.param_by_name

        if processed:
            raise ValueError('ImzML file unsupported: only continuous mode is '
                             'supported')

        # TODO is self.format.main_imd defined when entering this function ?
        metadata = self.format.main_imd

        metadata.width = self.raw_parser.imzmldict['max count of pixels x']
        metadata.height = self.raw_parser.imzmldict['max count of pixels y']
        # NOTE this ignores 3D imzML files, but the interface is not finalized yet
        metadata.depth = 1
        # TODO in non time-varying data, is duration 1 ?
        metadata.duration = 1
        # NOTE this assumes a continuous file
        metadata.n_channels = self.raw_parser.mzLengths[0]
        # TODO what are the other metadata ?

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
