"ImzML file format description"

from __future__ import annotations

from functools import cached_property

from pims.formats import AbstractFormat

from pims_plugin_format_msi.imzml.checker import ImzMLChecker
from pims_plugin_format_msi.imzml.convertor import ImzMLToZarrConvertor
from pims_plugin_format_msi.imzml.parser import ImzMLParser


class NotImplementedClass:
    """class that does nothing when constructed and raises a NotImplementedError
    when accessed in any way"""
    # remove when sufficient progress has been made
    __slots__ = ()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __getattribute__(self, __name: str):
        raise NotImplementedError()

    def __setattr__(self, __name: str, __value) -> None:
        raise NotImplementedError()


class ImzMLFormat(AbstractFormat):
    "PIMS Format definition for ImzML"

    checker_class = ImzMLChecker
    parser_class = ImzMLParser
    reader_class = NotImplementedClass  # TODO ?
    convertor_class = ImzMLToZarrConvertor

    histogram_reader_class = NotImplementedClass  # TODO ?

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        # defines the datapath for the source
        self._path = path
        self._enabled = True

    @classmethod
    def get_name(cls) -> str:
        "get_name returns the name of the file format"
        return "ImzML"

    @classmethod
    def is_spatial(cls) -> str:
        # ImzML should be used for conversion only
        return False

    @classmethod
    def is_spectral(cls) -> str:
        # ImzML should be used for conversion only
        return False

    @classmethod
    def get_remarks(cls) -> str:
        """Get format remarks in a human-readable way."""
        return "Only continuous ImzML images are supported for now"

    @classmethod
    def is_writable(cls):
        return False

    # Conversion

    @cached_property
    def need_conversion(self) -> bool:
        """
        Whether the image in this format needs to be converted to another one.
        Decision can be made based on the format metadata.
        """
        return True
