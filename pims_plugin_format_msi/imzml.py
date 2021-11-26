"""imzml: PIMS format definition for ImzML mass spectrometry images
implementing required utilities (checker, parser) and conversion.
"""

from pims.formats import AbstractFormat
#from pims.formats.utils.abstract import CachedDataPath
#from pims.formats.utils.checker import AbstractChecker
#from pims.formats.utils.histogram import DefaultHistogramReader

class ImzMLFormat(AbstractFormat):
    """
    ImzMLFormat: PIMS format class
    """

    checker_class = None # TODO
    parser_class = None # TODO
    reader_class = None # TODO
    histogram_reader_class = None # TODO

    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        # TODO what are these used for ?
        self._path = path
        self._enabled = True

    @classmethod
    def get_name(cls) -> str:
        "get_name returns the name of the file format"
        return "ImzML"

    @classmethod
    def get_remarks(cls) -> str:
        "get_remarks returns information about the file format"
        return "One .zarr folder containing an 'intensity' and 'mzs' groups"

    @classmethod
    def is_spatial(cls) -> str:
        "TODO: what is this used for ?"
        return True # TODO is this right ?
