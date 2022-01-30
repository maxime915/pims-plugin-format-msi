"""convertor: convert ImzML source to internal Zarr representation"""

from __future__ import annotations

import abc
import contextlib
import logging
from typing import List, TYPE_CHECKING, Tuple, Type

import numpy as np
import zarr
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser

from pims.formats import AbstractFormat
from pims.formats.common.zarr import ZarrFormat
from pims.formats.utils.convertor import AbstractConvertor
from pims_plugin_format_msi.__version__ import VERSION
from pims_plugin_format_msi.utils.temp_store import single_temp_store
from .utils import get_imzml_pair

if TYPE_CHECKING:
    from pims.files.file import Path

# byte size over which the whole structure is not copied
_DISK_COPY_THRESHOLD = 8 * 10 ** 9

SHAPE = Tuple[int, int, int, int]


# class PIMSOpenMicroscopyEnvironmentZarr(AbstractFormat):
#     "placeholder while PIMS lack the internal Zarr format"
PIMSOpenMicroscopyEnvironmentZarr = ZarrFormat


@contextlib.contextmanager
def load_parser(imzml: Path, ibd: Path):
    """load a parser object from pyimzml

    Args:
        imzml (Path): path to the ImzML file
        ibd (Path): path to the ibd file

    Yields:
        Iterator[pyimzml.ImzMLParser.ImzMLParser]
    """
    with open(str(ibd), mode='rb') as ibd_file:
        yield PyImzMLParser(
            filename=str(imzml),
            parse_lib='lxml',
            ibd_file=ibd_file  # the ibd file has to be opened manually
        )


def copy_array(source: zarr.Array, destination: zarr.Array) -> None:
    "copy a array (ragged arrays non supported)"
    array_size = source.nbytes  # Zarr can be trusted here
    if array_size <= _DISK_COPY_THRESHOLD:
        # load all data in memory then write at once
        #   - usually faster
        destination[:] = source[:]
    else:
        # chunk by chunk loading
        #   - smaller memory footprint
        destination[:] = source


class _BaseImzMLConvertor(abc.ABC):
    "base class hiding the continuous VS processed difference behind polymorphism"

    def __init__(self, root: zarr.Group, name: str, parser: PyImzMLParser) -> None:
        super().__init__()

        self.root = root
        self.name = name
        self.parser = parser

    @abc.abstractmethod
    def get_labels(self) -> List[str]:
        "return the list of labels associated with the image"

    def add_base_metadata(self) -> None:
        """add some OME-Zarr compliant metadata to the root group:
        - multiscales
        - labels

        as well as custom PIMS - MSI metadata in 'pims-msi'
        """

        # multiscales metadata
        self.root.attrs['multiscales'] = [{
            'version': '0.3',
            'name': self.name,
            # store intensities in dataset 0
            'datasets': [{'path': '0'}, ],
            # NOTE axes attribute may change significantly in 0.4.0
            'axes': ['c', 'z', 'y', 'x'],
            'type': 'none',  # no downscaling (at the moment)
        }]

        self.root.attrs['pims-msi'] = {
            'version': VERSION,
            'source': self.parser.filename,
            # image resolution ?
            'uuid': self.parser.metadata.file_description.cv_params[0][2],
            # find out if imzML come from a conversion, include it if so ?
        }

        # label group
        self.root.create_group('labels').attrs['labels'] = self.get_labels()

    @abc.abstractmethod
    def create_zarr_arrays(self):
        """generate empty arrays inside the root group
        """

    @abc.abstractmethod
    def read_binary_data(self) -> None:
        """fill in the arrays defined with the ibd file from the source

        NOTE: missing coordinates will not write to the array, make sure the
        current value for the array is suitable.
        """

    def run(self) -> None:
        "main method"
        self.add_base_metadata()
        self.create_zarr_arrays()
        self.read_binary_data()


class _ContinuousImzMLConvertor(_BaseImzMLConvertor):
    def get_labels(self) -> List[str]:
        return ['mzs/0']

    def get_intensity_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the intensity array"
        return (
            self.parser.mzLengths[0],                        # c = m/Z
            1,                                               # z = 1
            self.parser.imzmldict['max count of pixels y'],  # y
            self.parser.imzmldict['max count of pixels x'],  # x
        )

    def get_mz_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the mzs array"
        return (
            self.parser.mzLengths[0],  # c = m/Z
            1,                         # z
            1,                         # y
            1,                         # x
        )

    def create_zarr_arrays(self):
        """generate empty arrays inside the root group
        """

        # array for the intensity values (main image)
        intensities = self.root.zeros(
            '0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
            # default chunks & compressor
        )

        # xarray zarr encoding
        intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)

        # array for the m/Z (as a label)
        self.root.zeros(
            'labels/mzs/0',
            shape=self.get_mz_shape(),
            dtype=self.parser.mzPrecision,
            # default chunks
            compressor=None,
        )

        # # NOTE: for now, z axis is supposed to be a Zero for all values
        # # array for z value (as a label)
        # z_values = self.root.zeros(
        #     'labels/z/0',
        #     shape=self.get_z_shape(),
        #     dtype=float,
        #     compressor=None,
        # )

    def read_binary_data(self) -> None:
        intensities = self.root[0]
        mzs = self.root.labels.mzs[0]
        with single_temp_store() as fast_store:
            # create an array for the temporary intensities
            fast_intensities = zarr.group(fast_store).zeros(
                '0',
                shape=intensities.shape,
                dtype=intensities.dtype,
                chunks=(-1, 1, 1, 1),  # similar to the .ibd structure
                compressor=None,
            )

            # fill m/Z into the destination group
            self.parser.m.seek(self.parser.mzOffsets[0])
            mzs[:, 0, 0, 0] = np.fromfile(self.parser.m, count=self.parser.mzLengths[0],
                                          dtype=self.parser.mzPrecision)

            # fill intensities into the fast group
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                self.parser.m.seek(self.parser.intensityOffsets[idx])
                fast_intensities[:, 0, y-1, x-1] = np.fromfile(
                    self.parser.m, count=self.parser.intensityLengths[idx],
                    dtype=self.parser.intensityPrecision)

            # re-chunk
            copy_array(fast_intensities, intensities)


class _ProcessedImzMLConvertor(_BaseImzMLConvertor):
    def get_labels(self) -> List[str]:
        return ['mzs/0', 'lengths/0']

    def get_intensity_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the intensity array"
        return (
            max(self.parser.mzLengths),                      # c = m/Z
            1,                                               # z = 1
            self.parser.imzmldict['max count of pixels y'],  # y
            self.parser.imzmldict['max count of pixels x'],  # x
        )

    def get_mz_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the mzs array"
        return self.get_intensity_shape()

    def get_lengths_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the lengths array"
        return (
            1,                                               # c = m/Z
            1,                                               # z = 1
            self.parser.imzmldict['max count of pixels y'],  # y
            self.parser.imzmldict['max count of pixels x'],  # x
        )

    def create_zarr_arrays(self):
        """generate empty arrays inside the root group
        """

        # array for the intensity values (main image)
        intensities = self.root.zeros(
            '0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
            # default chunks & compressor
        )

        # xarray zarr encoding
        intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)

        # array for the m/Z (as a label)
        self.root.zeros(
            'labels/mzs/0',
            shape=self.get_mz_shape(),
            dtype=self.parser.mzPrecision,
            # default chunks & compressor
        )

        # # NOTE: for now, z axis is supposed to be a Zero for all values
        # # array for z value (as a label)
        # z_values = self.root.zeros(
        #     'labels/z/0',
        #     shape=self.get_z_shape(),
        #     dtype=float,
        #     compressor=None,
        # )

        # array for the lengths (as a label)
        self.root.zeros(
            'labels/lengths/0',
            shape=self.get_lengths_shape(),
            dtype=np.uint32,
            # default chunks
            compressor=None,
        )

    def read_binary_data(self) -> None:
        intensities = self.root[0]
        mzs = self.root.labels.mzs[0]
        lengths = self.root.labels.lengths[0]

        with single_temp_store() as fast_store:
            fast_group = zarr.group(fast_store)

            # create arrays for the temporary intensities & masses
            fast_intensities = fast_group.zeros(
                '0',
                shape=intensities.shape,
                dtype=intensities.dtype,
                chunks=(-1, 1, 1, 1),  # similar to the .ibd structure
                compressor=None,
            )
            fast_mzs = fast_group.zeros(
                'mzs',
                shape=mzs.shape,
                dtype=mzs.dtype,
                chunks=(-1, 1, 1, 1),
                compressor=None,
            )

            # read the data into the fast arrays
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                length = self.parser.mzLengths[idx]
                lengths[0, 0, y-1, x-1] = length
                spectra = self.parser.getspectrum(idx)
                fast_mzs[:length, 0, y-1, x-1] = spectra[0]
                fast_intensities[:length, 0, y-1, x-1] = spectra[1]

            # re-chunk
            copy_array(fast_intensities, intensities)
            copy_array(fast_mzs, mzs)


def convert_to_store(name: str, source_dir: Path, dest_store: zarr.DirectoryStore) -> None:
    """convert an imzML from a folder containing the imzML & ibd files to a \
        Zarr group.

    Args:
        source (Path): Folder containing the imzML & ibd files
        destination (MutableMapping): where to store the image (see \
            zarr.DirectoryStore, zarr.MemoryStore, etc.)

    Raises:
        ValueError: if no valid imzML file can be found in the source folder
    """

    pair = get_imzml_pair(source_dir)

    if pair is None:
        raise ValueError('not an imzML file')

    # load the parser
    with load_parser(*pair) as parser:

        # check for binary mode
        is_continuous = 'continuous' in parser.metadata.file_description.param_by_name
        is_processed = 'processed' in parser.metadata.file_description.param_by_name

        if is_continuous == is_processed:
            raise ValueError("invalid file mode, expected exactly one of "
                             "'continuous' or 'processed'")

        # create OME-Zarr structure
        root = zarr.group(store=dest_store)

        if is_continuous:
            _ContinuousImzMLConvertor(root, name, parser).run()
        else:
            _ProcessedImzMLConvertor(root, name, parser).run()


def _get_xarray_axes(root: zarr.Group) -> List[str]:
    "return a copy of the 'axes' multiscales metadata, used for XArray"
    return root.attrs['multiscales'][0]['axes']


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

        if dest_path.exists():
            return False

        with contextlib.ExitStack() as stack:
            name = dest_path.true_stem

            # create the directory for the destination
            dest_store = zarr.DirectoryStore(dest_path)

            # register a callback for automatic removal:
            #   unless stack.pop_all() is called the file will be removed
            #   before the context manager exit
            stack.callback(dest_store.rmdir)

            try:
                # do conversion in dedicated function
                convert_to_store(name, self.source.path, dest_store)
            except (ValueError, KeyError) as error:
                logging.error('conversion error', exc_info=error)
                return False  # store is automatically removed by callback
            except Exception as error:
                # this should ideally never happen
                logging.error('unexpected exception caught', exc_info=error)
                return False  # store is automatically removed by callback

            # remove callback to avoid file removal & indicate successful conversion
            stack.pop_all()
            return True

        return False

    def conversion_format(self) -> Type[AbstractFormat]:
        """
        Get the format to which the image in this format will be converted,
        if needed.
        """
        return PIMSOpenMicroscopyEnvironmentZarr
