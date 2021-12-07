"""convertor: convert ImzML source to internal Zarr representation"""

from __future__ import annotations

import contextlib
from typing import List, Tuple, Type

import numpy as np
import pyimzml.ImzMLParser
import zarr
from pims.files.file import Path
from pims.formats import AbstractFormat
from pims.formats.utils.convertor import AbstractConvertor

from ..utils.temp_store import single_temp_store
from .utils import get_imzml_pair

from ..__version__ import VERSION

_DISK_COPY_THRESHOLD = 8 * 10 ** 9


class PIMSOpenMicroscopyEnvironmentZarr(AbstractFormat):
    "placeholder while PIMS lack the internal Zarr format"


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
        yield pyimzml.ImzMLParser.ImzMLParser(
            filename=str(imzml),
            parse_lib='lxml',
            ibd_file=ibd_file  # the ibd file has to be opened manually
        )


def _read_continuous_imzml(parser: pyimzml.ImzMLParser.ImzMLParser,
                           intensities: zarr.Array, mzs: zarr.Array) -> None:
    """main conversion function for continuous files

    Args:
        parser (pyimzml.ImzMLParser.ImzMLParser): initialized parser for the image
        intensities (zarr.Array): array of the right shape
        mzs (zarr.Array): array of the right shape

    NOTE: missing coordinates will not write to the array, make sure the default
    value for the array is suitable
    """

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
        parser.m.seek(parser.mzOffsets[0])
        mzs[:, 0, 0, 0] = np.fromfile(parser.m, count=parser.mzLengths[0],
                                      dtype=parser.mzPrecision)

        # fill intensities into the fast group
        for idx, (x, y, _) in enumerate(parser.coordinates):
            parser.m.seek(parser.intensityOffsets[idx])
            fast_intensities[:, 0, y-1, x-1] = np.fromfile(
                parser.m, count=parser.intensityLengths[idx],
                dtype=parser.intensityPrecision)

        # re-chunk
        array_size = fast_intensities.nbytes  # Zarr can be trusted here
        if array_size <= _DISK_COPY_THRESHOLD:
            # load all data in memory then write at once
            #   - usually faster
            intensities[:] = fast_intensities[:]
        else:
            # chunk by chunk loading
            #   - smaller memory footprint
            intensities[:] = fast_intensities


def _add_base_metadata(root: zarr.Group, name: str, source: str, uuid: str) -> None:
    """add some OME-Zarr compliant metadata to the root group:
        - multiscales
        - labels

    as well as custom PIMS - MSI metadata in 'pims-msi'
    """

    # multiscales metadata
    root.attrs['multiscales'] = [{
        'version': '0.3',
        'name': name,
        # store intensities in dataset 0
        'datasets': [{'path': '0'}, ],
        # NOTE axes attribute may change significantly in 0.4.0
        'axes': ['c', 'z', 'y', 'x'],
        'type': 'none',  # no downscaling (at the moment)
    }]

    root.attrs['pims-msi'] = {
        'version': VERSION,
        'source': source,
        # image resolution ?
        'uuid': uuid,
        # find out if imzML come from a conversion, include it if so ?
    }

    # label group
    root.create_group('labels').attrs['labels'] = [
        'mzs/0',  # path to the m/Z values for this image
        # 'mzs/z',  # path to the depth values for this image
    ]


def _create_zarr_arrays(root: zarr.Group, shape: Tuple[int, int, int, int],
                        intensity_dtype: np.dtype, mz_dtype: np.dtype
                        ) -> Tuple[zarr.Array, zarr.Array]:
    """create the arrays for the zarr Group

    Args:
        root (zarr.Group): group for the whole image
        shape (Tuple[int]): (number of spectra, depth, height, width)
        intensity_dtype (np.dtype): datatype for the intensities (e.g. 'f4', 'f8', etc.)
        mz_dtype (np.dtype): datatype for the masses (e.g. 'f4', 'f8', etc.)

    Returns:
        - the intensity array
        - the masses array
        # TODO: the depth array
    """

    # array for the intensity values (main image)
    intensities = root.zeros(
        '0',
        shape=shape,
        dtype=intensity_dtype,
        # default chunks & compressor (NOTE: subject to change)
    )

    # xarray zarr enconding
    intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(root)

    # array for the m/Z (as a label)
    mzs = root.zeros(
        'labels/mzs/0',
        shape=(shape[0], 1, 1, 1),
        dtype=mz_dtype,
        # default chunks
        compressor=None,
    )

    # NOTE: for now, z axis is supposed to be a Zero for all values
    # array for z value (as a label)
    # z_values = root.zeros(
    #     'labels/z/0',
    #     shape=(1, shape[1], 1, 1),
    #     dtype=float,
    #     compressor=None,
    # )

    return intensities, mzs


def _get_xarray_axes(root: zarr.Group) -> List[str]:
    "return a copy of the 'axes' multiscales metadata, used for XArray"
    return root.attrs['multiscales'][0]['axes']


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

    with load_parser(*pair) as parser:

        # create OME-Zarr structure
        root = zarr.group(store=dest_store)

        _add_base_metadata(root, name=name, source=parser.filename,
                           uuid=parser.metadata.file_description.cv_params[0][2])

        # check for binary mode
        is_continuous = 'continuous' in parser.metadata.file_description.param_by_name
        is_processed = 'processed' in parser.metadata.file_description.param_by_name

        if is_continuous == is_processed:
            raise ValueError("invalid file mode, "
                             "expected one of 'continuous' or 'processed'")

        if is_continuous:
            shape = (parser.mzLengths[0],                        # c = m/Z
                     1,                                          # z = 1
                     parser.imzmldict['max count of pixels y'],  # y
                     parser.imzmldict['max count of pixels x'])  # x

            intensities, mzs = _create_zarr_arrays(
                root, shape, parser.intensityPrecision, parser.mzPrecision)

            _read_continuous_imzml(parser, intensities, mzs)

        else:  # -> processed file
            raise NotImplementedError("processed type ImzML files unsupported")


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
            except (ValueError, KeyError) as exception:
                # TODO use a proper logger and clean this up
                print(f'caught {exception=}')
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
