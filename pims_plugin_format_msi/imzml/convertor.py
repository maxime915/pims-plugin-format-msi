"""convertor: convert ImzML source to internal Zarr representation"""

from __future__ import annotations

import contextlib
import pathlib
from typing import MutableMapping, Type

import numpy as np
import pyimzml.ImzMLParser
import zarr
from pims.files.file import Path
from pims.formats import AbstractFormat
from pims.formats.utils.convertor import AbstractConvertor

from ..utils.temp_store import single_temp_store
from .utils import get_imzml_pair

_DISK_COPY_THRESHOLD = 8 * 10 ** 9


class PIMSOpenMicroscopyEnvironmentZarr(AbstractFormat):
    "placeholder while PIMS lack the internal Zarr format"


def _convert(parser: pyimzml.ImzMLParser.ImzMLParser,
             intensities: zarr.Array,
             mzs: zarr.Array) -> None:
    """main conversion function

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


def convert_to_store(source: Path, destination: MutableMapping) -> None:
    """convert an imzML from a folder containing the imzML & ibd files to a \
        Zarr group.

    Args:
        source (Path): Folder containing the imzML & ibd files
        destination (MutableMapping): where to store the image (see \
            zarr.DirectoryStore, zarr.MemoryStore, etc.)

    Raises:
        ValueError: if no valid imzML file can be found in the source folder
    """

    pair = get_imzml_pair(source)

    if pair is None:
        raise ValueError('not an imzML file')

    imzml, ibd = pair

    parser = pyimzml.ImzMLParser.ImzMLParser(
        filename=str(imzml),
        parse_lib='lxml',
        ibd_file=open(str(ibd), mode='rb')
    )

    shape = (parser.mzLengths[0],                        # c = m/Z
             1,                                          # z = 1
             parser.imzmldict['max count of pixels y'],  # y
             parser.imzmldict['max count of pixels x'])  # x

    # create OME-Zarr structure
    root = zarr.group(store=destination)

    # multiscales metadata
    root.attrs['multiscales'] = [{
        'version': '0.3',
        'name': pathlib.Path(destination.path).stem,
        # store intensities in dataset 0
        'datasets': [{'path': '0'}, ],
        # NOTE axes attribute may change significantly in 0.4.0
        'axes': ['c', 'z', 'y', 'x'],
        'type': 'none',  # no downscaling (at the moment)
        # NOTE there are probably other metadata useful, to investigate
        'metadata': {
                'original': str(source.path),
                'uuid': parser.metadata.file_description.cv_params[0][2],
        },
        # TODO image dimension & resolution (?)
    }]

    # Omero attributes
    root.attrs['omero'] = {
        'channels': [],  # TODO
        'rdefs': {'model': ''}  # TODO
    }

    # label group
    root.create_group('labels').attrs['labels'] = [
        'mzs/0',  # path to the m/Z values for this image
    ]

    # array for the intensity values (main image)
    intensities = root.zeros(
        '0',
        shape=shape,
        dtype=parser.intensityPrecision,
        # default chunks & compressor (NOTE: subject to change)
    )

    # xarray zarr enconding
    intensities.attrs['_ARRAY_DIMENSIONS'] = root.attrs['multiscales'][0]['axes']

    # array for the m/Z (as a label)
    mzs = root.zeros(
        'labels/mzs/0',
        shape=(shape[0], 1, 1, 1),
        dtype=parser.mzPrecision,
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

    _convert(parser, intensities, mzs)


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
            # create the directory for the destination
            dest_store = zarr.DirectoryStore(dest_path)

            # register a callback for automatic removal:
            #   unless stack.pop_all() is called the file will be removed
            #   before the context manager exit
            stack.callback(dest_store.rmdir)

            try:
                # do conversion in dedicated function
                convert_to_store(self.source.path, dest_store)
            except (ValueError, KeyError) as e:
                raise e
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
