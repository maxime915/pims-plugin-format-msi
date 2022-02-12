"""convertor: convert ImzML source to internal Zarr representation"""

import abc
import contextlib
import logging
import warnings
from functools import cached_property
from math import ceil
from pathlib import Path
from typing import BinaryIO, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
from zarr.util import normalize_chunks

from pims_plugin_format_msi.__version__ import VERSION

SHAPE = Tuple[int, int, int, int]


class ImzMLData(NamedTuple):
    """ImzMLData: holds a summary data for an ImzML pair"""

    spectra: pd.DataFrame

    mz_dtype: np.dtype
    int_dtype: np.dtype

    pixel_count: Tuple[int, int, int]

    is_continuous: bool

    source_imzml: Path
    source_ibd: Path

    uuid: str


def get_data_from_parser(
    imzml: Path, ibd: Path, ignore_warnings: bool = False
) -> ImzMLData:
    """parse an imzml / ibd file pair to an ImzMLData object

    Parameters
    ----------
    imzml : Path
        path to the .imzML file
    ibd : Path
        path to the .ibd file
    ignore_warnings : bool, optional
        remove bad accession warnings, by default True

    Returns
    -------
    ImzMLData
        A python representation of some essential data from the .imzML file

    Raises
    ------
    ValueError
        on invalid file mode
    """

    if ignore_warnings:
        warnings.filterwarnings("ignore", r"Accession I?MS")

    parser = PyImzMLParser(str(imzml), parse_lib="lxml", ibd_file=None)

    # check for binary mode
    is_continuous = "continuous" in parser.metadata.file_description.param_by_name
    is_processed = "processed" in parser.metadata.file_description.param_by_name

    if is_continuous == is_processed:
        raise ValueError(
            "invalid file mode, expected exactly one of " "'continuous' or 'processed'"
        )

    spectra = pd.DataFrame(parser.coordinates, columns=["x", "y", "z"])
    spectra = spectra.assign(mz_offset=parser.mzOffsets)
    spectra = spectra.assign(int_offset=parser.intensityOffsets)
    spectra = spectra.assign(length=parser.mzLengths)

    # offset coordinates for 0-based indexing
    spectra["x"] -= 1
    spectra["y"] -= 1
    spectra["z"] -= 1

    return ImzMLData(
        spectra=spectra,
        mz_dtype=np.dtype(parser.mzPrecision),
        int_dtype=np.dtype(parser.intensityPrecision),
        pixel_count=(
            1,
            parser.imzmldict["max count of pixels y"],
            parser.imzmldict["max count of pixels x"],
        ),
        is_continuous=is_continuous,
        source_imzml=imzml,
        source_ibd=ibd,
        uuid=parser.metadata.file_description.cv_params[0][2],
    )


def add_chunk_idx(
    spatial_shape: Tuple[int, int, int],
    spatial_chunks: Tuple[int, int, int],
    spectra: pd.DataFrame,
) -> Dict[int, Tuple[int, int, int]]:
    """add a 'chunk_idx' column to the dataframe

    Parameters
    ----------
    spatial_shape : Tuple[int, int, int]
        the ZYX shape of the image
    spatial_chunks : Tuple[int, int, int]
        the ZYX chunk shape of the image
    spectra : pd.DataFrame
        dataframe obtained from ImzMLData

    Returns
    -------
    Dict[int, Tuple[int, int, int]]
        a mapping from a flat chunk_idx to a 3D chunk idx per axis
    """

    # coordinates of spectra are assumed 0 based

    # use ceil() for incomplete chunks
    chunk_counts = [ceil(s / c) for (s, c) in zip(spatial_shape, spatial_chunks)]

    # per_chunk_idx is a multi-dimensional index, bounded by chunk_counts.
    # To have a scalar value, the index must be flattened, this uses column-major
    # order but row major would work too.
    strides = [1]
    for chunk_width in chunk_counts[:-1]:
        strides.append(strides[-1] * chunk_width)

    # get all coordinates in ZYX order
    coords = [spectra["z"], spectra["y"], spectra["x"]]
    # x = chunk_idx * chunk_shape + reminder -> find chunk_idx
    per_chunk_idx = [coord // chunk for (coord, chunk) in zip(coords, spatial_chunks)]
    # flatten using stride
    flat_idx = sum(idx * stride for idx, stride in zip(per_chunk_idx, strides))

    flat_to_idx = dict(zip(flat_idx, zip(*per_chunk_idx)))

    flat_to_idx = {}
    for flat, *per_chunk in zip(flat_idx, *per_chunk_idx):
        flat_to_idx[flat] = per_chunk

    if "chunk_idx" not in spectra:
        spectra["chunk_idx"] = flat_idx

    return flat_to_idx


def read_chunk_into(
    array: zarr.Array,
    file: BinaryIO,
    spectra: pd.DataFrame,
    chunk_idx: Tuple[int, int, int],
    offset_idx: int,
):
    """read a chunk into the zarr array efficiently

    WARNING: offset_idx must take the index column into account (e.g. if there \
        is an index, `offset_idx=spectra.columns.get_loc('mz_offset')+1` )

    Args:
        array (zarr.Array): zarr Array to write the chunk into
        file (BinaryIO): file to read the band from
        spectra (pd.DataFrame): bands information *for the current chunk only*
        chunk_idx (Tuple[int, int, int]): per axis index for the chunk
        offset_idx (int): column index for the offset column
        dtype (np.dtype): dtype of the
    """

    shape = array.shape[1:]
    chunk_width = array.chunks[1:]

    low_idx = (
        chunk_idx[0] * chunk_width[0],
        chunk_idx[1] * chunk_width[1],
        chunk_idx[2] * chunk_width[2],
    )

    high_idx = (
        min(shape[0], (chunk_idx[0] + 1) * chunk_width[0]),
        min(shape[1], (chunk_idx[1] + 1) * chunk_width[1]),
        min(shape[2], (chunk_idx[2] + 1) * chunk_width[2]),
    )

    chunk_slice = (
        slice(None),
        slice(low_idx[0], high_idx[0]),
        slice(low_idx[1], high_idx[1]),
        slice(low_idx[2], high_idx[2]),
    )

    buffer = np.zeros(
        order=array.order,
        shape=(
            array.shape[0],
            high_idx[0] - low_idx[0],
            high_idx[1] - low_idx[1],
            high_idx[2] - low_idx[2],
        ),
        dtype=array.dtype,
    )

    # read the chunk into a buffer
    for row in spectra.itertuples():
        length = row.length
        file.seek(row[offset_idx])
        idx = (
            slice(length),
            row.z - low_idx[0],
            row.y - low_idx[1],
            row.x - low_idx[2],
        )
        buffer[idx] = np.fromfile(file, count=length, dtype=array.dtype)

    # write to zarr (disk)
    array[chunk_slice] = buffer


class BaseImzMLConvertor(abc.ABC):
    "base class hiding the continuous VS processed difference behind polymorphism"

    def __init__(
        self,
        root: zarr.Group,
        name: str,
        data: ImzMLData,
        chunks=True,
        compressor="default",
        order="C",
        max_size: int = 4 * 2**30,
    ) -> None:
        super().__init__()

        self.root = root
        self.name = name
        self.data = data

        self.chunks = chunks
        self.compressor = compressor
        self.order = order

        self.max_size = max_size

        if self.max_size <= 0:
            raise ValueError(f"{max_size=} should be positive")

    @abc.abstractmethod
    def get_labels(self) -> List[str]:
        "return the list of labels associated with the image"

    def add_base_metadata(self) -> None:
        """add some OME-Zarr compliant metadata to the root group:
        - multiscales
        - labels

        as well as custom PIMS - MSI metadata in 'pims-msi'
        """

        axes = [dict(name="c", type="channel")]
        for axis in ["z", "y", "x"]:
            axes.append(dict(name=axis, type="spatial"))

        # multiscales metadata
        self.root.attrs["multiscales"] = [
            {
                "version": "0.3",
                "name": self.name,
                # store intensities in dataset 0
                "datasets": [
                    {"path": "0"},
                ],
                # NOTE axes attribute may change significantly in 0.4.0
                "axes": ["c", "z", "y", "x"],
                "type": "none",  # no downscaling (at the moment)
            }
        ]

        self.root.attrs["pims-msi"] = {
            "version": VERSION,
            "source": str(self.data.source_imzml),
            # image resolution ?
            "uuid": self.data.uuid,
            # find out if imzML come from a conversion, include it if so ?
            "binary_mode": ["processed", "continuous"][self.data.is_continuous],
        }

        # label group
        self.root.create_group("labels").attrs["labels"] = self.get_labels()

    @property
    @abc.abstractmethod
    def intensity_shape(self) -> SHAPE:
        "return an int tuple describint the array shape"

    @cached_property
    def intensity_chunks(self) -> SHAPE:
        "return an int tuple describing the chunk shape"

        # m/Z doesn't weight a lot in continuous file
        item_size = self.data.int_dtype.itemsize
        if not self.data.is_continuous:
            item_size = max(item_size, self.data.mz_dtype.itemsize)

        shape = self.intensity_shape

        chunks = list(normalize_chunks(self.chunks, shape, item_size))

        last_idx = 1
        while True:
            # data is read in full band chunks
            temp_chunk_size = shape[0] * np.prod(chunks[1:]) * item_size

            if temp_chunk_size < self.max_size:
                break

            # while too much, decrease the chunks shape in x, y, z
            idx = min(1, (last_idx + 1) % len(chunks))
            while idx < len(chunks):
                if chunks[idx] == 1:
                    idx += 1
                    continue
                chunks[idx] = ceil(chunks[idx] / 2.0)
                last_idx = idx
                break

            if idx == len(chunks):
                raise ValueError(
                    "masses are too deep to find an appropriate chunks size"
                )

        assert temp_chunk_size < self.max_size

        return tuple(chunks)

    @abc.abstractmethod
    def mz_chunks(self) -> SHAPE:
        "chunk shape for the masses label array"

    @abc.abstractmethod
    def create_zarr_arrays(self):
        """generate empty arrays inside the root group"""

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


class ContinuousImzMLConvertor(BaseImzMLConvertor):
    "convertor class for a continuous type file"

    def get_labels(self) -> List[str]:
        return ["mzs/0"]

    @cached_property
    def intensity_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the intensity array"
        # c=m/Z, z=1, y, x
        return (self.data.spectra.loc[0, "length"],) + self.data.pixel_count

    @cached_property
    def mz_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the mzs array"
        return (
            self.data.spectra.loc[0, "length"],  # c = m/Z
            1,  # z
            1,  # y
            1,  # x
        )

    def mz_chunks(self) -> SHAPE:
        int_chunks = self.intensity_chunks
        return int_chunks[:1] + (1, 1, 1)

    def create_zarr_arrays(self):
        """generate empty arrays inside the root group"""

        # array for the intensity values (main image)
        intensities = self.root.zeros(
            "0",
            shape=self.intensity_shape,
            dtype=self.data.int_dtype,
            chunks=self.intensity_chunks,
            compressor=self.compressor,
            order=self.order,
        )

        # xarray zarr encoding
        intensities.attrs["_ARRAY_DIMENSIONS"] = _get_xarray_axes(self.root)

        # array for the m/Z (as a label)
        self.root.zeros(
            "labels/mzs/0",
            shape=self.mz_shape,
            dtype=self.data.mz_dtype,
            chunks=self.mz_chunks(),
            compressor=None,  # useless for masses
            # order is irrelevant: data is effectively 1 dimensional
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

        flat_to_idx = add_chunk_idx(
            intensities.shape[1:], intensities.chunks[1:], self.data.spectra
        )
        chunk_lst = self.data.spectra.chunk_idx.unique()

        with open(self.data.source_ibd, mode="rb") as ibd_file:
            # read m/Z
            ibd_file.seek(self.data.spectra.mz_offset[0])
            mzs[:, 0, 0, 0] = np.fromfile(
                ibd_file, count=self.data.spectra.length[0], dtype=self.data.mz_dtype
            )

            # read intensities chunk by chunks
            for chunk_idx in chunk_lst:
                per_chunk_idx = flat_to_idx[chunk_idx]
                spectra = self.data.spectra
                spectra = spectra[spectra.chunk_idx == chunk_idx]
                read_chunk_into(
                    array=intensities,
                    file=ibd_file,
                    spectra=spectra,
                    chunk_idx=per_chunk_idx,
                    offset_idx=spectra.columns.get_loc("int_offset") + 1,
                )


class ProcessedImzMLConvertor(BaseImzMLConvertor):
    "convertor class for a processed type file"

    def get_labels(self) -> List[str]:
        return ["mzs/0", "lengths/0"]

    @cached_property
    def intensity_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the intensity array"
        max_length = self.data.spectra.length.max()
        return (max_length,) + self.data.pixel_count

    # no need to override intensity_chunks

    @cached_property
    def mz_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the mzs array"
        return self.intensity_shape

    def mz_chunks(self) -> SHAPE:
        return self.intensity_chunks

    @property
    def lengths_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the lengths array"
        return (1,) + self.data.pixel_count

    @property
    def lengths_chunks(self) -> SHAPE:
        "return an int tuple describing the chunks of the lengths array"
        return (1,) + self.intensity_chunks[1:]

    def create_zarr_arrays(self):
        """generate empty arrays inside the root group"""

        # array for the intensity values (main image)
        intensities = self.root.zeros(
            "0",
            shape=self.intensity_shape,
            dtype=self.data.int_dtype,
            chunks=self.intensity_chunks,
            compressor=self.compressor,
            order=self.order,
        )

        # xarray zarr encoding
        intensities.attrs["_ARRAY_DIMENSIONS"] = _get_xarray_axes(self.root)

        # array for the m/Z (as a label)
        self.root.zeros(
            "labels/mzs/0",
            shape=self.mz_shape,
            dtype=self.data.mz_dtype,
            chunks=self.mz_chunks(),
            compressor=self.compressor,
            order=self.order,
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
            "labels/lengths/0",
            shape=self.lengths_shape,
            dtype=np.uint32,
            chunks=self.lengths_chunks,
            compressor=None,
            order=self.order,
        )

    def read_binary_data(self) -> None:
        intensities = self.root[0]
        mzs = self.root.labels.mzs[0]

        zarr_lengths = self.root.labels.lengths[0]
        lengths = np.zeros(zarr_lengths.shape, dtype=zarr_lengths.dtype)

        flat_to_idx = add_chunk_idx(
            intensities.shape[1:], intensities.chunks[1:], self.data.spectra
        )
        chunk_lst = self.data.spectra.chunk_idx.unique()

        with open(self.data.source_ibd, mode="rb") as ibd_file:

            # read lengths into a numpy array as buffer
            for row in self.data.spectra.itertuples():
                lengths[0, row.z, row.y, row.x] = row.length
            # write to disk at once
            zarr_lengths[:] = lengths

            for chunk_idx in chunk_lst:
                per_chunk_idx = flat_to_idx[chunk_idx]
                spectra = self.data.spectra
                spectra = spectra[spectra.chunk_idx == chunk_idx]

                # read mzs
                read_chunk_into(
                    array=mzs,
                    file=ibd_file,
                    spectra=spectra,
                    chunk_idx=per_chunk_idx,
                    offset_idx=spectra.columns.get_loc("mz_offset") + 1,
                )

                # read intensities
                read_chunk_into(
                    array=intensities,
                    file=ibd_file,
                    spectra=spectra,
                    chunk_idx=per_chunk_idx,
                    offset_idx=spectra.columns.get_loc("int_offset") + 1,
                )


def _get_xarray_axes(root: zarr.Group) -> List[str]:
    "return a copy of the 'axes' multiscales metadata, used for XArray"
    return root.attrs["multiscales"][0]["axes"]


def convert(
    imzml: Path,
    ibd: Path,
    zarr_path: Path,
    /,
    name: Optional[str] = None,
    **kwargs,
) -> bool:
    """convert: standalone conversion function

    Parameters
    ----------
    imzml : Path
        path to the .imzML file
    ibd : Path
        path to the .ibd file
    zarr_path : Path
        path to the .zarr directory, must not exist
    name : Optional[str], optional
        metadata for the image, inferred from the .imzML file if omitted, by default None

    Returns
    -------
    bool
        conversion status (if False, nothing is written to the zarr directory)
    """

    if zarr_path.exists():
        logging.error("attempting to convert to an existing file, aborting.")
        return False

    with contextlib.ExitStack() as stack:
        if not name:
            name = zarr_path.stem

        # create the directory for the destination
        dest_store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=dest_store)

        # register a callback for automatic removal:
        #   unless stack.pop_all() is called the file will be removed
        #   before the context manager exit
        stack.callback(dest_store.rmdir)

        try:
            mz_data = get_data_from_parser(imzml, ibd)

            if mz_data.is_continuous:
                ContinuousImzMLConvertor(root, name, mz_data, **kwargs).run()
            else:
                ProcessedImzMLConvertor(root, name, mz_data, **kwargs).run()

        except (ValueError, KeyError) as error:
            logging.error("conversion error", exc_info=error)
            return False  # store is automatically removed by callback
        except Exception as error:
            # this should ideally never happen
            logging.exception(error)
            return False  # store is automatically removed by callback

        # remove callback to avoid file removal & indicate successful conversion
        stack.pop_all()
        return True

    return False