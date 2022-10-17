"""convertor: CLI tool to convert file to zarr

python -m pims_plugin_format_msi.utils.imzml.convertor --help
"""

import argparse
import pathlib
import sys
import uuid

from .backend import convert


def get_parser(
    *,
    add_benchmark: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("imzml", type=str, help="input imzML file")
    parser.add_argument(
        "zarr",
        nargs="?",
        type=str,
        default=None,
        help="output path, defaults to random unique suffix",
    )

    # chunk
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument(
        "--auto-chunk",
        dest="auto_chunk",
        action="store_true",
        help="automatic chunking for array storage (default option)",
    )
    chunk_group.add_argument(
        "--no-chunk",
        dest="auto_chunk",
        action="store_false",
        help="disable chunking for array storage",
    )
    chunk_group.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[],
        help="Chunk shape : list of positive integers (-1 means equal to array shape)",
    )
    parser.set_defaults(auto_chunk=True)

    # rand name
    rand_name = parser.add_mutually_exclusive_group()
    rand_name.add_argument(
        "--add-rand-suffix",
        dest="rand_name",
        action="store_true",
        help="if zarr is not specified, guess the zarr file by adding a random UUID4 suffix and changing the extension",
    )
    rand_name.add_argument(
        "--no-rand-suffix",
        dest="rand_name",
        action="store_false",
        help="if zarr is not specified, guess the zarr file by changing the extension (default)",
    )
    parser.set_defaults(rand_name=False)

    # C / Fortran array order
    parser.add_argument(
        "--order",
        default="C",
        choices=["C", "F"],
        help="Memory layout to be used within each chunk (C vs Fortran)",
    )

    # Compressor -> default, none, or some evaluated object
    parser.add_argument(
        "--compressor",
        type=str,
        default="default",
        help="either"
        + ' "default", "none" or some python code to construct'
        + " the compressor object",
    )

    parser.add_argument(
        "--max-GB-size",
        type=int,
        help=(
            "maximum amount of memory used by the conversion "
            "expressed in gigabytes (without unit)"
            "(no chunk will exceed this dimension, this will "
            "limit the parallelism of the conversion when it "
            "is implemented"
        ),
    )

    parser.add_argument(
        "--ignore-accession-warnings",
        type=bool,
        default=False,
        help="remove pyImzML's warnings about incorrect accession",
    )

    parser.add_argument(
        "--auto-crop",
        type=bool,
        default=False,
        help="remove padding (pixels with no spectra) before conversion",
    )

    parser.add_argument(
        "--trim-long-outliers",
        type=bool,
        default=False,
        help="remove spectra which have a length longer than p50 + icr-f * (p75 - p25)",
    )

    parser.add_argument(
        "--trim-long-outliers-icf-f",
        type=bool,
        default=False,
        help="inter quartile factor, see --trim-long-outliers (ignored if --trim-long-outliers is False)",
    )

    if add_benchmark:
        parser.add_argument("--benchmark", action="store_true", help="time conversion")

    return parser


def get_args(*, parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = get_parser()
    args = vars(parser.parse_args())

    rand_name = args.pop("rand_name", False)
    if not args.get("zarr", ""):
        # get unique new filename
        if rand_name:
            args["zarr"] = args["imzml"][:-5] + str(uuid.uuid4()) + ".zarr"
        else:
            args["zarr"] = args["imzml"][:-5] + "zarr"

    if args.get("chunks", False):
        chunks = args["chunks"]
        for i, c in enumerate(chunks):
            try:
                chunks[i] = int(c)
                if chunks[i] == -1:
                    pass  # will be set to shape[i]
                elif chunks[i] < 1:
                    raise ValueError(f'"{c}" is not a valid chunk size')
            except ValueError as e:
                sys.exit(f"unable to parse chunk option ({args.chunk}): {e}")
        args["chunks"] = tuple(chunks)
    else:
        args["chunks"] = args.pop("auto_chunk", False)

    # compressor : None, default or some evaluated object
    if args.get("compressor") == "none":
        args["compressor"] = None
    elif args.get("compressor") != "default":
        # Zlib, GZip, BZ2, LZMA?, Blosc, Zstd, lz4, ZFPY, and a lot of others...
        import codecs

        try:
            args["compressor"] = eval(args["compressor"])
        except Exception as e:
            sys.exit(f"unable to evaluate compressor: {e}")

    max_size = args.pop("max_GB_size", None)
    if isinstance(max_size, int):
        if max_size < 1:
            raise ValueError("max_size should be at least 1GB")
        args["max_size"] = max_size * 2**30

    return args


def main():
    parser = get_parser()
    args = get_args(parser=parser)

    imzml = pathlib.Path(args.pop("imzml"))
    if not imzml.exists():
        raise ValueError(f"imzml file does not exist: {imzml}")

    ibd = imzml.with_suffix(".ibd")
    if not ibd.exists():
        raise ValueError(f"could not infer IBD file for {imzml}")

    dest = pathlib.Path(args.pop("zarr"))

    convert(imzml, ibd, dest, **args)


main()
