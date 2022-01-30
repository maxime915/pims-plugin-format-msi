"utils: utility function related to ImzML"
from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from pims.files.file import Path


def get_imzml_pair(path: Path) -> Optional[Tuple[Path, Path]]:
    """get the pair of '.imzml' and '.ibd' files in the archive

    Args:
        path (Path): Folder containing both files

    Returns:
        Optional[Tuple[Path, Path]]: The pair of files, or None
    """

    if not path.is_dir():
        return None

    # get a list of files in the directory
    files = [f for f in path.iterdir() if f.is_file()]
    files_as_str = [str(f).lower() for f in files]

    # get all files ending in .imzML (case insensitive)
    imzml_files = [f for f in files if f.suffix.lower() == '.imzml']

    if not imzml_files:
        return None

    if len(imzml_files) > 1:
        warnings.warn('Found at least 2 imzML files')

    for imzml in imzml_files:
        target = imzml.with_suffix('.ibd')
        target_as_str = str(target).lower()

        # check if target exists (case sensitive)
        if target in files:
            return (imzml, target)

        try:
            # check for case insensitive file
            idx = files_as_str.index(target_as_str)

            # check for name without extension
            if imzml.with_suffix('') == files[idx].with_suffix(''):
                return (imzml, target)

        except ValueError:  # if target_as_str is not in files_as_str
            continue

    return None
