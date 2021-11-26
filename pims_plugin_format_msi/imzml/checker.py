"checker: detects ImzML file formats"

from __future__ import annotations

import warnings

from pims.formats.utils.abstract import CachedDataPath
from pims.formats.utils.checker import AbstractChecker


class ImzMLChecker(AbstractChecker):
    "PIMS Checker for ImzML files"

    @classmethod
    def match(cls, pathlike: CachedDataPath) -> bool:
        """Whether the path is in this format or not."""

        # requires a directory with .imzML and .ibd
        if not pathlike.path.is_dir():
            return False

        # get a list of files in the directory
        files = [f for f in pathlike.path.iterdir() if f.is_file()]
        files_as_str = [str(f).lower() for f in files]

        # get all files ending in .imzML (case insensitive)
        imzml_files = [f for f in files if f.suffix.lower() == '.imzml']

        if not imzml_files:
            return False

        if len(imzml_files) > 1:
            warnings.warn('More than 1 imzML file were uploaded')

        for imzml in imzml_files:
            target = imzml.with_suffix('.ibd')
            target_as_str = str(target).lower()

            # check if target exists (case sensitive)
            if target in files:
                return True

            # check for case insensitive file
            if target_as_str not in files_as_str:
                continue
            idx = files_as_str.index(target_as_str)

            # check for name without extension
            if imzml.with_suffix('') == files[idx].with_suffix(''):
                return True

            # TODO allow case mis-match ?
            # maybe check the UUID in the file if needed
            #   - easy to get in the .ibd
            #   - need to parse imzml ?

        return False
