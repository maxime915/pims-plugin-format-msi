"checker: detects ImzML file formats"

from __future__ import annotations

import re

from pims.files.file import Path
from pims.formats.utils.abstract import CachedDataPath
from pims.formats.utils.checker import AbstractChecker

from .utils import get_imzml_pair

IMZML_UUID_ACCESSOR = 'IMS:1000080'


def check_uuid(imzml_path: Path, ibd_path: Path) -> bool:
    """check_uuid: verify that a pair of imzML & ibd files have the same UUID

    Args:
        imzml_path (Path): path to the imzML file
        ibd_path (Path): path tot the ibd file

    Returns:
        bool: the equivalence of the UUID

    """

    try:
        # get binary UUID
        with open(ibd_path, mode='rb') as ibd:
            ibd_uuid = ibd.read(16).hex()

        # get imzML UUID
        with open(imzml_path, mode='r', encoding='utf-8') as imzml:
            for line in imzml:
                # re.match caches pattern so no need to worry about compiling RE
                if re.match(rf'.*{IMZML_UUID_ACCESSOR}.*', line):

                    # find UUID
                    imzml_uuid_lst = re.findall(r'value="{?(.*)}?"', line)
                    if len(imzml_uuid_lst) != 1:
                        return False

                    # remove hyphens
                    imzml_uuid = imzml_uuid_lst[0].replace('-', '')

        return imzml_uuid == ibd_uuid

    except (Exception):  # TODO which exception could be thrown ?
        return False


class ImzMLChecker(AbstractChecker):
    "PIMS Checker for ImzML files"

    @classmethod
    def match(cls, pathlike: CachedDataPath) -> bool:
        """Whether the path is in this format or not."""

        pair = get_imzml_pair(pathlike.path)

        if pair is None:
            return False

        return check_uuid(*pair)
