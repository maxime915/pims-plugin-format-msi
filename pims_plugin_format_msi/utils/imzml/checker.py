"checker: detects ImzML file formats"

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lxml.etree import iterparse

from pims.formats.utils.abstract import CachedDataPath
from pims.formats.utils.checker import AbstractChecker
from .utils import get_imzml_pair

if TYPE_CHECKING:
    from pims.files.file import Path

MZML_PREFIX = '{http://psi.hupo.org/ms/mzml}'
IMZML_UUID_ACCESSOR = 'IMS:1000080'


def check_uuid(imzml_path: Path, ibd_path: Path) -> bool:
    """check_uuid: verify that a pair of imzML & ibd files have the same UUID

    Args:
        imzml_path (Path): path to the imzML file
        ibd_path (Path): path tot the ibd file

    Returns:
        bool: true if the UUIDs match
    """

    try:        
        # get binary UUID as a lowercase hex
        with open(ibd_path, mode='rb') as ibd:
            ibd_uuid = ibd.read(16).hex()

        # start parsing the document: get the root element
        _, root = next(iterparse(str(imzml_path), events=['start']))

        # search for the UUID tagged accessor
        key = f'.//{MZML_PREFIX}cvParam[@accession="{IMZML_UUID_ACCESSOR}"]'
        element = root.find(key)
        if element is None:
            raise ValueError('unable to find UUID')

        # strip optional curly braces
        imzml_uuid = element.get('value')
        if imzml_uuid[0] == '{':
            if imzml_uuid[-1] != '}':
                raise ValueError('unable to parse UUID')
            imzml_uuid = imzml_uuid[1:-1]

        # remove hyphens & convert to lowercase
        imzml_uuid = imzml_uuid.replace('-', '').lower()

        return imzml_uuid == ibd_uuid

    except StopIteration:
        # empty XML file
        return False
    except OSError:
        # file not found
        return False
    except ValueError:
        # parsing error
        return False
    except Exception as error:
        logging.error('unexpected exception caught', exc_info=error)
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
