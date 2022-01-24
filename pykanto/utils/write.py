
from __future__ import annotations
import os.path
import tarfile
import re
import json
from _ctypes import PyObj_FromPtr
import os
from pathlib import Path
import sys
from typing import Dict, List
from tqdm import tqdm
import shutil
import ujson


def makedir(DIR: Path, return_path: bool = True) -> Path:
    """
    Make a safely nested directory. Returns the Path object by default. Modified
    from `code`_ by Tim Sainburg.
    .. _code: https://github.com/timsainb/src.avgn_paper/blob/vizmerge/src.avgn/utils/paths.py

    Args:
        DIR (Path): Path to be created. return_path (bool, optional): Whether to
        return the path. Defaults to True.

    Raises:
        TypeError: Wrong argument type to 'DIR'

    Returns:
        Path: Path to file or directory.
    """

    if not isinstance(DIR, Path):
        raise TypeError("Wrong argument type to 'DIR'")

    # If this is a file
    if len(DIR.suffix) > 0:
        DIR.parent.mkdir(parents=True, exist_ok=True)
    else:
        try:
            DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno != 17:
                print("Error:", e)
    if return_path:
        return DIR


def copy_xml_files(file_list: List[Path], dest_dir: Path) -> None:
    """
    Copies a list of files to `dest_dir / file.parent.name / file.name`

    Args:
        file_list (List[Path]): List of files to be copied.
        dest_dir (Path): Path to destination folder, will create it 
            if doesn't exist.
    """
    file: Path
    for file in tqdm(file_list, desc="Copying files", leave=True,
                     file=sys.stdout):
        dest_file: Path = dest_dir / file.parent.name / file.name
        makedir(dest_file)
        shutil.copyfile(file, dest_file, follow_symlinks=True)
    print(f"Done copying {len(file_list)} files to {dest_dir}")


class NoIndent(object):
    """ Value wrapper. """

    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    """ Encoder for json that allows for a NoIndent wrapper on lists

    Based upon the StackOverflow answer: https://stackoverflow.com/a/13252112/200663
    From Tim Sainburg's avgn repository.

    Extends:
        json.JSONEncoder

    Variables:
        regex {[type]} -- [description]
    """

    FORMAT_SPEC = "@@{}@@"
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get("sort_keys", None)
        super(NoIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, NoIndent)
            else super(NoIndentEncoder, self).default(obj)
        )

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(NoIndentEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(
                no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), json_obj_repr
            )

        return json_repr


def save_json(json_object: Dict, json_loc: Path) -> Dict:
    """
    Saves a .json file using ujson.

    Args:
        json_loc (Path): Path to json file.

    Returns:
        Dict: Json file as a dictionary.
    """
    with open(json_loc, 'w', encoding='utf-8') as f:
        ujson.dump(json_object, f, ensure_ascii=False, indent=4)


def make_tarfile(source_dir: Path, output_filename: Path) -> None:
    """
    Makes a tarfile from a given directory. From George V. Reilly in  
    `stackoverflow <https://stackoverflow.com/a/17081026>`_.

    Args:
        source_dir (Path): Directory to tar 
        output_filename (Path): Name of output file (e.g. file.tar.gz).
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))