# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""Create project directory trees, assign path variables, and get paths for
different types of files"""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

import os
from typing import List, Union
from pathlib import Path
import warnings

import ray
from pykanto.utils.compute import calc_chunks, get_chunks, print_dict, print_parallel_info, to_iterator, tqdmm
from pykanto.utils.read import read_json
from pykanto.utils.write import makedir, save_json

# ──── CLASSES ─────────────────────────────────────────────────────────────────


class ProjDirs():
    def __init__(self, PROJECT_DIR: Path, mkdir: bool = False):
        """
        Initialises a ProjDirs class, which is used to store a 
        project's file structure. This is required when constructing 
        a :class:`~pykanto.dataset.SongDataset` object and generally 
        useful to keep paths tidy and in the same location.

        Args:
            PROJECT_DIR (Path): Root directory of the project. 
            mkdir (bool, optional): Wether to create directories if they 
                don't already exist. Defaults to False.

        Examples:
            >>> PROJROOT = Path('user' / 'projects' / 'myproject')
            >>> DIRS = ProjDirs(PROJROOT, mkdir=False)
            >>> print(DIRS)
            Items held:
            PROJECT (/user/projects/myproject)
            DATA (/user/projects/myproject/data)
            RESOURCES (/user/projects/myproject/resources)
            REPORTS (/user/projects/myproject/reports)
            FIGURES (/user/projects/myproject/reports/figures)
        """
        if not isinstance(PROJECT_DIR, Path):
            raise TypeError(
                f"{PROJECT_DIR} must be an instance of (pathlib) Path")

        self.PROJECT = PROJECT_DIR
        self.DATA = PROJECT_DIR / "data"
        self.RESOURCES = PROJECT_DIR / "resources"
        self.REPORTS = PROJECT_DIR / "reports"
        self.FIGURES = self.REPORTS / "figures"

        if mkdir:
            for path in self.__dict__.values():
                makedir(path)

    def __str__(self) -> str:
        """
        Returns the class objects as a string.
        Returns:
            str: Pretty printed dictionary contents
        """
        return print_dict(self.__dict__)

    def append(self, new_attr: str, new_value: Path, mkdir: bool = False) -> None:
        """
        Appends a new attribute to the class instance.

        Args:
            new_attr (str): Name of the new attribute.
            new_value (Path): New directory.
            mkdir (bool, optional): Whether to create this directory
                if it doesn't already exist. Defaults to False.
        """
        if not new_attr.isupper():
            raise TypeError(f"{new_attr} must be uppercase")
        elif not isinstance(new_value, Path):
            raise TypeError(
                f"{new_value} must be an instance of (pathlib) Path")
        else:
            setattr(self, new_attr, new_value)
            if mkdir:
                makedir(new_value)

    def _deep_update_paths(self, OLD_PROJECT_DIR, NEW_PROJECT_DIR) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                self.__dict__[k] = change_data_loc(
                    v, OLD_PROJECT_DIR,  NEW_PROJECT_DIR)
            elif isinstance(v, list):
                self.__dict__[k] = [
                    change_data_loc(path, OLD_PROJECT_DIR, NEW_PROJECT_DIR)
                    for path in v]
            elif isinstance(v, dict):
                for k1, v1 in v.items():  # Level 1
                    if isinstance(v1, Path):
                        self.__dict__[k][k1] = change_data_loc(
                            v1, OLD_PROJECT_DIR, NEW_PROJECT_DIR)
                    elif isinstance(v1, dict):
                        for k2, v2 in v1.items():
                            self.__dict__[k][k1][k2] = change_data_loc(
                                v2, OLD_PROJECT_DIR, NEW_PROJECT_DIR)
                    elif k1 == 'already_checked':
                        continue
                    else:
                        print(k1, v1)
                        raise TypeError(
                            'Dictionary values must be either '
                            'of type Path or a second dictionary '
                            'with values of type Path')
            else:
                print(k, v)
                raise TypeError(
                    'Dictionary values must be of types Path | List | Dict')

    def update_json_locs(
            self, overwrite: bool = False,
            ignore_checks: bool = False) -> None:
        """
        Updates the 'wav_loc' field in JSON metadata files for a given project.
        This is useful if you have moved your data to a location different to
        where it was first generated. It will fix broken links to the .wav
        files, provided that the :class:`~pykanto.utils.paths.ProjDirs` object
        has a 'WAVFILES' attribute pointing to a valid directory containing
        'WAV' and 'JSON' subdirectories.

        Args:
            overwrite (bool, optional): Whether to force change paths 
                even if the current ones work. Defaults to False.
            ignore_checks (bool, optional): Wether to check that wav and 
                JSON files coincide. Useful if you just want to change JSONS in
                a different location to where the rest of the data are.
        """

        if not hasattr(self, 'WAVFILES'):
            raise KeyError(
                "This ProjDirs object does not have a WAVFILES attribute. "
                "You can append it like so: "
                "`DIRS.append('WAVFILES', Path(...))`")

        if not self.WAVFILES.is_dir():
            raise FileNotFoundError(f"{self.WAVFILES} does not exist.")

        WAV_LIST = sorted(list((self.WAVFILES / "WAV").glob("*.wav")))
        JSON_LIST = sorted(list((self.WAVFILES / "JSON").glob("*.JSON")))

        if not len(WAV_LIST) and not ignore_checks:
            raise FileNotFoundError(
                f'There are no .wav files in {self.WAVFILES / "WAV"}')
        if len(WAV_LIST) != len(JSON_LIST) and not ignore_checks:
            raise KeyError(
                "There is an unequal number of .wav and .json "
                f"files in {self.WAVFILES}")

        # Check that file can be read & wav_loc needs to be changed
        try:
            jf = read_json(JSON_LIST[0])
        except:
            raise FileNotFoundError(
                f"{JSON_LIST[0]} does not exist or is empty.")

        wavloc = Path(jf['wav_loc'])
        if wavloc.exists() and overwrite is False:
            warnings.warn(
                f'{wavloc} exists: no need to update paths. '
                'You can force update by setting `overwrite = True`.')
            return

        def change_wav_loc(file):
            try:
                jf = read_json(file)
            except:
                raise FileNotFoundError(f"{file} does not exist or is empty.")

            newloc = self.WAVFILES / "WAV" / Path(jf['wav_loc']).name
            if Path(jf['wav_loc']) == newloc:
                return
            else:
                try:
                    jf['wav_loc'] = str(newloc)
                    save_json(jf, file)
                except:
                    raise IndexError(f"Could not save {file}")

        def batch_change_wav_loc(files: List[Path]) -> None:
            if isinstance(files, list):
                for file in files:
                    change_wav_loc(file)
            else:
                change_wav_loc(files)
        kk = JSON_LIST[0]
        change_wav_loc(JSON_LIST[0])

        # Run in paralell
        b_change_wav_loc_r = ray.remote(batch_change_wav_loc)
        chunk_info = calc_chunks(len(JSON_LIST), verbose=True)
        chunk_length, n_chunks = chunk_info[3], chunk_info[2]
        chunks = get_chunks(JSON_LIST, chunk_length)
        print_parallel_info(
            len(JSON_LIST),
            'individual IDs', n_chunks, chunk_length)

        # Distribute with ray
        obj_ids = [b_change_wav_loc_r.remote(i) for i in chunks]
        pbar = {'desc': "Projecting and clustering vocalisations",
                'total': n_chunks}
        for obj_id in tqdmm(to_iterator(obj_ids), **pbar):
            pass

        print('Done')


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────


def change_data_loc(
        DIR: Path, PROJECT_DIR: Path, NEW_PROJECT_DIR: Path) -> Path:
    """
    Updates the location of the parent directories of a project, including 
    the project name, for a given path. Used when the location of a dataset changes 
    (e.g if transferring a project to a new machine).

    Args:
        DIR (Path): Path to update
        PROJECT_DIR ([type]): Old -broken- project directory.
        NEW_PROJECT_DIR (Path): New working project directory.

    Returns:
        Path: Updated path.
    """
    index = DIR.parts.index(PROJECT_DIR.name)
    new_path = NEW_PROJECT_DIR.joinpath(*DIR.parts[index+1:])
    return new_path


def get_wav_filepaths(ORIGIN_DIR:
                      Union[str, Path]) -> List[Path]:
    """
    Get a list of wav files in a directory, including subdirectories, if
    there is a corresponding .xml file with segmentation information. Works for
    segmentation metadata from .xml files output by Sonic Visualiser.

    Args:
        ORIGIN_DIR (PosixPath): Directory to search
    """
    file_list: List[Path] = []
    for root, _, files in os.walk(str(ORIGIN_DIR)):
        for file in files:
            if ((file.endswith(".wav") or file.endswith(".WAV"))
                    and str(file.rsplit('.', 1)[0] + ".xml") in files):
                file_list.append(Path(root) / file)
    if len(file_list) == 0:
        raise FileNotFoundError(
            "There are no .wav files with matching .xml in this directory")
    else:
        print(f"Found {len(file_list)} .wav files with matching .xml files")
        return file_list


# NOTE: move this to custom
def get_xml_filepaths(ORIGIN_DIR:
                      Union[str, Path]) -> List[Path]:
    file_list: List[Path] = []
    for root, _, files in os.walk(str(ORIGIN_DIR)):
        xml_files = [file for file in files if file.endswith('.xml')]
        for file in xml_files:
            file_list.append(Path(root) / file)

    if len(file_list) == 0:
        raise FileNotFoundError("There are no .xml files in this directory")
    else:
        nfolders = len(set([path.parent.name for path in file_list]))
        print(
            f"Found {len(file_list)} .xml files in "
            f"{nfolders} folders")
        return file_list


def link_project_data(origin: os.PathLike, project_data_dir: Path) -> None:
    """
    Creates a symlink from a project's data folder (not under version control)
    to the directory where the data lives (e.g. on an external HDD).

    Args: origin (os.PathLike): Path to the directory where the data are.
        project_data_dir (Path): A project's data folder to link with 'origin'.

    Raises: ValueError: The 'project_data_dir' already contains data or is a
        symlink.
    """

    if not os.path.isdir(project_data_dir):
        os.symlink(origin, project_data_dir, target_is_directory=True)
        print("Symbolic link created successfully.")
    elif len(os.listdir(project_data_dir)) == 0:
        os.rmdir(project_data_dir)
        os.symlink(origin, project_data_dir, target_is_directory=True)
        print("Symbolic link created successfully.")
    else:
        raise ValueError('The destination directory is not empty.')
