# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""Create project directory trees, assign path variables, and get paths for
different types of files"""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

import os
import warnings
from pathlib import Path
from typing import List, Tuple

import pkg_resources
import ray
from pykanto.utils.compute import (
    calc_chunks,
    get_chunks,
    print_dict,
    print_parallel_info,
    to_iterator,
    with_pbar,
)
from pykanto.utils.io import makedir, read_json, save_json
from pykanto.utils.types import ValidDirs

# ──── CLASSES ─────────────────────────────────────────────────────────────────


class ProjDirs:
    """
    Initialises a ProjDirs class, which is used to store a
    project's file structure. This is required when constructing
    a :class:`~pykanto.dataset.KantoData` object and generally
    useful to keep paths tidy and in the same location.

    Args:
        PROJECT (Path): Root directory of the project.
        RAW_DATA (Path): (Immutable) location of the raw data to be used in
            this project.
        DATASET_ID (str): Name of the dataset.
        mkdir (bool, optional): Wether to create directories if they
            don't already exist. Defaults to False.

    Attributes:
        PROJECT (Path): Root directory of the project.
        DATA (Path): Directory for project data.
        RAW_DATA (Path): (Immutable) location of the raw data to be used in
            this project.
        SEGMENTED (Path): Directory for segmented audio data.
        SPECTROGRAMS (Path): Directory for project spectrograms.
        RESOURCES (Path): Directory for project resources.
        REPORTS (Path): Directory for project reports.
        FIGURES (Path): Directory for project figures.
        DATASET (Path): Directory for project datasets.
        DATASET_ID (str): Name of the dataset.

    Examples:
        >>> from pathlib import Path
        >>> from pykanto.utils.paths import ProjDirs
        >>> DATASET_ID = "BIGBIRD"
        >>> PROJROOT = Path('home' / 'user' / 'projects' / 'myproject')
        >>> RAW_DATA= Path('bigexternaldrive' / 'fieldrecordings')
        >>> DIRS = ProjDirs(PROJROOT, RAW_DATA, DATASET_ID, mkdir=True)
        ... 📁 project
        ... ├── 📁 data
        ... │   ├── 📁 datasets
        ... │   │   └── 📁 <DATASET_ID>
        ... │   │       ├── <DATASET_ID>.db
        ... │   │       └── 📁 spectrograms
        ... |   ├── 📁 RAW_DATA
        ... │   │   └── 📁 <DATASET_ID>
        ... │   └── 📁 segmented
        ... │       └── 📁 <lowercase name of RAW_DATA>
        ... ├── 📁 resources
        ... ├── 📁 reports
        ... │   └── 📁 figures
        ... └── <other project files>
    """

    def __init__(
        self,
        PROJECT: Path,
        RAW_DATA: Path,
        DATASET_ID: str,
        mkdir: bool = False,
    ):

        # Type check input paths
        d = ValidDirs(PROJECT, RAW_DATA, DATASET_ID)

        # Define project directories
        self.PROJECT = d.PROJECT
        self.DATA = d.PROJECT / "data"
        self.RAW_DATA = d.RAW_DATA
        self.SEGMENTED = self.DATA / "segmented" / d.RAW_DATA.name.lower()

        self.RESOURCES = d.PROJECT / "resources"
        self.REPORTS = d.PROJECT / "reports"
        self.FIGURES = self.REPORTS / "figures"

        self.DATASET = self.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
        self.DATASET_ID = DATASET_ID
        self.SPECTROGRAMS = self.DATA / "datasets" / DATASET_ID / "spectrograms"

        # Create them if needed
        if mkdir:
            for attr in self.__dict__.values():
                if isinstance(attr, Path):
                    makedir(attr)

    def __str__(self) -> str:
        """
        Returns the class objects as a string.
        Returns:
            str: Pretty printed dictionary contents
        """
        return print_dict(self.__dict__)

    def append(
        self, new_attr: str, new_value: Path, mkdir: bool = False
    ) -> None:
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
                f"{new_value} must be an instance of (pathlib) Path"
            )
        else:
            setattr(self, new_attr, new_value)
            if mkdir:
                makedir(new_value)

    def _deep_update_paths(self, OLD_PROJECT, NEW_PROJECT) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                self.__dict__[k] = change_data_loc(v, OLD_PROJECT, NEW_PROJECT)
            elif isinstance(v, list):
                self.__dict__[k] = [
                    change_data_loc(path, OLD_PROJECT, NEW_PROJECT)
                    for path in v
                ]
            elif isinstance(v, dict):
                for k1, v1 in v.items():  # Level 1
                    if isinstance(v1, Path):
                        self.__dict__[k][k1] = change_data_loc(
                            v1, OLD_PROJECT, NEW_PROJECT
                        )
                    elif isinstance(v1, dict):
                        for k2, v2 in v1.items():
                            self.__dict__[k][k1][k2] = change_data_loc(
                                v2, OLD_PROJECT, NEW_PROJECT
                            )
                    elif k1 == "already_checked":
                        continue
                    else:
                        print(k1, v1)
                        raise TypeError(
                            "Dictionary values must be either "
                            "of type Path or a second dictionary "
                            "with values of type Path"
                        )
            else:
                print(k, v)
                raise TypeError(
                    "Dictionary values must be of types Path | List | Dict"
                )

    def update_json_locs(
        self, overwrite: bool = False, ignore_checks: bool = False
    ) -> None:
        """
        Updates the `wav_file` field in JSON metadata files for a given project.
        This is useful if you have moved your data to a new location. It will
        fix broken links to the .wav files, provided that the
        :class:`~pykanto.utils.paths.ProjDirs` object has a `SEGMENTED`
        attribute pointing to a valid directory containing `/WAV` and `/JSON`
        subdirectories.

        Args:
            overwrite (bool, optional): Whether to force change paths
                even if the current ones work. Defaults to False.
            ignore_checks (bool, optional): Wether to check that wav and
                JSON files coincide. Useful if you just want to change JSONS in
                a different location to where the rest of the data are.
                Defaults to False.
        """
        # TODO@nilomr: #11 Path to spectrogram .npy files breaks if dataset changes location
        if not hasattr(self, "SEGMENTED"):
            raise KeyError(
                "This ProjDirs object does not have a SEGMENTED attribute. "
                "You can append it like so: "
                "`DIRS.append('SEGMENTED', Path(...))`"
            )

        if not self.SEGMENTED.is_dir():
            raise FileNotFoundError(f"{self.SEGMENTED} does not exist.")

        WAV_LIST = sorted(list((self.SEGMENTED / "WAV").glob("*.wav")))
        JSON_LIST = sorted(list((self.SEGMENTED / "JSON").glob("*.JSON")))

        if not len(WAV_LIST) and not ignore_checks:
            raise FileNotFoundError(
                f'There are no .wav files in {self.SEGMENTED / "WAV"}'
                " will not look for JSON files."
            )
        if len(WAV_LIST) != len(JSON_LIST) and not ignore_checks:
            raise KeyError(
                "There is an unequal number of .wav and .json "
                f"files in {self.SEGMENTED}"
            )

        # Check that file can be read & wav_file needs to be changed
        try:
            jf = read_json(JSON_LIST[0])
        except:
            raise FileNotFoundError(
                f"{JSON_LIST[0]} does not exist or is empty."
            )

        wavloc = Path(jf["wav_file"])
        print(wavloc)

        if wavloc.exists() and overwrite is False:
            raise FileExistsError(
                f"{wavloc} exists: no need to update paths. "
                "You can force update by setting `overwrite = True`."
            )

        def change_wav_file_field(file):
            try:
                jf = read_json(file)
            except:
                raise FileNotFoundError(f"{file} does not exist or is empty.")

            newloc = self.SEGMENTED / "WAV" / Path(jf["wav_file"]).name
            if Path(jf["wav_file"]) == newloc:
                return
            else:
                try:
                    jf["wav_file"] = str(newloc)
                    save_json(jf, file)
                except:
                    raise IndexError(f"Could not save {file}")

        def batch_change_wav_file_field(files: List[Path]) -> None:
            if isinstance(files, list):
                for file in files:
                    change_wav_file_field(file)
            else:
                change_wav_file_field(files)

        change_wav_file_field(JSON_LIST[0])

        # Run in paralell
        b_change_wav_file_r = ray.remote(batch_change_wav_file_field)
        chunk_info = calc_chunks(len(JSON_LIST), verbose=True)
        chunk_length, n_chunks = chunk_info[3], chunk_info[2]
        chunks = get_chunks(JSON_LIST, chunk_length)
        print_parallel_info(
            len(JSON_LIST), "individual IDs", n_chunks, chunk_length
        )

        # Distribute with ray
        obj_ids = [b_change_wav_file_r.remote(i) for i in chunks]
        pbar = {
            "desc": "Updating the `wav_file` field in JSON metadata files.",
            "total": n_chunks,
        }
        for obj_id in with_pbar(to_iterator(obj_ids), **pbar):
            pass

        print("Done")


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────


def get_wavs_w_annotation(
    wav_filepaths: List[Path], annotation_paths: List[Path]
) -> List[Tuple[Path, Path]]:
    """
    Returns a list of tuples containing [0] paths to wavfiles for which there is
    an annotation file and [1] paths to its annotation file. Assumes that wav
    and paths to the annotation files share the same file name and only their
    file extension changes.

    Args:
        wav_filepaths (List[Path]): List of paths to wav files.
        annotation_paths (List[Path]): List of paths to annotation files.

    Returns:
        List[Tuple[Path, Path]]: Filtered list.
    """

    wavcase = wav_filepaths[0].suffix

    matches = [
        (wav_filepaths[0].parent / f"{file.stem}{wavcase}", file)
        for file in annotation_paths
        if wav_filepaths[0].parent / f"{file.stem}{wavcase}" in wav_filepaths
    ]

    return matches


def change_data_loc(DIR: Path, PROJECT: Path, NEW_PROJECT: Path) -> Path:
    """
    Updates the location of the parent directories of a project, including the
    project name, for a given path. Used when the location of a dataset changes
    (e.g if transferring a project to a new machine).

    Args:
        DIR (Path): Path to update
        PROJECT ([type]): Old -broken- project directory.
        NEW_PROJECT (Path): New working project directory.

    Returns:
        Path: Updated path.
    """
    index = DIR.parts.index(PROJECT.name)
    new_path = NEW_PROJECT.joinpath(*DIR.parts[index + 1 :])
    return new_path


def get_file_paths(
    root_dir: Path, extensions: List[str], verbose: bool = False
) -> List[Path]:
    """
    Returns paths to files with given extension found recursively within a
    directory.

    Args:
        root_dir (Path): Root directory to search recursively.
        extensions (List[str]): File extensions to look for (e.g., .wav)

    Raises:
        FileNotFoundError: No files found.

    Returns:
        List[Path]: List with path to files.
    """
    file_list: List[Path] = []
    ext = "".join(
        [
            f"{x} and/or "
            if i != len(extensions) - 1 and len(extensions) > 1
            else f"{x}"
            for i, x in enumerate(extensions)
        ]
    )

    for root, _, files in os.walk(str(root_dir)):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_list.append(Path(root) / file)
    if len(file_list) == 0:
        raise FileNotFoundError(
            f"There are no {ext} files in directory {root_dir}"
        )
    else:
        if verbose:
            print(f"Found {len(file_list)} {ext} files in {root_dir}")
    return file_list


def link_project_data(origin: os.PathLike, project_data_dir: Path) -> None:
    """
    Creates a symlink from a project's data folder (not under version control)
    to the directory where the data lives (e.g. on an external HDD).

    Args:
        origin (os.PathLike): Path to the directory containing your 'raw'
            data folder.
        project_data_dir (Path): A project's data folder to link with 'origin'.

    Note:
        This will work in unix-like systems but might cause problems in Windows.
        See `how to enable symlinks in
        Windows <https://csatlas.com/python-create-symlink/#windows>`_

    Raises:
        ValueError: The 'project_data_dir' already contains data or is a
        symlink.
        FileExistsError: File exists; your target folder already exists.
    """

    if not os.path.isdir(project_data_dir):
        os.symlink(origin, project_data_dir, target_is_directory=True)
        print("Symbolic link created successfully.")
    elif len(os.listdir(project_data_dir)) == 0:
        os.rmdir(project_data_dir)
        os.symlink(origin, project_data_dir, target_is_directory=True)
        print("Symbolic link created successfully.")
    else:
        warnings.warn(
            "link_project_data() failed: "
            "the destination directory is not empty."
        )


def pykanto_data(dataset: str = "GREAT_TIT") -> ProjDirs:
    """
    Loads pykanto's sample datasets. These are minimal data examples intended
    for testing and tutorials.

    Args:
        dataset (str, optional): Dataset name, one of ["STORM-PETREL",
            "BENGALESE_FINCH", "GREAT_TIT", "AM"]. Defaults to "GREAT_TIT".

    Returns:
        ProjDirs: An object with paths to data directories that can then be used
        to create a dataset.
    """
    dfolder = "segmented" if dataset == "GREAT_TIT" else "raw"
    DATA_PATH = Path(pkg_resources.resource_filename("pykanto", "data"))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = (
        DATA_PATH
        / dfolder
        / (dataset.lower() if dataset == "GREAT_TIT" else dataset)
    )
    DIRS = ProjDirs(
        PROJECT, RAW_DATA, dataset, mkdir=True if dataset != "AM" else False
    )
    return DIRS
