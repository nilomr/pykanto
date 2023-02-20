# ─── DESCRIPTION ──────────────────────────────────────────────────────────────
"""
Build the main dataset class and its methods to visualise, segment and label 
animal vocalisations.
"""

# ─── LIBRARIES ────────────────────────────────────────────────────────────────

from __future__ import annotations

import copy
import inspect
import pickle
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from random import sample
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from bokeh.palettes import Set3_12

import pykanto.plot as kplot
from pykanto.app.data import prepare_datasource_parallel
from pykanto.parameters import Parameters
from pykanto.signal.cluster import reduce_and_cluster_parallel
from pykanto.signal.segment import (
    drop_zero_len_units,
    segment_song_into_units_parallel,
)
from pykanto.signal.spectrogram import (
    _save_melspectrogram_parallel,
    get_indv_units_parallel,
)
from pykanto.utils.compute import (
    dictlist_to_dict,
    flatten_list,
    print_dict,
    timing,
    with_pbar,
)
from pykanto.utils.io import (
    _get_json,
    _get_json_parallel,
    makedir,
    save_to_jsons,
)
from pykanto.utils.paths import ProjDirs, get_file_paths, get_wavs_w_annotation

# ─── CLASSES ──────────────────────────────────────────────────────────────────


class KantoData:
    """
    Main dataset class. See `__init__` docstring.
    """

    # TODO@nilomr #10 Refactor private methods in KantoData class
    def __init__(
        self,
        DIRS: ProjDirs,
        parameters: None | Parameters = None,
        random_subset: None | int = None,
        overwrite_dataset: bool = False,
        overwrite_data: bool = False,
    ) -> None:
        """
        Instantiates the main dataset class.

        Note:
            If your dataset contains noise samples (useful when training a
            neural network), these should be labelled as 'NOISE' in the
            corresponding json file.
            i.e., `json_file["label"] == 'NOISE'`.

        Args:
            DIRS (ProjDirs): Project directory structure. Must contain
                a 'SEGMENTED' attribute pointing to a directory that contains
                the segmented data organised in two folders: 'WAV' and
                'JSON' for audio and metadata, respectively.
            parameters (Parameters, optional): Parameters for dataset.
                Defaults to None. See :class:`~pykanto.parameters.Parameters`.
            random_subset (None, optional): Size of random subset for
                testing. Defaults to None.
            overwrite_dataset (bool, optional): Whether to overwrite dataset if
                it already exists. Defaults to False.
            overwrite_data (bool, optional): Whether to overwrite spectrogram
                files if they already exist. Defaults to False.

        Raises:
            FileExistsError: DATASET_ID already exists. You can overwrite it
                by setting `overwrite_dataset=True`
        """

        # Get or set parameters
        if parameters is None:
            warnings.warn(
                "You have not provided parameters for this dataset."
                "Setting default parameters - these will almost certainly be "
                "inadequate for your dataset."
            )
            self.parameters = Parameters()
        else:
            self.parameters = parameters

        # Get dataset identifier and location
        self.DIRS = copy.deepcopy(DIRS)
        self.DATASET_ID = DIRS.DATASET_ID

        # Throw error if dataset already exists
        if self.DIRS.DATASET.is_file() and overwrite_dataset is False:
            raise FileExistsError(
                f"{self.DIRS.DATASET} already exists. You can overwrite "
                "it by setting `overwrite_dataset=True`"
            )

        # Build dataset
        self._get_wav_json_filedirs(random_subset=random_subset)
        # TODO: perform a field check for the json files and provide
        # feedback on any missing/wrong fields, or their value types.
        self._load_metadata()
        self._get_unique_ids()
        self._create_df()
        self._compute_melspectrograms(
            dereverb=self.parameters.dereverb, overwrite_data=overwrite_data
        )

        # Save current state of dataset
        self.save_to_disk(verbose=self.parameters.verbose)
        print("Done")

    def __str__(self) -> str:
        """
        Returns the class object as a string.
        Returns:
            str: Pretty printed dictionary contents
        Examples:
            >>> dataset = KantoData(...)
            >>> print(dataset)
            Items held:
            minmax_values: {'max_freq': 10249} + 3 other entries.
            vocalisations: pd.DataFrame with length 20 and columns
        """
        return print_dict(self.__dict__)

    # ──────────────────────────────────────────────────────────────────────────
    # KantoData: Private methods

    @timing
    def _get_wav_json_filedirs(self, random_subset: None | int = None) -> None:
        """
        Get paths of wav and json files for this dataset.

        Args:
            random_subset (int, optional): Get a random subset without
                repetition. Defaults to None.

        Warning:
            Will drop any .wav for which there is no .json!

        """
        if not hasattr(self.DIRS, "SEGMENTED"):
            raise KeyError(
                "The ProjDirs object you used to initialise "
                "this KantoData object does not have a SEGMENTED attribute. "
                "Search the docs for KantoData for more information."
            )

        # TODO: read both lower and uppercase wav and json extensions.
        # Load and check file paths:
        wav_filepaths, json_filepaths = [
            get_file_paths(self.DIRS.SEGMENTED, [ext])
            for ext in [".wav", ".JSON"]
        ]
        file_tuples = get_wavs_w_annotation(wav_filepaths, json_filepaths)

        nfiles = len(file_tuples)
        if nfiles < len(wav_filepaths):
            ndrop = len(wav_filepaths) - nfiles
            warnings.warn(
                "There is an unequal number of matching .wav and .json "
                f"files in {self.DIRS.SEGMENTED}. "
                f"Keeping only those that match: dropped {ndrop}"
            )

        # Subset as per parameters if possible
        if self.parameters.subset:
            if nfiles == 1 and self.parameters.subset != (0, nfiles):
                warnings.warn("Cannot subset: there is only one file.")
                pass

            elif (
                self.parameters.subset[0] < nfiles
                and self.parameters.subset[1] <= nfiles
            ):
                file_tuples = file_tuples[
                    self.parameters.subset[0] : self.parameters.subset[1]
                ]

            else:
                raise IndexError(
                    "List index out of range: the provided "
                    f"'subset' parameter {self.parameters.subset} is ouside "
                    f"the range of the dataset (0, {nfiles})."
                )

        if random_subset:
            if random_subset > nfiles:
                random_subset = nfiles
                warnings.warn(
                    "The 'random_subset' you specified is larger "
                    "than the dataset: setting to the length of the dataset."
                )
            file_tuples = sample(file_tuples, random_subset)

        # Add paths to self
        self._wavfiles, self._jsonfiles = [
            sorted(list(i)) for i in zip(*file_tuples)
        ]

    @timing
    def _load_metadata(self) -> None:
        """
        Loads json metadata in sequence if there are < 100 items, in parallel
        chunks of length chunk_len if there are > 1000 items.
        """

        n_jsonfiles = len(self._jsonfiles)
        if n_jsonfiles < 100:
            jsons = [
                _get_json(json)
                for json in with_pbar(
                    self._jsonfiles,
                    desc="Loading JSON files",
                    total=n_jsonfiles,
                )
            ]
        else:
            jsons = _get_json_parallel(
                self._jsonfiles, verbose=self.parameters.verbose
            )
        self.metadata = {Path(json["wav_file"]).stem: json for json in jsons}

        # Match wav_file field with actual wav_file location for this dataset
        # Partially fixes nilomr/pykanto#12
        for wavfile in self._wavfiles:
            self.metadata[wavfile.stem]["wav_file"] = wavfile.as_posix()

    def _get_unique_ids(self) -> None:
        """
        Adds a 'unique_ID' attribute holding an array with unique IDs in the
        dataset.
        """
        self.ID = np.array([value["ID"] for _, value in self.metadata.items()])
        self.unique_ID = np.unique(self.ID)

    def _create_df(self) -> None:
        """
        Creates pandas DataFrames from the metadata dictionary and adds them to
            the KantoData instance.

        Warning:
            Removes onset/offset pairs which (under current spectrogram
            parameter combinations) would result in a unit of length zero.
        """
        data_dict = {}
        for key, file in self.metadata.items():
            data = file

            if all(e in file for e in ["onsets", "offsets"]):
                data["onsets"], data["offsets"] = drop_zero_len_units(
                    self, np.array(file["onsets"]), np.array(file["offsets"])
                )
            data_dict[key] = data

        pathcols = ["source_wav", "wav_file"]
        self.data = pd.DataFrame.from_dict(data_dict, orient="index").drop(
            columns=pathcols
        )
        self.files = pd.DataFrame.from_dict(
            data_dict, orient="index", columns=pathcols
        )

        # Get minimum and maximum frequencies and durations in song dataset
        self.minmax_values = {
            "max_freq": max(self.data["upper_freq"]),
            "min_freq": min(self.data["lower_freq"]),
            "max_duration": max(self.data["length_s"]),
            "min_duration": min(self.data["length_s"]),
        }

    @timing
    def _compute_melspectrograms(
        self, overwrite_data: bool = False, **kwargs
    ) -> None:
        """
        Compute melspectrograms and add their location to the KantoData object.
        It applies dereverberation to and bandpasses spectrogram data by
        default.

        Args:
            overwrite_data (bool): Whether to overwrite any spectrogram files
            that already exist for this dataset.
            **kwargs: Passed to
                :func:`~pykanto.signal.spectrogram.save_melspectrogram`.
        """

        def _spec_exists(key):  # TODO: refactor
            file = self.metadata[key]["wav_file"]
            ID = self.metadata[key]["ID"]
            path = self.DIRS.SPECTROGRAMS / ID / (Path(file).stem + ".npy")
            if path.exists():
                return {Path(file).stem: path}

        # Check if spectrograms already exist for any keys:
        existing = [_spec_exists(key) for key in self.data.index]
        existing = dictlist_to_dict([x for x in existing if x is not None])

        # Get new keys (all keys if overwrite_data=True)
        if overwrite_data:
            new_keys = self.data.index.tolist()
        else:
            new_keys = [key for key in self.data.index if key not in existing]

        # Compute melspectrograms
        specs = (
            _save_melspectrogram_parallel(self, new_keys, **kwargs)
            if new_keys
            else {}
        )
        specs = {
            **specs,
            **(existing if (existing and not overwrite_data) else {}),
        }
        self.files["spectrogram"] = pd.Series(specs, dtype=object)

    # ──────────────────────────────────────────────────────────────────────────
    # KantoData: Public methods

    def plot_summary(
        self, nbins: int = 50, variable: str = "frequency"
    ) -> None:
        """
        Plots a histogram + kernel density estimate of the frequency
        distribution of vocalisation duration and frequencies.

        Note:
            Durations and frequencies come from bounding boxes,
            not vocalisations. This function, along with
            :func:`~pykanto.dataset.show_extreme_songs`, is useful to spot
            any outliers, and to quickly explore the full range of data.

        Args:
            nbins (int, optional): Number of bins in histogram. Defaults to 50.
            variable (str, optional): One of 'frequency', 'duration',
                'sample_size', 'all'. Defaults to 'frequency'.

        Raises:
            ValueError: `variable` must be one of
                ['frequency', 'duration', 'sample_size', 'all']
        """

        kplot.build_plot_summary(self, nbins=nbins, variable=variable)

    def plot(self, key: str, segmented: bool = False, **kwargs) -> None:
        """
        Plot an spectrogram from the dataset.

        Args:
            key (str): Key of the spectrogram.
            segmented (bool, optional): Whether to overlay onset/offset
                information. Defaults to False.
            kwargs: passed to :func:`~pykanto.plot.melspectrogram`

        Examples:
            Plot the first 10 specrograms in the vocalisations dataframe:
            >>> for spec in dataset.data.index[:10]:
            ...     dataset.plot(spec)

        """
        title = kwargs.pop("title") if "title" in kwargs else Path(key).stem

        if segmented:
            if "onsets" not in self.data.columns:
                raise KeyError(
                    "Setting `segmented` to True requires that you have "
                    "run `.segment_into_units()` or provided unit "
                    "segmentation information."
                )
            else:
                kplot.segmentation(self, key, title=title, **kwargs)
        else:
            kplot.melspectrogram(
                self.files.at[key, "spectrogram"],
                parameters=self.parameters,
                title=title,
                **kwargs,
            )

    def sample_info(self) -> None:
        """
        Prints the length of the KantoData and other information.
        """
        out = inspect.cleandoc(
            f"""
        Total length: {len(self.metadata)}
        Unique IDs: {len(self.unique_ID)}"""
        )
        print(out)

    def plot_example(
        self,
        n_songs: int = 1,
        query: str = "maxfreq",
        order: str = "descending",
        return_keys: bool = False,
        **kwargs,
    ) -> None | List[str]:
        """
        Show mel spectrograms for songs at the ends of the time or
        frequency distribution.

        Note:
            Durations and frequencies come from bounding boxes,
            not vocalisations. This function, along with
            :func:`~pykanto.dataset.plot_summary`, is useful to spot any
            outliers, and to quickly explore the full range of data.

        Args:
            n_songs (int, optional): Number of songs to return. Defaults to 1.
            query (str, optional):
                What to query the database for. Defaults to 'maxfreq'. One of:

                - 'duration'
                - 'maxfreq'
                - 'minfreq'
            order (str, optional): Defaults to 'descending'. One of:

                - 'ascending'
                - 'descending'
            return_keys (bool, optional): Defaults to 'False'. Whether to return
                the keys of the displayed spectrograms.
            **kwargs: Keyword arguments to be passed to
                :func:`~pykanto.plot.melspectrogram`
        """
        argdict = {
            "duration": "length_s",
            "maxfreq": "upper_freq",
            "minfreq": "lower_freq",
        }
        if query not in argdict:
            raise ValueError(
                "show_extreme_songs: query must be one of "
                "['duration', 'maxfreq', 'minfreq']"
            )

        testkeys = (
            self.data[argdict[query]]
            .sort_values(ascending=True if order == "ascending" else False)[
                :n_songs
            ]
            .index
        )

        for key in testkeys:
            kplot.melspectrogram(
                self.files.at[key, "spectrogram"],
                parameters=self.parameters,
                title=Path(key).stem,
                **kwargs,
            )
        if return_keys:
            return list(testkeys)

    # REVIEW: quarantine for now
    # def relabel_noise_segments(self, keys: List[str]) -> None:
    #     """
    #     Moves entries from the vocalisations subset to the noise subset.
    #     You will be prompted to choose whether to save the dataset to disk.

    #     Warning:
    #         Changes 'label' in the json metadata within the dataset to
    #         'NOISE', but does not change the original json file from which
    #         you created the entry.
    #         Make sure you leave a trail when relabelling segments (i.e., do
    #         it programmatically) or else your work will not be fully
    #         reproducible!

    #     Args:
    #         keys (List[str]): A list of the segments to move.

    #     Examples:
    #         >>> dataset = KantoData(...)
    #         >>> keys_to_move = ['ABC.wav','CBA.wav']
    #         >>> dataset.relabel_segments(keys_to_move)

    #     """
    #     df_tomove = self.data.loc[keys]
    #     self.noise = pd.concat([self.noise, df_tomove], join="inner")
    #     self.data.drop(keys, inplace=True)
    #     # Change label in json metadata - TODO: also in original json file
    #     for key in keys:
    #         self.metadata[key]["label"] = "NOISE"
    #     if len(self.noise.loc[keys]) == len(keys):
    #         print("The entries were moved successfully.")
    #         tx = "Do you want to save the dataset to disk now?" " (Enter y/n)"
    #         while (res := input(tx).lower()) not in {"y", "n"}:
    #             pass
    #         if res == "y":
    #             print("Done.")
    #             self.save_to_disk(verbose=self.parameters.verbose)

    @timing
    def segment_into_units(self, overwrite: bool = False) -> None:
        """
        Adds segment onsets, offsets, unit and silence durations
        to :attr:`~.KantoData.data`.

        Warning:
            If segmentation fails for a vocalisation it will be dropped from
            the database so that it doesn't cause trouble downstream.

        Args:
            overwrite (bool, optional): Whether to overwrite unit
            segmentation information if it already exists. Defaults to False.

        Raises:
            FileExistsError: The vocalisations in this dataset have already
                been segmented.
        """

        if (
            all(item in self.data.columns for item in ["onsets", "offsets"])
            and overwrite is False
        ):
            warnings.warn(
                "The vocalisations in this dataset have already been segmented: "
                "will use existing segmentation information."
                "If you want to do it again, you can overwrite the existing "
                "segmentation information by it by setting `overwrite=True`"
            )
            print("Using existing unit onset/offset information.")
            onoff_df = self.data[["onsets", "offsets"]].dropna()
            onoff_df["index"] = onoff_df.index

        elif overwrite is True or not all(
            item in self.data.columns for item in ["onsets", "offsets"]
        ):
            units = segment_song_into_units_parallel(self, self.data.index)
            onoff_df = pd.DataFrame(
                units, columns=["index", "onsets", "offsets"]
            ).dropna()
            onoff_df.set_index("index", inplace=True)

        if not all(
            item in self.data.columns
            for item in ["unit_durations", "silence_durations"]
        ):
            self.data.drop(
                ["unit_durations", "silence_durations"],
                axis=1,
                errors="ignore",
                inplace=True,
            )
            onoff_df["unit_durations"] = (
                onoff_df["offsets"] - onoff_df["onsets"]
            )
            onoff_df["silence_durations"] = onoff_df.apply(
                lambda row: [
                    a - b for a, b in zip(row["onsets"][1:], row["offsets"])
                ],
                axis=1,
            ).to_numpy()
            self.data.drop(
                ["index", "onsets", "offsets"],
                axis=1,
                errors="ignore",
                inplace=True,
            )
            self.data = self.data.merge(
                onoff_df, left_index=True, right_index=True
            )
            self.data.drop(["index"], axis=1, errors="ignore", inplace=True)

        # Save
        self.save_to_disk(verbose=self.parameters.verbose)
        n_units = sum(
            [len(self.data.at[i, "unit_durations"]) for i in self.data.index]
        )
        print(f"Found and segmented {n_units} units.")

    @timing
    def subset(self, ids: List[str], new_dataset: str) -> KantoData:
        """
        Creates a new dataset containing a subset of the IDs present
        in the original dataset (e.g., different individual birds).

        Note:
            - Existing files common to both datasets
              (vocalisation spectrograms, unit spectrograms,
              wav files) will not be copied or moved from their
              original location.

            - Any newly generated files (e.g., by running a function
              that saves spectrograms to disk) will be exclusive to
              the new dataset.

        Args:
            dataset ([type]): Source dataset.
            ids (List[str]): IDs to keep.
            new_dataset (str): Name of new dataset.

        Returns:
            KantoData: A subset of the dataset.
        """
        # Copy dataset
        subself = copy.deepcopy(self)

        # Remove unwanted data
        subself.data = subself.data[subself.data["ID"].isin(ids)]
        if hasattr(self, "average_units"):
            rm_keys = [key for key in subself.average_units if key not in ids]
            _ = [subself.average_units.pop(key) for key in rm_keys]
        if hasattr(self, "units"):
            subself.units = subself.units[subself.units["ID"].isin(ids)]
        popkeys = [
            key
            for key, value in subself.metadata.items()
            if value["ID"] not in ids
        ]
        _ = [subself.metadata.pop(key) for key in popkeys]
        subself._get_unique_ids()

        # Change pointers / paths
        subself.DATASET_ID = new_dataset
        subself.DIRS.__dict__ = {
            k: Path(str(v).replace(self.DATASET_ID, new_dataset))
            for k, v in self.DIRS.__dict__.items()
        }

        # Save
        makedir(subself.DIRS.DATASET)
        pickle.dump(subself, open(subself.DIRS.DATASET, "wb"))
        print(f"Saved subset to {subself.DIRS.DATASET}")
        return subself

    def save_to_disk(self, verbose: bool = True) -> None:
        """
        Save dataset to disk.
        """
        # Save dataset
        out_dir = self.DIRS.DATASET
        makedir(out_dir)
        pickle.dump(self, open(out_dir, "wb"))
        if verbose:
            print(f"Saved dataset to {out_dir}")

    def to_csv(self, path: Path, timestamp: bool = True) -> None:
        """
        Output vocalisation (and, if present, unit) metadata in the dataset as
        a .csv file.

        Args:
            path (Path): Directory where to save the file(s).
            timestamp (bool, optional): Whether to add timestamp to file name.
                Defaults to True.
        """
        t = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}' if timestamp else ""
        self.data.to_csv(path / f"{self.DIRS.DATASET.stem}_{t}.csv")
        if hasattr(self, "units"):
            self.units.to_csv(path / f"{self.DIRS.DATASET.stem}_UNITS_{t}.csv")

    def reload(self) -> KantoData:
        """
        Load the current dataset from disk. Remember to assign the output to
        a variable!

        Warning:
            You will lose any changes that happened after the last time
            you saved the dataset to disk.

        Returns:
            KantoData: Last saved version of the dataset.

        Examples:
            >>> dataset = dataset.reload()
        """
        return pickle.load(open(self.DIRS.DATASET, "rb"))

    @timing
    def get_units(self, pad: bool = False) -> None:
        """
        Creates and saves a dictionary containing spectrogram
        representations of the units or the average of
        the units present in the vocalisations of each individual ID
        in the dataset.

        Args:
            pad (bool, optional): Whether to pad spectrograms
                to the maximum lenght (per ID). Defaults to False.

        Note:
            If `pad = True`, unit spectrograms are padded to the maximum
            duration found among units that belong to the same ID,
            not the dataset maximum.

        Note:
            If each of your IDs (grouping factor, such as individual or
            population) has lots of data and you are using a machine with very
            limited memory resources you will likely run out of it when
            executing this funtion in parallel. If this is the case, you can
            limit the number of cpus to be used at once by setting the
            `num_cpus` parameter to a smaller number.
        """
        # TODO: #7 Prepare data for interactive app using custom grouping
        # factor (current default: ID), or allow user to to choose something
        # other than individual ID as grouping factor. @nilomr
        song_level = self.parameters.song_level
        dic_locs = get_indv_units_parallel(
            self,
            pad=pad,
            song_level=song_level,
            num_cpus=self.parameters.num_cpus,
        )
        try:
            self.files.insert(0, "ID", self.data["ID"])
        except ValueError as e:
            if str(e) == "cannot insert ID, already exists":
                pass
            else:
                raise e
        colname = "average_units" if song_level else "units"
        au = pd.DataFrame(
            pd.Series(dic_locs),
            columns=[colname],
        )
        if colname in self.files.columns:
            self.files.drop(columns=colname, inplace=True)

        self.files = pd.merge(self.files, au, right_index=True, left_on="ID")
        self.save_to_disk(verbose=self.parameters.verbose)

    def write_to_json(self) -> None:
        """
        Write the dataset to the existing JSON files for each vocalisation.
        """
        save_to_jsons(self)

    @timing
    def cluster_ids(self, min_sample: int = 10) -> None:
        """
        Dimensionality reduction using UMAP + unsupervised clustering using
        HDBSCAN. This will fail if the sample size for an ID (grouping factor,
        such as individual or population) is too small.

        Adds cluster membership information and 2D UMAP coordinates to
        :attr:`~.KantoData.data` if `song_level=True`
        in :attr:`~.KantoData.parameters`, else to :attr:`~.KantoData.units`.

        Args:
            min_sample (int): Minimum sample size below which an ID will
                be skipped. Defaults to 10, but you can reallistically expect
                good automatic results above ~100.

        """
        df = reduce_and_cluster_parallel(
            self, min_sample=min_sample, num_cpus=self.parameters.num_cpus
        )

        if self.parameters.song_level:
            self.data = self.data.combine_first(df)
        else:
            self.units = df
            # TODO #14@nilomr: Add individual element information to self.data?

        self.save_to_disk(verbose=self.parameters.verbose)

    @timing
    def prepare_interactive_data(
        self, spec_length: float | None = None
    ) -> None:
        """
        Prepare lightweigth representations of each vocalization
        or unit (if song_level=False in :attr:`~.KantoData.parameters`)
        for each individual in the dataset.

        Args:
            spec_length (float, optional): In seconds, duration of
            spectrogram that will be produced. Shorter segments
            will be padded, longer segments trimmed. Defaults to maximum note
            duration in the dataset.

        """
        song_level = self.parameters.song_level

        # Check that we have the relevant data to proceed
        if not hasattr(self, "units") and not song_level:
            raise KeyError(
                "This function requires the output of "
                "`self.cluster_ids()`. "
                "Make sure you that have run it, then try again"
            )

        if not set(["auto_class", "umap_x", "umap_y"]).issubset(
            self.data.columns if song_level else self.units.columns
        ):
            raise KeyError(
                "This function requires the output of "
                "`self.cluster_ids()`. "
                "Make sure you that have run it, then try again"
            )

        dt = prepare_datasource_parallel(
            self,
            spec_length=spec_length,
            song_level=song_level,
            num_cpus=self.parameters.num_cpus,
        )

        # Add their locations to the files df, then save
        int_df = pd.DataFrame(
            flatten_list(dt),
            columns=["ID", "voc_app_data" if song_level else "unit_app_data"],
        )
        int_df["voc_check" if song_level else "unit_check"] = False
        ID = self.files.ID
        self.files = self.files.drop(
            self.files.columns.intersection(int_df.columns), axis=1
        )
        self.files["ID"] = ID
        self.files = (
            self.files.reset_index()
            .merge(int_df, how="left", on="ID")
            .set_index("index")
        )

        self.save_to_disk(verbose=self.parameters.verbose)

    def open_label_app(
        self, palette: Tuple[Literal[str], ...] = Set3_12, max_n_labs: int = 10
    ) -> None:
        """
        Opens a new web browser tab with an interactive app that can be used to
        check the accuracy of the automaticaly assigned labels. Uses average
        units per vocalisation (if song_level=True in self.parameters) or
        individual units.

        Note:
            Starting this app requires the output of
            :func:`~pykanto.dataset.KantoData.prepare_interactive_data`; you will
            be prompted to choose whether to run it if it is missing.

        Note:
            The app will try to create a palette based on the maximum number
            of categories in any individual in the dataset. You will get
            a ValueError if the palette you provided is not large enough
            (the default palette allows for a maximum of 12 categories).
            You can use your own palettes or import existing ones,
            see 'Examples' below.

        Args:
            palette (List[str], optional): A colour palette of
                length >= max_n_labs. Defaults to list(Set3_12).
            max_n_labs (int, optional): maximum number of classes expected in
                the dataset. Defaults to 10.

        Examples:
            Allow for a maximum of 20 labels by setting ``max_n_labs = 20`` and
            ``palette = list(Category20_20)``.

            >>> from bokeh.palettes import Category20_20
            ...
            >>> self.open_label_app(max_n_labs = 20,
            >>> palette = list(Category20_20))

        """
        song_level = self.parameters.song_level
        datatype = "data" if song_level else "units"
        labtype = "voc_app_data" if song_level else "unit_app_data"

        # Can this be run?
        # Save and reload dataset:
        self.save_to_disk(verbose=self.parameters.verbose)
        self = self.reload()

        if not labtype in self.files or "auto_class" not in getattr(
            self, datatype
        ):
            raise IndexError(
                "This function requires the output of "
                " `self.cluster_ids()` and `self.prepare_interactive_data()` "
                "Make sure you that have run this, then try again."
            )

        if all(self.files["voc_check" if song_level else "unit_check"]) is True:
            raise KeyError(
                "You have already checked all the labels in this dataset. "
                "If you want to re-check any/all, change the check status of "
                "any ID from True to False in `self.files.voc_check` or "
                "`self.files.unit_check`, then run again."
            )

        # Check that we are where we should
        # REVIEW: easy to break!
        app_path = Path(__file__).parent / "app"
        if not app_path.is_dir():
            raise FileNotFoundError(str(app_path))

        # Check palette validity
        max_labs_data = (
            getattr(self, datatype)
            .dropna(subset=["auto_class"])
            .groupby(["ID"])["auto_class"]
            .nunique()
            .max()
        )
        max_labs = max_n_labs if max_n_labs > max_labs_data else max_labs_data
        if len(palette) < max_labs:
            raise ValueError(
                "The chosen palette does not have enough values. "
                f"Palette lenght : {len(palette)}, needed: "
                f"{max_labs}"
            )

        # Run app
        command = (
            f"bokeh serve --show {str(app_path)} --args "
            f"{str(self.DIRS.DATASET)} {max_labs} {song_level} {palette}"
        )
        with subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        ) as p:
            for line in p.stdout:
                print(line, end="")  # process line here

        if p.returncode != 0:
            print()
            raise subprocess.CalledProcessError(
                p.returncode,
                p.args,
                "Troubleshooting: the port may already be in use. Check "
                "that the port is free and try again.",
            )
