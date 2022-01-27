# ─── DESCRIPTION ──────────────────────────────────────────────────────────────
"""
Create a SongDataset object and auxiliary classes
"""

# ─── LIBRARIES ────────────────────────────────────────────────────────────────

from __future__ import annotations

import copy
import inspect
import math
import pickle
import subprocess
import warnings
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
from bokeh.palettes import Set3_12

import pykanto.plot as kplot
from pykanto import __name__ as modname
from pykanto.intlabel.data import prepare_datasource
from pykanto.parameters import Parameters
from pykanto.signal.cluster import reduce_and_cluster_parallel
from pykanto.signal.segment import segment_song_into_units_parallel
from pykanto.signal.spectrogram import (_save_melspectrogram_parallel,
                                        get_indv_units_parallel)
from pykanto.utils.compute import (dictlist_to_dict, print_dict, timing,
                                   to_iterator, tqdmm)
from pykanto.utils.paths import ProjDirs
from pykanto.utils.read import _get_json, _get_json_parallel
from pykanto.utils.write import makedir

# ─── CLASSES ──────────────────────────────────────────────────────────────────


class SongDataset():
    """
    Main dataset class. See __init__ documentation.

    Attributes:
        DIRS (:class:`~pykanto.utils.paths.ProjDirs`):
        vocalisations: Placeholder
        noise: Placeholder
        units: Placeholder
        parameters (:class:`~pykanto.parameters.Parameters`)

    """

    def __init__(
        self, DATASET_ID: str, DIRS: ProjDirs,
        parameters: Parameters = None,
        random_subset: int = None,
        overwrite_dataset: bool = False,
        overwrite_data: bool = False
    ) -> None:
        """
        Instantiates the main dataset class.

        Note:
            If your dataset contains noise samples (useful when training a
            neural network), these should be labelled as 'NOISE' in the
            corresponding json file.
            I.e., `json_file["label"] == 'NOISE'`.
            Files where `json_file["label"] == []` or "label" is not a key
            will be automatically given the label 'VOCALISATION'.

        Args:
            DATASET_ID (str): Name of new dataset.
            DIRS (ProjDirs): Project directory structure. Must contain 
                a 'WAVFILES' attribute pointing to a directory that contains 
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
                'You have not provided parameters for this dataset.'
                'Setting default parameters - these will almost certainly be '
                'inadequate for your dataset.')
            self.parameters = Parameters()
        else:
            self.parameters = parameters

        # Get dataset identifier and location
        self.DIRS = copy.deepcopy(DIRS)
        self.DATASET_ID = DATASET_ID
        self.DIRS.DATASET = DIRS.DATA / "datasets" / \
            DATASET_ID / f"{DATASET_ID}.db"
        self.DIRS.SPECTROGRAMS = (
            self.DIRS.DATA / "datasets" / self.DATASET_ID / "spectrograms")

        # Throw error if dataset already exists
        if self.DIRS.DATASET.is_file() and overwrite_dataset is False:
            raise FileExistsError(
                f'{DATASET_ID} already exists. You can overwrite '
                'it by setting `overwrite_dataset=True`')

        # Build dataset
        self._get_wav_json_filedirs(random_subset=random_subset)
        # TODO: perform a field check for the json files and provide
        # feedback on any missing/wrong fields, or their value types.
        self._load_metadata()
        self._get_unique_ids()
        self._get_sound_info()
        self._compute_melspectrograms(
            dereverb=self.parameters.dereverb, overwrite_data=overwrite_data)

        # Save current state of dataset
        self.save_to_disk(verbose=self.parameters.verbose)
        print('Done')

    def __str__(self) -> str:
        """
        Returns the class object as a string.
        Returns:
            str: Pretty printed dictionary contents
        Examples:
            >>> dataset = SongDataset(...)
            >>> print(dataset)
            Items held:
            minmax_values: {'max_freq': 10249} + 3 other entries.
            vocalisations: pd.DataFrame with length 20 and columns
        """
        return print_dict(self.__dict__)

    # ──────────────────────────────────────────────────────────────────────────
    # SongDataset: Private methods

    @timing
    def _get_wav_json_filedirs(self, random_subset: int = None) -> None:
        """
        Get paths of wav and json files for this dataset.

        Args:
            random_subset (int, optional): Get a random subset without
                repetition. Defaults to None.

        Warning:
            Will drop any .wav for which there is no .json!

        """
        if not hasattr(self.DIRS, 'WAVFILES'):
            raise KeyError(
                'The ProjDirs object you used to initialise '
                'this SongDataset object does not have a WAVFILES attribute. '
                'See the docs for SongDataset for mroe information.')

        # TODO: read both lower and uppercase wav and json extensions.
        # Load and check file paths:
        self.DIRS.WAV_LIST = sorted(
            list((self.DIRS.WAVFILES / "WAV").glob("*.wav")))
        self.DIRS.JSON_LIST = sorted(
            list((self.DIRS.WAVFILES / "JSON").glob("*.JSON")))
        if not len(self.DIRS.WAV_LIST):
            raise FileNotFoundError(
                f'There are no .wav files in {self.DIRS.WAV_LIST}')

        wavnames = [wav.name for wav in self.DIRS.WAV_LIST]
        matching_jsons = [json for json in self.DIRS.JSON_LIST
                          if json.stem in wavnames]

        if len(self.DIRS.JSON_LIST) != len(matching_jsons):
            ndrop = len(self.DIRS.JSON_LIST) - len(matching_jsons)
            warnings.warn(
                "There is an unequal number of matching .wav and .json "
                f"files in {self.DIRS.WAVFILES}."
                f"Keeping only those that match: dropped {ndrop}")
            keepnames = [json.stem for json in matching_jsons]
            self.DIRS.WAV_LIST = [wav for wav in self.DIRS.WAV_LIST
                                  if wav.name in keepnames]
            self.DIRS.JSON_LIST = matching_jsons

        # Subset as per parameters if possible
        if len(self.DIRS.WAV_LIST) == 1:
            if self.parameters.subset != (0, -1):
                warnings.warn("Cannot subset: there is only one file.")
                pass
        elif self.parameters.subset[0] < len(
                self.DIRS.WAV_LIST) and self.parameters.subset[1] <= len(
                self.DIRS.WAV_LIST):
            self.DIRS.WAV_LIST = self.DIRS.WAV_LIST[
                self.parameters.subset[0]:self.parameters.subset[1]]
            self.DIRS.JSON_LIST = self.DIRS.JSON_LIST[
                self.parameters.subset[0]:self.parameters.subset[1]]
        else:
            raise IndexError(
                "List index out of range: the provided "
                f"'subset' parameter {self.parameters.subset} is ouside "
                f"the range of the dataset (0, {len(self.DIRS.WAV_LIST)}).")

        if random_subset:
            if random_subset > len(self.DIRS.JSON_LIST):
                random_subset = len(self.DIRS.JSON_LIST)
                warnings.warn(
                    "The 'random_subset' you specified is larger "
                    "than the dataset: setting to the length of the dataset.")
            self.DIRS.WAV_LIST, self.DIRS.JSON_LIST = zip(
                *sample(list(zip(
                    self.DIRS.WAV_LIST,
                    self.DIRS.JSON_LIST
                )), random_subset))
            self.DIRS.WAV_LIST = list(self.DIRS.WAV_LIST)
            self.DIRS.JSON_LIST = list(self.DIRS.JSON_LIST)

    @timing
    def _load_metadata(self) -> None:
        """
        Loads json metadata in sequence if there are < 100 items, 
        in parallel chunks of length chunk_len if there are > 1000 items.
        """

        n_jsonfiles = len(self.DIRS.JSON_LIST)

        if n_jsonfiles < 100:
            jsons = [
                _get_json(json)
                for json in tqdmm(
                    self.DIRS.JSON_LIST, desc="Loading JSON files",
                    total=n_jsonfiles)]

        else:
            jsons = _get_json_parallel(self.DIRS.JSON_LIST)

        self.metadata = {
            Path(json["wav_loc"]).name: json for json in jsons
        }

    def _get_unique_ids(self) -> None:
        """
        Adds a 'unique_ID' attribute holding and array 
        with unique IDs in the dataset.
        """
        self.ID = np.array(
            [value["ID"] for _, value in self.metadata.items()]
        )
        self.unique_ID = np.unique(self.ID)

    def _get_sound_info(self) -> None:
        """
        Adds pandas DataFrames with vocalisation and noise 
        information to dataset ('vocalisations' and 'noise' 
        attributes, respectively).
        Also adds information about mim/max bounding box
        frequencies and durations in the dataset 
        ('minmax_values' attribute), and unit onsets/offsets
        if present.
        """
        vocalisations_dict = {}
        noise_dict = {}
        for key, file in self.metadata.items():
            data = {
                "wav_loc": file['wav_loc'],
                "ID": file["ID"],
                "lower_freq": file["lower_freq"],
                "upper_freq": file["upper_freq"],
                "length_s": file['length_s'],

            }
            if all(e in file for e in ['onsets', 'offsets']):
                data['onsets'], data['offsets'] = (
                    np.array(file['onsets']), np.array(file['offsets']))

            if file['label'] == 'VOCALISATION':
                vocalisations_dict[key] = data
            elif file['label'] == 'NOISE':
                noise_dict[key] = data
            else:
                warnings.warn(
                    f"Warning: {key} has an incorrect label: {file['label']}")

        # Add to a dataframe
        self.vocalisations = pd.DataFrame.from_dict(
            vocalisations_dict, orient='index')
        self.noise = pd.DataFrame.from_dict(noise_dict, orient='index')

        # Get minimum and maximum frequencies and durations in song dataset
        self.minmax_values = {
            'max_freq': max(self.vocalisations['upper_freq']),
            'min_freq': min(self.vocalisations['lower_freq']),
            'max_duration': max(self.vocalisations['length_s']),
            'min_duration': min(self.vocalisations['length_s'])
        }

    @timing
    def _compute_melspectrograms(
            self, overwrite_data: bool = False, **kwargs) -> None:
        """
        Compute melspectrograms and add their location to the SongDataset object.
        It applies both dereverberation and bandpasses vocalisation data by
        default, and neither of these to noise data.

        Args:
            overwrite_data (bool): Whether to overwrite any spectrogram files 
            that already exist for this dataset.
            **kwargs: Passed to 
                :func:`~pykanto.signal.spectrogram.save_melspectrogram`.
        """

        def _spec_exists(key):  # TODO: refactor
            file = self.metadata[key]['wav_loc']
            ID = self.metadata[key]['ID']
            path = self.DIRS.SPECTROGRAMS / ID / (Path(file).stem + '.npy')
            if path.exists():
                return {Path(file).name: path}

        # Check if spectrograms already exist for any keys:
        existing_voc = [_spec_exists(key) for key in self.vocalisations.index]
        existing_voc = dictlist_to_dict(
            [x for x in existing_voc if x is not None])
        existing_noise = [_spec_exists(key) for key in self.noise.index]
        existing_noise = dictlist_to_dict(
            [x for x in existing_noise if x is not None])

        # Get new keys (all keys if overwrite_data=True)
        if overwrite_data:
            new_voc_keys = self.vocalisations.index.tolist()
            new_noise_keys = self.noise.index.tolist()
        else:
            new_voc_keys = [key for key in self.vocalisations.index
                            if key not in existing_voc]
            new_noise_keys = [
                key for key in self.noise.index
                if key not in existing_noise]

        # Compute vocalisation melspectrograms
        specs = _save_melspectrogram_parallel(
            self, new_voc_keys, **kwargs) if new_voc_keys else {}
        specs = {
            **specs, **(existing_voc if (existing_voc and
                                         not overwrite_data) else{})}
        self.vocalisations['spectrogram_loc'] = pd.Series(specs)

        # Now compute noise melspectrograms if present
        n_specs = _save_melspectrogram_parallel(
            self, new_noise_keys, **kwargs) if new_noise_keys else {}
        n_specs = {
            **n_specs, **(existing_noise if (existing_noise and
                                             not overwrite_data) else {})}
        self.noise['spectrogram_loc'] = pd.Series(n_specs)

    # ──────────────────────────────────────────────────────────────────────────
    # SongDataset: Public methods

    def summary_plot(self, nbins: int = 50,
                     variable: str = 'frequency') -> None:
        """
        Plots a histogram + kernel densiyy estimate of the frequency
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

        if variable not in ['frequency', 'duration', 'sample_size', 'all']:
            raise ValueError(
                "`variable` must be one of ['frequency', 'duration', 'sample_size', 'all']")

        # Plot size and general aesthetics
        sns.set(font_scale=1.5, rc={'axes.facecolor': '#ededed'}, style="dark")
        fig, axes = plt.subplots(
            figsize=(18 if variable == 'all' else 6, 6),
            ncols=3 if variable == 'all' else 1)

        # Build frequency or duration plots
        for i, var in enumerate(['frequency', 'duration']):
            if var != variable and variable != 'all':
                continue

            if var == 'frequency':
                data = {'upper_freq': self.vocalisations['upper_freq'],
                        'lower_freq': self.vocalisations['lower_freq']}
            else:
                data = {'song_duration': self.vocalisations['length_s']}

            nax = i if variable == 'all' else 0
            sns.histplot(data, bins=nbins, kde=True,
                         palette=['#107794', '#d97102']
                         if var == 'frequency' else['#107794'], legend=False,
                         ax=axes if variable != 'all' else axes[i],
                         linewidth=0.2, log_scale=True
                         if var == 'duration' else False,
                         line_kws=dict(linewidth=5, alpha=0.7))

            if var == 'duration':
                (axes[i] if variable == 'all' else axes).xaxis.set_major_formatter(
                    mpl.ticker.ScalarFormatter())

            if var == 'frequency':
                (axes[i] if variable == 'all' else axes).legend(
                    labels=['Min', 'Max'],
                    loc=2, bbox_to_anchor=(0.70, 1),
                    borderaxespad=0, frameon=False)
            ((axes[i] if variable == 'all' else axes).set(
                xlabel='Frequency (Hz)'
                if var == 'frequency' else 'Duration (s)', ylabel='Count'))

        # Build sample size plot
        if variable in ['sample_size', 'all']:
            individual_ss = self.vocalisations['ID'].value_counts()
            data = pd.DataFrame(individual_ss).rename(columns={'ID': 'n'})
            data['ID'] = individual_ss.index
            sns.histplot(
                data=data, palette=['#107794'],
                bins=nbins, ax=axes if variable != 'all' else axes[2], alpha=0.6,
                legend=False)
            (axes[2] if variable == 'all' else axes).set(
                xlabel=f'Sample size (total: {len(self.vocalisations)})',
                ylabel='Count')
            # Reduce xtick density
            nlabs = len(
                (axes[2] if variable == 'all' else axes).get_xticklabels())
            mid = math.trunc(nlabs/2)
            for i, label in enumerate(
                    (axes[2] if variable == 'all' else axes).get_xticklabels()):
                if i not in [0, mid, nlabs-1]:
                    label.set_visible(False)

        # Common axes
        if variable == 'all':
            for ax in axes:
                ax.yaxis.labelpad = 15
                ax.xaxis.labelpad = 15
        else:
            axes.yaxis.labelpad = 15
            axes.xaxis.labelpad = 15

        fig.tight_layout()
        plt.show()

    def sample_info(self) -> None:
        """
        Prints the length of the SongDataset and other information.
        """
        out = inspect.cleandoc(f"""
        Total length: {len(self.metadata)}
        Vocalisations: {len(self.vocalisations)}
        Noise: {len(self.noise)}
        Unique IDs: {len(self.unique_ID)}""")
        print(out)

    def show_extreme_samples(
            self,
            n_songs: int = 1,
            query: str = 'maxfreq',
            order: str = 'descending',
            return_keys: bool = False,
            **kwargs) -> None | List[str]:
        """
        Show mel spectrograms for songs at the ends of the time or
        frequency distribution.

        Note:
            Durations and frequencies come from bounding boxes, not vocalisations.
            This function, along with :func:`~pykanto.dataset.summary_plot`,
            is useful to spot any outliers, and to quickly explore the full range of data.

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
        if query == 'duration':
            idx = 'length_s'
        elif query == 'maxfreq':
            idx = 'upper_freq'
        elif query == 'minfreq':
            idx = 'lower_freq'
        else:
            raise ValueError(
                "show_extreme_songs: query must be one of "
                "['duration', 'maxfreq', 'minfreq']")

        testkeys = self.vocalisations[idx].sort_values(
            ascending=True if order == 'ascending' else False)[:n_songs].index

        for key in testkeys:
            if return_keys:
                print(key)
            kplot.melspectrogram(
                self.vocalisations.at[key, 'spectrogram_loc'],
                parameters=self.parameters, title=Path(key).stem, **kwargs)

        if return_keys:
            return list(testkeys)

    def relabel_noise_segments(self, keys: List[str]) -> None:
        """
        Moves entries from the vocalisations subset to the noise subset.
        You will be prompted to choose whether to save the dataset to disk.

        Warning:
            Changes 'label' in the json metadata within the dataset to
            'NOISE', but does not change the original json file from which
            you created the entry.
            Make sure you leave a trail when relabelling segments (i.e., do
            it programmatically) or else your work will not be fully
            reproducible!

        Args:
            keys (List[str]): A list of the segments to move.

        Examples:
            >>> dataset = SongDataset(...)
            >>> keys_to_move = ['ABC.wav','CBA.wav']
            >>> dataset.relabel_segments(keys_to_move)

        """
        df_tomove = self.vocalisations.loc[keys]
        self.noise = pd.concat([self.noise, df_tomove], join='inner')
        self.vocalisations.drop(keys, inplace=True)
        # Change label in json metadata - TODO: also in original json file
        for key in keys:
            self.metadata[key]['label'] = 'NOISE'
        if len(self.noise.loc[keys]) == len(keys):
            print('The entries were moved successfully.')
            tx = ('Do you want to save the dataset to disk now?'
                  ' (Enter y/n)')
            while (res := input(tx).lower()) not in {"y", "n"}:
                pass
            if res == 'y':
                print('Done.')
                self.save_to_disk(verbose=self.parameters.verbose)

    @timing
    def segment_into_units(self, overwrite: bool = False) -> None:  # ANCHOR
        """
        Adds segment onsets, offsets, unit and silence durations
        to :attr:`~.SongDataset.vocalisations`.

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
        # First check that you are not running this by mistake:

        # Throw segmentation already exists
        if 'unit_durations' in self.vocalisations.columns and overwrite is False:
            raise FileExistsError(
                'The vocalisations in this dataset have already been segmented. '
                'If you want to do it again, you can overwrite the existing '
                'segmentation information by it by setting `overwrite=True`')

        # Find song units iff onset/offset metadata not present
        if not 'onsets' in self.vocalisations.columns:
            units = segment_song_into_units_parallel(
                self, self.vocalisations.index)

        # Otherwise just use that
        else:
            print('Using existing unit onset/offset information.')
            onoff_df = self.vocalisations[["onsets", "offsets"]].dropna()
            onoff_df['index'] = onoff_df.index

        # Calculate durations and add to dataset
        onoff_df = pd.DataFrame(
            units, columns=['index', 'onsets', 'offsets']).dropna()
        onoff_df.set_index('index', inplace=True)
        onoff_df['unit_durations'] = onoff_df['offsets'] - onoff_df['onsets']
        onoff_df['silence_durations'] = onoff_df.apply(
            lambda row: [a - b for a, b in zip(row['onsets'][1:],
                                               row['offsets'])], axis=1)
        self.vocalisations = self.vocalisations.drop(
            ['index', 'onsets', 'offsets'],
            axis=1, errors='ignore')
        self.vocalisations = self.vocalisations.merge(
            onoff_df, left_index=True, right_index=True)

        # Save
        self.save_to_disk(verbose=self.parameters.verbose)
        n_units = sum([len(self.vocalisations.at[i, 'unit_durations'])
                       for i in self.vocalisations.index])
        print(f'Found and segmented {n_units} units.')

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

    @timing
    def subset(self, ids: List[str], new_dataset: str) -> SongDataset:
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
            SongDataset: A subset of the dataset.
        """
        # Copy dataset
        subself = copy.deepcopy(self)

        # Remove unwanted data
        subself.vocalisations = subself.vocalisations[subself.vocalisations['ID'].isin(
            ids)]
        if hasattr(self, 'average_units'):
            rm_keys = [key for key in subself.average_units if key not in ids]
            _ = [subself.average_units.pop(key) for key in rm_keys]
        if hasattr(self, 'units'):
            subself.units = subself.units[subself.units['ID'].isin(ids)]
        if hasattr(self, 'noise'):
            if len(subself.noise):
                subself.noise = subself.noise[subself.noise['ID'].isin(ids)]
        popkeys = [key for key, value in subself.metadata.items()
                   if value['ID'] not in ids]
        _ = [subself.metadata.pop(key) for key in popkeys]
        subself._get_unique_ids()

        # Change pointers / paths
        subself.DATASET_ID = new_dataset
        subself.DIRS.__dict__ = {
            k: Path(str(v).replace(self.DATASET_ID, new_dataset)) for k,
            v in self.DIRS.__dict__.items()}

        # Save
        makedir(subself.DIRS.DATASET)
        pickle.dump(subself, open(subself.DIRS.DATASET, "wb"))
        print(f"Saved subset to {subself.DIRS.DATASET}")
        return subself

    def reload(self) -> SongDataset:
        """
        Load the current dataset from disk.

        Warning:
            You will lose any changes that happened after the last time
            you saved the dataset to disk.

        Returns:
            SongDataset: Last saved version of the dataset.

        Examples:
            >>> dataset = dataset.reload()
        """
        return pickle.load(open(self.DIRS.DATASET, "rb"))

    def plot_vocalisation_segmentation(self, key: str, **kwargs) -> None:
        """
        Plots a vocalisation and overlays the results
        of the segmentation process.

        Args:
            key (str): Vocalisation identificator.
            **kwargs: Keyword arguments to be passed to
                :func:`~pykanto.plot.melspectrogram`
        """
        if 'onsets' not in self.vocalisations.columns:
            raise KeyError("This method requires that you have "
                           "run `.segment_into_units()` or provided unit "
                           "segmentation information.")
        kplot.segmentation(self, key, **kwargs)

    @timing
    def get_units(self, pad: bool = True) -> None:
        """
        Creates and saves a dictionary containing spectrogram
        representations of the units or the average of
        the units present in the vocalisations of each individual ID
        in the dataset. The dictionary's location can be found
        at `self.DIRS.UNITS` if `song_level = False`, and
        at `self.DIRS.AVG_UNITS` if `song_level = True` in 
        :attr:`~.SongDataset.parameters`.

        Args:
            pad (bool, optional): Whether to pad spectrograms 
                to the maximum lenght (per ID). Defaults to True.

        Note:
            If ``pad = True``, unit spectrograms are padded to the maximum
            duration found among units that belong to the same ID, 
            not the global maximum.
        """
        song_level = self.parameters.song_level
        dic_locs = get_indv_units_parallel(self, pad=pad, song_level=song_level)

        if song_level:
            self.DIRS.AVG_UNITS = dic_locs
        else:
            self.DIRS.UNITS = dic_locs
        self.save_to_disk(verbose=self.parameters.verbose)

    @timing
    def cluster_individuals(self, min_sample: int = 10) -> None:
        """
        Dimensionality reduction using UMAP + unsupervised clustering
        using HDBSCAN. This will fail if the sample size for an individual
        is too small.

        Adds cluster membership information and 2D UMAP coordinates to 
        :attr:`~.SongDataset.vocalisations` if `song_level=True` 
        in :attr:`~.SongDataset.parameters`, else to :attr:`~.SongDataset.units`.

        Args:
            min_sample (int): Minimum sample size below which an individual will
                be skipped. Defaults to 10, but you can expect good automatic 
                results above ~100.

        """
        df = reduce_and_cluster_parallel(self, min_sample=min_sample)

        if self.parameters.song_level:
            self.vocalisations = self.vocalisations.combine_first(df)
        else:
            self.units = df

        self.save_to_disk(verbose=self.parameters.verbose)

    @timing
    def prepare_interactive_data(
            self, spec_length: float | None = None) -> None:
        """
        Prepare lightweigth representations of each vocalization
        or unit (if song_level=False in :attr:`~.SongDataset.parameters`) 
        for each individual in the dataset. These are linked from 
        VOCALISATION_LABELS or UNIT_LABELS in :attr:`~.SongDataset.DIRS`.

        Args:
            spec_length (float, optional): In seconds, duration of 
            spectrogram that will be produced. Shorter segments 
            will be padded, longer segments trimmed. Defaults to maximum note 
            duration in the dataset.

        """
        song_level = self.parameters.song_level

        # Check that we have the relevant data to proceed
        if not hasattr(self, 'units') and not song_level:
            raise KeyError(
                "This function requires the output of "
                "`self.cluster_individuals()`. "
                "Make sure you that have run it, then try again")

        elif not set(
                ['auto_cluster_label', 'umap_x', 'umap_y']).issubset(
                self.vocalisations.columns if song_level
                else self.units.columns):
            raise KeyError(
                "This function requires the output of "
                "`self.cluster_individuals()`. "
                "Make sure you that have run it, then try again")

        # Prepare dataframes with spectrogram pngs
        prepare_datasource_r = ray.remote(prepare_datasource)
        indvs = np.unique(
            self.vocalisations['ID'] if song_level else self.units['ID'])

        # Set or get spectrogram length (affects width of unit/voc preview)
        # NOTE: this is set to maxlen=50%len if using units, 4*maxlen if using
        # entire vocalisations. This might not be adequate in all cases.
        if not spec_length:
            max_l = max(
                [i
                 for array in self.vocalisations.unit_durations.values
                 for i in array])
            spec_length = (max_l + 0.5*max_l) if not song_level else 4*max_l
        spec_length_frames = int(np.floor(
            spec_length * self.parameters.sr / self.parameters.hop_length_ms))

        # Distribute with Ray
        obj_ids = [
            prepare_datasource_r.remote(
                self, individual, spec_length=spec_length_frames,
                song_level=song_level)
            for individual in tqdmm(
                indvs, desc='Initiate: Prepare interactive visualisation')]

        dt = [obj for obj in tqdmm(
            to_iterator(obj_ids),
            desc='Prepare interactive visualisation', total=len(indvs))]

        # Add their locations to the DIRS object, then save
        if song_level:
            self.DIRS.VOCALISATION_LABELS = {
                'predatasource': dict(dt),
                'already_checked': []}
        else:
            self.DIRS.UNIT_LABELS = {
                'predatasource': dict(dt),
                'already_checked': []}

        self.save_to_disk(verbose=self.parameters.verbose)

    def open_label_app(
            self, palette: List[str] = Set3_12, max_n_labs: int = 10) -> None:
        """
        Opens a new web browser tab with an interactive app that can be used to
        check the accuracy of the automaticaly assigned labels. Uses average
        units per vocalisation (if song_level=True in self.parameters) or
        individual units.

        Note:
            Starting this app requires the output of
            :func:`~pykanto.dataset.SongDataset.prepare_interactive_data`; you will
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

        if song_level:
            dataset_type = 'vocalisations'
            dataset_labels = 'VOCALISATION_LABELS'
        else:
            dataset_type = 'units'
            dataset_labels = 'UNIT_LABELS'

        # Can this be run?
        # Save and reload dataset:
        self.save_to_disk(verbose=self.parameters.verbose)
        self = self.reload()

        if not hasattr(self.DIRS, dataset_labels):
            tx = (
                'This app requires the output of '
                '`self.prepare_interactive_data()`. Do you want to run it now?'
                ' (Enter y/n)')
            while (res := input(tx).lower()) not in {"y", "n"}:
                pass

            if res == 'y':
                self.prepare_interactive_data()
            else:
                print('Stopped.')
                return None

        if 'auto_cluster_label' not in getattr(self, dataset_type).columns:
            raise ValueError('You need to run `self.cluster_individuals`'
                             ' before you can check its results.')

        if set(
            getattr(self.DIRS, dataset_labels)['already_checked']) == set(
            getattr(self, dataset_type).dropna(
                subset=['auto_cluster_label'])['ID']):
            raise KeyError(
                "You have already checked all the labels in this dataset. "
                "If you want to re-check any/all, remove individuals from "
                "`self.(vocalisations/unit)_labels['already_checked']`, "
                "then run again.")

        # Check that we are where we should
        # REVIEW: easy to break!
        app_path = (Path(__file__).parent / 'intlabel')
        if not app_path.is_dir():
            raise FileNotFoundError(str(app_path))

        # Check palette validity
        max_labs_data = getattr(self, dataset_type).dropna(
            subset=['auto_cluster_label'])[
            'auto_cluster_label'].astype(int).max() + 1

        max_labs = max_n_labs if max_n_labs > max_labs_data else max_labs_data

        if len(palette) < max_labs:
            raise ValueError(
                "The chosen palette does not have enough values. "
                f"Palette lenght : {len(palette)}, needed: "
                f"{max_labs}")

        # Run app
        command = (
            f'bokeh serve --show {str(app_path)} --args '
            f'{str(self.DIRS.DATASET)} {max_labs} {song_level} {palette}')
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, bufsize=1,
                              universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='')  # process line here

        if p.returncode != 0:
            print()
            raise subprocess.CalledProcessError(
                p.returncode, p.args,
                "Troubleshooting: the port may already be in use. Check "
                "that the port is free and try again.")
