# ─── DESCRIPTION ────────────────────────────────────────────────────────────────
# Parameter class - to keep all settings in one place

import json
from typing import Any

from pykanto.utils.compute import print_dict


class Parameters():
    """
    General parameters class - stores parameters to be passed to 
    a :meth:`~pykanto.dataset.SongDataset` object.
    """

    def __init__(self, **kwargs: Any):
        """
        Sets default parameters, then updates with user-provided parameters.
        """

        # Set defaults first

        # Spectrogramming
        self.window_length = 1024
        """Number of samples. Defaults to 1024."""
        self.hop_length = 128
        """Number of samples between successive frames. Defaults to 128."""
        self.fft_size = 1024
        """Number of bins used to divide the window. Defaults to 1024."""
        self.num_mel_bins = 224
        """Number of Mel frequency bins to use 
        (n of 'rows' in mel-spectrogram). Defaults to 224."""
        self.sr = 22050
        """The sampling rate (samples taken per second). 
        Defaults to 22050."""
        self.hop_length_ms = self.sr / self.hop_length
        """Calculated automatically."""
        self.fft_rate = 1000 / self.hop_length_ms
        """Calculated automatically."""
        self.top_dB = 65
        """Top dB to keep. Defaults to 65.
        Example: if ``top_dB`` = 65 anything below -65dB will be masked."""
        self.lowcut = 100
        """Minimum frequency (in Hz) to include by default. Defaults to 100."""
        self.highcut = 1000
        """Maximum frequency (in Hz) to include by default. 
        Defaults to 1000."""
        self.dereverb: bool = False

        # Segmentation
        self.max_dB = -30
        """Maximum threshold to reach during segmentation, in dB. 
        Defaults to -30"""
        self.dB_delta = 5
        """Size of thresholding steps, in dB. Defaults to 5."""
        self.silence_threshold = 0.2
        """Threshold separating silence from voiced segments. 
        Between 0.1 and 0.3 tends to work well. Defaults to 0.2."""
        self.max_unit_length = 0.4
        """Maximum unit length allowed. Defaults to 0.4. """
        self.min_unit_length = 0.03
        """Minimum unit length allowed. Defaults to 0.03."""
        self.min_silence_length = 0.001
        """Minimum silence length allowed. Defaults to 0.001."""
        self.gauss_sigma = 3
        """Sigma for gaussian kernel. Defaults to 3."""

        # general settings
        self.song_level = False
        """Whether to return the average of all units.
        Defaults to False (return individual units in each vocalisation)"""
        self.subset = (0, -1)
        """Indices of the first and last items to include in the dataset. 
        Defaults to (0,-1); full dataset"""
        self.n_jobs = -2
        """How many jobs to create. Default is available number of cores -2."""
        self.verbose = True
        """How much detail to provide. Defaults to True (verbose)."""
        # Update with user-provided parameters.
        self.__dict__.update(kwargs)

    def update(self, **kwargs: Any) -> None:
        """
        Update one or more parameters.

        Raises:
            KeyError: Not a valid parameter name.
        """
        for key, val in kwargs.items():
            if key not in self.__dict__:
                raise KeyError(
                    f"'{key}'' is not an existing parameter."
                    f"Use `.add({key} = {val})` instead f you want to add a new parameter.")
        self.__dict__.update(kwargs)

    def add(self, **kwargs: Any) -> None:
        """
        Add one or more new parameters.
        """
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        """
        Returns the class objects as a string.
        Returns:
            str: Pretty printed dictionary contents.
        """
        return print_dict(self.__dict__)
