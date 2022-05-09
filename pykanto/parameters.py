# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""
Classes and methods to store and modify pykanto parameters.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────
from __future__ import annotations
import json
from typing import Any, Tuple
import attr
from pykanto.utils.compute import print_dict

# ──── CLASSES AND METHODS ─────────────────────────────────────────────────────


class Parameters:
    # TODO: #8 Parameters should be an attrs class with validation @nilomr
    """
    General parameters class - stores parameters to be passed to
    a :meth:`~pykanto.dataset.KantoData` object.
    """

    def __init__(self, **kwargs: Any):
        """
        Sets default parameters, then updates with user-provided parameters.
        """

        # Set defaults first

        # Spectrogramming
        self.window_length: int = 1024
        """Number of samples. Defaults to 1024."""
        self.hop_length: int = 128
        """Number of samples between successive frames. Defaults to 128."""
        self.fft_size: int = 2048
        """Number of bins used to divide the window. Defaults to 2048."""
        self.num_mel_bins: int = 224
        """Number of Mel frequency bins to use 
        (n of 'rows' in mel-spectrogram). Defaults to 224."""
        self.sr: int = 22050
        """The sampling rate (samples taken per second). 
        Defaults to 22050."""
        self.hop_length_ms: float = self.sr / self.hop_length
        """Calculated automatically."""
        self.fft_rate: float = 1000 / self.hop_length_ms
        """Calculated automatically."""
        self.top_dB: int = 65
        """Top dB to keep. Defaults to 65.
        Example: if ``top_dB`` = 65 anything below -65dB will be masked."""
        self.lowcut: int = 1000
        """Minimum frequency (in Hz) to include by default. Defaults to 1000."""
        self.highcut: int = 10000
        """Maximum frequency (in Hz) to include by default. 
        Defaults to 10000."""
        self.dereverb: bool = False
        """Wether to reduce reverberation in the spectrograms. 
        Defaults to False"""

        # Segmentation
        self.max_dB: int = -30
        """Maximum threshold to reach during segmentation, in dB. 
        Defaults to -30"""
        self.dB_delta: int = 5
        """Size of thresholding steps, in dB. Defaults to 5."""
        self.silence_threshold: float = 0.2
        """Threshold separating silence from voiced segments. 
        Between 0.1 and 0.3 tends to work well. Defaults to 0.2."""
        self.max_unit_length: float = 0.4
        """Maximum unit length allowed. Defaults to 0.4. """
        self.min_unit_length: float = 0.03
        """Minimum unit length allowed. Defaults to 0.03."""
        self.min_silence_length: float = 0.001
        """Minimum silence length allowed. Defaults to 0.001."""
        self.gauss_sigma: int = 3
        """Sigma for gaussian kernel. Defaults to 3."""

        # general settings
        self.song_level: bool = False
        """Whether to return the average of all units.
        Defaults to False (return individual units in each vocalisation)"""
        self.subset: Tuple[int, int] | None = None
        """Indices of the first and last items to include in the dataset. 
        Optional, defaults to None."""
        self.num_cpus: None | int = None
        """How many cpus to use for parallel computing. Default is available
        number of cpus"""
        self.verbose: bool = False
        """How much detail to provide. Defaults to False (not verbose)."""
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
                    f"Use `.add({key} = {val})` instead f you want to add a new parameter."
                )
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
