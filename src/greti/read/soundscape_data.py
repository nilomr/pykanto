
import os
import time
from scipy import signal
import re
from src.avgn.utils.json import NoIndentEncoder
import json
import wave
import numpy as np
from pathlib2 import Path
from src.greti.read.paths import safe_makedir
import audio_metadata
import datetime as dt
from multiprocess import Pool, cpu_count
from tqdm.auto import tqdm


def pseudo_psd(spectrogram):
    psd = np.mean(spectrogram, axis=1)
    psd = (psd - psd.min()) / (psd - psd.min()).sum()
    return psd


def normalise_01(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def save_mean_chunks(DATA_DIR, DATASET_ID, wavfile, average_over_min=5, normalise=False):

    with wave.open(str(wavfile)) as wav:
        datetime = dt.datetime.strptime(wavfile.stem, "%Y%m%d_%H%M%S")
        frames = wav.getnframes()
        sr = wav.getframerate()
        data = np.frombuffer(wav.readframes(frames), dtype=np.int16)
        average_over_s = average_over_min * 60
        segment_n = (frames / sr) / average_over_s
        segment_len = int(frames / segment_n)
        for start, end in zip(range(0, frames, segment_len), range(segment_len, frames, segment_len)):
            chunk = data[start:end]
            _, _, spectrogram = signal.spectrogram(chunk, sr)
            if normalise:
                avg_spec = normalise_01(pseudo_psd(spectrogram))
            else:
                avg_spec = pseudo_psd(spectrogram)
            # make a JSON dictionary to go with the .wav file
            seg_datetime = datetime + dt.timedelta(seconds=start / sr)
            meta = audio_metadata.load(wavfile)
            tags = audio_metadata.load(wavfile)["tags"].comment[0]
            audiomoth = re.search(
                r"AudioMoth.(.*?) at gain", tags).group(1)
            json_dict = {}
            json_dict["nestbox"] = wavfile.parts[-2]
            json_dict["recorder"] = audiomoth
            json_dict["recordist"] = "Nilo Merino Recalde"
            json_dict["source_datetime"] = str(datetime)
            json_dict["datetime"] = str(seg_datetime)
            json_dict["date"] = str(seg_datetime.date())
            json_dict["time"] = str(seg_datetime.time())
            json_dict["timezone"] = "UTC"
            json_dict["samplerate_hz"] = sr
            json_dict["length_s"] = len(chunk) / sr
            json_dict["bit_depth"] = meta["streaminfo"].bit_depth
            json_dict["tech_comment"] = tags
            json_dict["source_loc"] = wavfile.as_posix()
            json_dict["spec"] = avg_spec.tolist()

            # Dump json
            json_txt = json.dumps(
                json_dict, cls=NoIndentEncoder, indent=2)
            json_out = (
                DATA_DIR
                / "processed"
                / DATASET_ID
                / "JSON"
                / (wavfile.parts[-2] + "-" + ''.join(e for e in str(seg_datetime) if e.isalnum()) + ".JSON")
            )
            # Save .json
            safe_makedir(json_out.as_posix())
            f = open(json_out.as_posix(), "w")
            print(json_txt, file=f)
            f.close()


def batch_save_mean_chunks(DATA_DIR, DATASET_ID, origin, time_range=(4, 10), average_over_min=5, normalise=False):
    """Extracts all sound segments found in a folder/subfolders. Uses multiprocessing.

    Args:

        origin (PosixPath): folder with raw data to be segmented
        DATA_DIR (Posixpath): Path to higher-level data folder
        DT_ID (str): Identifier for dataset ("%Y-%m-%d_%H-%M-%S")
        subset (str, optional): Label to export. Defaults to "GRETI_HQ"
    """
    for root, dirs, files in os.walk(str(origin)):

        p = Pool(processes=cpu_count() - 4)
        start = time.time()

        for wav in tqdm(
            files, desc="{Reading, averaging and saving segments}", position=0, leave=True,
        ):
            if wav.endswith(".wav") or wav.endswith(".WAV"):
                try:
                    wavfile = Path(root) / wav
                    # Check time
                    datetime = dt.datetime.strptime(
                        wavfile.stem, "%Y%m%d_%H%M%S")
                    if datetime.time().hour >= time_range[0] and datetime.time().hour <= time_range[1]:
                        p.apply_async(
                            save_mean_chunks,
                            args=(DATA_DIR, DATASET_ID, wavfile),
                            kwds={"average_over_min": average_over_min,
                                  "normalise": normalise},
                        )
                except:
                    print(f'There is an issue with {wav}')

        p.close()
        p.join()
        print("Complete")
        end = time.time()
        print("total time (s)= " + str(end - start))
