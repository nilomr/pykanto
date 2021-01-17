# %%
# Set year
import numpy as np
import pandas as pd
from src.greti.read.paths import DATA_DIR

# %%
# import recorded nestboxes
YEAR = "2020"
files_path = DATA_DIR / "raw" / YEAR
filelist = np.sort(list(files_path.glob("**/*.WAV")))
recorded_nestboxes = pd.DataFrame(set([file.parent.name for file in filelist]))


DATASET_ID = "GRETI_HQ_2020"  # Name of output dataset

# %%

# TODO: resample so they are more evenly spaced
