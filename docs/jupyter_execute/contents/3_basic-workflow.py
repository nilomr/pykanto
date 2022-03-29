#!/usr/bin/env python
# coding: utf-8

# # Basic workflow

# In[2]:


from pykanto.utils.paths import pykanto_data, ProjDirs
from pykanto.dataset import SongDataset
from pykanto.parameters import Parameters


# We are going to load one of the very small sample datasets that are packaged with `pykanto`—this will be enough for a first test to familiarise yourself with the package. See [working with paths and directories](../contents/2_paths-and-dirs.md) to learn how to load your own data.
# 
# These are a few songs from two male great tits (_Parus major_) in [my study population](http://wythamtits.com/) in Wytham Woods, Oxfordshire, UK.

# In[3]:


DATASET_ID = "GREAT_TIT"
DIRS = pykanto_data(dataset=DATASET_ID)
print(DIRS)


# Now we can create a SongDataset object, which is the main class in `pykanto` and acts as a sort of database.

# In[12]:



params = Parameters() # Using default parameters, which you should't
dataset = SongDataset(DATASET_ID, DIRS, parameters=params)
dataset.vocs.head()


# We now have an object `dataset`, which is an instance of the `SongDataset` class and has all of its methods. For example, you might want to segment 

# In[11]:


dataset.segment_into_units()
for voc in dataset.vocs.index[:1]:
    dataset.plot_voc_seg(voc)


# In[ ]:


# for song_level in [True, False]:
#     dataset.parameters.update(song_level=song_level)
#     dataset.get_units()

# dataset.reload()
# for song_level in [True, False]:
#     dataset.parameters.update(song_level=song_level)
#     dataset.cluster_ids(min_sample=5)

# for song_level in [True, False]:
#     dataset.parameters.update(song_level=song_level)
#     dataset.prepare_interactive_data()

# dataset.parameters.update(song_level=True)
# dataset.open_label_app()


# If need to load an existing dataset:
# (This needs you to create a ProjDirs object)

# In[ ]:


# out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
# dataset = pickle.load(open(out_dir, "rb"))

