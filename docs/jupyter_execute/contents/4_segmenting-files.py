#!/usr/bin/env python
# coding: utf-8

# # File segmentation

# ## Segmenting files with custom metadata fields

# 
# Let's say you are using
# [AudioMoth](https://www.openacousticdevices.info/audiomoth) recorders and want
# to retrieve some non-standard metadata from its audio files: (1) the device ID,
# and (2) the date and time time of an audio segment.
# 
# Here's how you can do it:

# In[1]:


import datetime as dt
import re
from typing import Any, Dict
import attr
from attr import validators
from dateutil.parser import parse
from pykanto.signal.segment import ReadWav, SegmentMetadata, segment_files
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import (get_file_paths, get_wavs_w_annotation,
                                 pykanto_data)
from pykanto.utils.types import Annotation
from pykanto.utils.write import makedir


# First, to make it easier to see what fields are available you can create a
# `ReadWav` object from a file and print its metadata, like so:

# In[2]:


# Loads a sample AudioMoth file, included with pykanto
DIRS = pykanto_data(dataset="AM")

wav_dirs = get_file_paths(DIRS.RAW_DATA, extensions=['.WAV'])
meta = ReadWav(wav_dirs[0]).all_metadata
print(meta)


# Now let's acess the metadata of interest and tell `pykanto` that we want to add
# these to the .JSON files and, later, to our database.
# 
# First, add any new attributes, along with their data type annotations and any
# validators to the Annotation class. This will make sure that your new
# attributes, or fields, are properly parsed.

# In[3]:


@attr.s
class CustomAnnotation(Annotation):
    rec_unit: str = attr.ib(validator=validators.instance_of(str))
    # This is intended as a short example, but in reality you could make sure that
    # this string can be parsed as a datetime object.
    datetime: str = attr.ib(validator=validators.instance_of(str))

Annotation.__init__ = CustomAnnotation.__init__


# Then, [monkey-patch](https://en.wikipedia.org/wiki/Monkey_patch) the
# `get_metadata` methods of the ReadWav and SegmentMetadata classes to add any
# extra fields that your project might require. This will save you from having to
# define the full classes and their methods again from scratch. Some people would
# say this is ugly, and I'd tend to agree, but it is the most concise way of doing
# this that I could think of that preserves enough flexibility.

# In[4]:


def ReadWav_patch(self) -> Dict[str, Any]:
    comment = self.all_metadata['tags'].comment[0]
    add_to_dict = {
        'rec_unit': str(re.search(r"AudioMoth.(.*?) at gain", comment).group(1)),
        'datetime': str(parse(re.search(r"at.(.*?) \(UTC\)", comment).group(1)))
    }
    return {**self.metadata.__dict__, **add_to_dict}


def SegmentMetadata_patch(self) -> Dict[str, Any]:
    start = self.all_metadata.start_times[self.index] / self.all_metadata.sample_rate
    datetime = parse(self.all_metadata.datetime) + dt.timedelta(seconds=start)
    add_to_dict = {
        'rec_unit': self.all_metadata.rec_unit,
        'datetime': str(datetime),
    }
    return {**self.metadata.__dict__, **add_to_dict}


ReadWav.get_metadata = ReadWav_patch
SegmentMetadata.get_metadata = SegmentMetadata_patch


# Now you can segment your annotated files like you would normally do - their
# metadata will contain your custom fields.

# In[5]:


wav_filepaths, xml_filepaths = [get_file_paths(
    DIRS.RAW_DATA, [ext]) for ext in ['.WAV', '.xml']]
files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)

wav_outdir, json_outdir = [makedir(DIRS.SEGMENTED / ext)
                           for ext in ["WAV", "JSON"]]

segment_files(
    files_to_segment,
    wav_outdir,
    json_outdir,
    parser_func=parse_sonic_visualiser_xml
)


# Note: if you want to run this in paralell with ray (as in
# `segment_files_parallel`) monkey-patching will not work: for now, you will have
# to properly extend `ReadWav` and `SegmentMetadata`.
