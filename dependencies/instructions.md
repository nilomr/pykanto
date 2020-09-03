Instructions
==============

[TOC]

Nilo M. Recalde, 2020

### General requirements

Clone the following:

`https://github.com/timsainb/vocalization-segmentation.git`



#### To segment songs into songs

 1. `conda env create -f 0.0_great-tit-song-segment.yml`
 2. `conda activate 0.0_great-tit-song-segment`
 3. Navigate to `/dependencies/AviaNZ-2.1`
 4. Run `pip install -r requirements.txt --user`
 5. Build Cython extensions `cd ext; python3 setup.py build_ext -i; cd ..`
 6. Run `python3 AviaNZ.py`

Henceforth, `conda activate 0.0_great-tit-song-segment && j /dependencies/AviaNZ-2.1 && python3 AviaNZ.py`


**Note**: I haven't figured out how to change the config directory to `/dependencies/AviaNZ-2.1/Config`, so the `AviaNZconfig.txt` file has to be edited in `/home/[username]/.avianz/AviaNZconfig.txt`. 

My current settings in `AviaNZconfig.txt` are:

    # not complete, only relevant bits

    {
    "window_width": 1024,
    "incr": 300,
    "minFreq": 1900,
    "maxFreq": 9000,
    "maxSearchDepth": 50,
    "minSegment": 50,
    "DOC": true,
    "ReorderList": false,
    "secsSave": 60,
    "windowWidth": 27.5,
    "widthOverviewSegment": 20.0,
    "maxFileShow": 400,
    "fileOverlap": 15,
    "brightness": 83,
    "contrast": 88,
    "overlap_allowed": 5,
    "reviewSpecBuffer": 1,
    "cmap": "Viridis",
    "showAmplitudePlot": true,
    "showAnnotationOverview": true,
    "showPointerDetails": false,
    "readOnly": false,
    "transparentBoxes": true,
    "showListofFiles": true,
    "invertColourMap": false,
    "saveCorrections": true,
    "operator": "Nilo",
    "reviewer": "Nilo",
    "window": "Hann",
    }

> Took me 14 min to segment 1h of audio.
> 39.375 8-hour day days if segmenting only 2h per day and bird

> Took me 14 min to check 1 nestbox (21h of recordings).
> 39.375 8-hour day days if segmenting only 2h per day and bird

***

#### To segment songs into syllables

> Last updated 11 June 2020: If on debian-based linux, follow:

1. Clone the repository with `git clone https://github.com/CreanzaLab/chipper.git`

2. Create a new conda environment: `conda create -n chipper_env python=3.7 -y`

3. Activate the conda environment: `conda activate chipper_env` and avigate to `./chipper`.

4. `pip install --no-binary kivy kivy`. 

5. Run `pip install -r requirements.txt`

6. Install kivy packages

        garden install --kivy graph
        garden install --kivy filebrowser
        garden install --kivy matplotlib
        garden install --kivy progressspinner

7. Next, make the setup.py file that is missing in `./chipper`

        from setuptools import find_packages, setup

        setup(
            name='chipper',
            packages=find_packages(),
            version='1.0',
        )

    and run `python setup.py develop`

8. Navigate to `./chipper/chipper` and run `python run_chipper.py`  



<br>

#### To Do:
- [ ] Write function to get recordings from n mins before sunrise that day to n minutes after. 
- [ ] Add a 'global' progress bar to the batch segment function!
- [ ] Improve way to deal with year - either select one year of data or all years

