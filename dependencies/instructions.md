Instructions
==============

[TOC]

Nilo M. Recalde, 2020


#### To segment songs into bouts

 1. `conda env create -f 0.0_great-tit-song-segment.yml`
 2. `conda activate 0.0_great-tit-song-segment`
 3. Navigate to `/dependencies/AviaNZ-2.1`
 4. Run `pip install -r requirements.txt --user`
 5. Build Cython extensions `cd ext; python3 setup.py build_ext -i; cd ..`
 6. Run `python3 AviaNZ.py`

**Note**: I haven't figured out how to change the config directory to `/dependencies/AviaNZ-2.1/Config`, so the `AviaNZconfig.txt` file has to be edited in `/home/[username]/.avianz/AviaNZconfig.txt`. 

My current settigs are:

    {
    "window_width": 1024,
    "incr": 300,
    "minFreq": 1900,
    "maxFreq": 9000,
    "maxSearchDepth": 50,
    "minSegment": 50,
    "drawingRightBtn": false,
    "specMouseAction": 3,
    "StartMaximized": true,
    "MultipleSpecies": false,
    "RequireNoiseData": false,
    "DOC": true,
    "ReorderList": false,
    "SoundFileDir": "./Sound Files",
    "secsSave": 60,
    "windowWidth": 27.5,
    "widthOverviewSegment": 20.0,
    "maxFileShow": 400,
    "fileOverlap": 15,
    "brightness": 83,
    "contrast": 88,
    "overlap_allowed": 5,
    "reviewSpecBuffer": 1,
    "BirdListShort": "/home/nilomr/projects/0.0_great-tit-song/dependencies/AviaNZ-2.1/Config/BirdList.txt",
    "BirdListLong": "/home/nilomr/projects/0.0_great-tit-song/dependencies/AviaNZ-2.1/Config/BirdList.txt",
    "RecentFiles": [
    "/home/nilomr/projects/0.0_great-tit-song/data/raw/2020/B3/20200423_030000.WAV",
    "/home/nilomr/projects/0.0_great-tit-song/data/raw/2020/B3/20200423_040000.WAV",
    "/home/nilomr/projects/0.0_great-tit-song/data/raw/2020/B3/20200423_050000.WAV",
    "/home/nilomr/projects/0.0_great-tit-song/data/raw/2020/B3/20200424_080000.WAV"
    ],
    "ColourList": [
    "Grey",
    "Viridis",
    "Inferno",
    "Plasma",
    "Autumn",
    "Cool",
    "Bone",
    "Copper",
    "Hot",
    "Jet",
    "Thermal",
    "Flame",
    "Yellowy",
    "Bipolar",
    "Spectrum"
    ],
    "ColourSelected": [
    0,
    0,
    255,
    100
    ],
    "ColourNamed": [
    34,
    186,
    0,
    90
    ],
    "ColourNone": [
    255,
    0,
    0,
    100
    ],
    "ColourPossible": [
    255,
    255,
    0,
    100
    ],
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
    "protocolOn": false,
    "protocolSize": 10,
    "protocolInterval": 60,
    "fs_start": 0,
    "fs_end": 0,
    "window": "Hann",
    "FiltersDir": "Filters"
    }



***

#### To segment bouts into syllables

> Last updated 11 June 2020: If on ubuntu/debian-based linux, follow:

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
- [ ] Add to bird list: "song matching", "2nd GT", "Blue tit" etc
- [ ] 
- [ ] 
- [ ] 
- [ ] 

