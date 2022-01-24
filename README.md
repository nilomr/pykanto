![version](https://img.shields.io/badge/package_version-0.1.0-orange)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)
![license](https://img.shields.io/github/license/mashape/apistatus.svg)
![Open Source Love](https://img.shields.io/badge/open%20source%3F-yes!-lightgrey)
![Python 3.6](https://img.shields.io/badge/python-3.6-brightgreen.svg)
![R](https://img.shields.io/badge/R-3.6.2-informational)

***

![Wytham Woods, Oxford](/resources/README_img/wytham_net.jpg)

This repository contains code to segment, label and analyse the songs of great tits (*Parus major*) recorded in Wytham Woods, Oxford, 2020- .

#### Table of contents
  - [Installation](#installation)
  - [Project Organisation](#project-organisation)
  - [Acknowledgements](#acknowledgements)

## Installation

Avoid a [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell)!
 > While it is possible to use pip without a virtual environment, it is not advised: virtual environments create a clean Python environment that does not interfere with any existing system installation, can be easily removed, and contain only the package versions your application needs. They help avoid a common challenge known as dependency hell.

- Both pytorch and cuml should probably be installed by te user via conda - a nightmare otherwise:

```
# pytorch and cuml installation via conda
conda install -c pytorch pytorch torchvision torchaudio   
conda install -c rapidsai -c nvidia -c conda-forge cuml 
```

- git (gitpython) not a dependency but recommended to find project root. Provide alternatives also! 
```
conda install -c conda-forge gitpython
```

- Do not upgrade/downgrade pandas within a project/environment: see https://stackoverflow.com/questions/68625748/spark-attributeerror-cant-get-attribute-new-block-on-module-pandas-core-in

- Show hot to start a project and then symlink the data folder to an external drive, configure git for version control, etc

# Known issues:
If using the autoreload ipython magic, might get the following error:
`Can't pickle <class 'pykanto.dataset.SongDataset'>: it's not the same object as pykanto.dataset.SongDataset`
Fix: restart your kernel without the autoreload extension.

1. Clone the repository:
`git clone https://github.com/nilomr/wytham-maps.git`.
2. Install source code:
`pip install .` (install) or `pip install -e .` (developer install).
3. Follow the instructions in the [docs](/docs) in the correct order.


# Download xeno-canto data
```
pip install xeno-canto
```

xml along these lines - one option
```
<?xml version="1.0" encoding="UTF-8"?>
<sv>
  <data>
    <model id="1" name="" sampleRate="48000" start="5767168" end="163020800" 
    type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="0" 
    subtype="box" minimum="1811.16" maximum="5665.32" units="Hz" />
    <dataset id="0" dimensions="2">
      <point frame="5767168" value="3672.92" duration="1499136" 
        extent="1698.44" label="NOISE" />
      <point frame="90030976" value="2219.44" duration="103040" 
        extent="3070.26" label="" />
      <point frame="90445824" value="2284.77" duration="77056" 
        extent="3086.59" label="" />
    </dataset>
  </data>
  <display>
    <layer id="2" type="boxes" name="Boxes" model="1"  verticalScale="0"  
    colourName="White" colour="#ffffff" darkBackground="true" />
  </display>
</sv>
```
## Project Organisation


    ├── LICENSE
    │
    ├── README.md          <- The top-level README.
    │
    ├── data               <- Main data folder. It is not version-tracked-the relevant program(s)  
    │   ├── external          create it automatically.
    │   ├── interim        
    │   ├── processed      
    │   └── raw            
    │
    ├── dependencies       <- Files required to reproduce the analysis environment(s).
    │
    ├── docs               <- Project documentation (installation instructions, etc.).
    │
    ├── notebooks          <- Jupyter and Rmd notebooks. Naming convention is number (for ordering),
    |   |                     the creator's initials, and a short `-` delimited description, e.g.
    │   └── ...               `1.0_nmr-label-songs`.  
    │                         
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. Not currently tracked;
    |   |                     created automatically when needed.
    │   └── figures        <- Generated figures.
    │
    ├── setup.py           <- Makes project pip installable (so pykanto can be imported).
    |
    ├── ...                <- R project, .gitignore, etc.
    │
    └── pykanto                <- Source code. Install with `pip install .` (install) or 
                              `pip install -e .` (developer install).

## Acknowledgements


- `Many functions in pykanto/avgn` 
Sainburg T, Thielk M, Gentner TQ (2020) Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires. PLOS Computational Biology 16(10): e1008228. [DOI](https://doi.org/10.1371/journal.pcbi.1008228)

- `pykanto/vocalseg` Code modified from Tim Sainburg. [Original repository](https://github.com/timsainb/vocalization-segmentation).

- The `dereverberate()` function, [here](https://github.com/nilomr/great-tit-song/blob/24d9527d0512e6d735e9849bc816511c9eb24f99/pykanto/greti/audio/filter.py#L66), is based on code by Robert Lachlan ([Luscinia](https://rflachlan.github.io/Luscinia/)).

--------

<p><small>A project by Nilo M. Recalde | based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
