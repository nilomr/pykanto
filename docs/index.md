

![pykanto-docs](custom/pykanto-logo-grey-04-docs.svg)

`pykanto` is a new software package to make the process of archiving,
cleaning, segmenting, labelling, and analysing animal vocalisations faster and
more reproducible. This website contains its documentation.<br>
[`This article`](https://arxiv.org/pdf/2302.10340v1.pdf) explains the motivation behind the project and provides a worked example using
it to analyse bird songs.

![Python 3.8](https://img.shields.io/badge/_python-≥3.8-lighgreen.svg)
![version](https://img.shields.io/badge/_version-0.1.0-orange.svg)

<div id="main-page">

```{eval-rst}

.. toctree::
   :caption: User guide
   :maxdepth: 1
   
   contents/installation
   contents/project-setup
   contents/basic-workflow
   contents/segmenting-files
   contents/segmenting-vocalisations
   contents/kantodata-dataset
   contents/interactive-app
   contents/hpc
   contents/feature-extraction
   contents/deep-learning
   contents/FAQs

```
# Modules

```{eval-rst}

.. autosummary::
   :caption: API reference
   :toctree: _autosummary
   :recursive:

   pykanto.signal
   pykanto.parameters
   pykanto.dataset
   pykanto.app
   pykanto.utils
   pykanto.plot

```

</div>