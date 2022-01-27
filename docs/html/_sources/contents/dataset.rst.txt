
Creating a SongDataset object
=============================

Parameters
----------

The print function will return the contents of the dataset. can also do with
parameters or directories

Most computationally intensive methods will have three versions:
``method`` (single)
``method_r`` (n elements), remote
``method_parallel``, parallelised/distributed w/ a ray cluster


.. code-block:: python
    :linenos:

    print(dataset.parameters)


To get some basic info on the data in the dataset:

.. code-block:: python
    :linenos:
    
    # Check dataset length
    dataset.sample_info()
.. code-block:: none

    Total length: 800
    Vocalisations: 740
    Noise: 60
    Unique IDs: 3

Plot some info 

.. code-block:: python
    :linenos:

    # Plot some information about the dataset
    dataset.summary_plot(variable='all')


Check sample size per individual ID in the dataset:

.. code-block:: python
    :linenos:

    dataset.vocalisations['ID'].value_counts()

.. code-block:: none

    B119    159
    B108    157
    B163    134
    B226    134
    B216    117
    

Loading an existing dataset

.. code-block:: python
    :linenos:
    
    DATASET_ID = "WYTHAM_GRETIS_2021_TEST"
    out_dir = DIRS.DATA / "datasets" / DATASET_ID / f"{DATASET_ID}.db"
    dataset = pickle.load(open(out_dir, "rb"))


Creating a dataset for which there is already derived data (e.g. spectrograms).
This is something that might happen if, say, creating a dataset fails but
at least some spectrogram files were saved succesfully. 

.. code-block:: python
    :linenos:

    DATASET_ID = "BIGBIRD"
    dataset = SongDataset(DATASET_ID, DIRS, parameters=params,
                        overwrite_dataset=True, overwrite_data=False)


Note: 
    You can use any matplotlib palette here using the 'cmap' argument.
    See `colourmaps`_.

.. _colourmaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html


