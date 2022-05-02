

Vocalisation segmentation
=========================


Using :meth:`~pykanto.dataset.KantoData.segment_into_units` finding and
segmenting 20.000 units takes around 16 seconds in a desktop 16-core machine,
and in my tests segmenting a dataset with over half a million units (556.472)
took just 132 seconds on a 48-core compute node.

`pykanto` includes three sample datasets:

1. Great tit songs

2. European storm-petrel purr songs

XC46092 © Dougie Preston // Shetland, United Kingdom
XC663885 © Simon S. Christiansen // Mykines, Denmark

3. Bengalese finch songs



.. code-block:: python
   :linenos:

    DATA_PATH = Path(pkg_resources.resource_filename('pykanto', 'data'))
    PROJECT = Path(DATA_PATH).parent
    RAW_DATA = DATA_PATH / 'raw' / DATASET_ID
    DIRS = ProjDirs(PROJECT, RAW_DATA, mkdir=True)

.. code-block:: python
   :linenos:

    # Get files to segment and segment them
    wav_filepaths, xml_filepaths = [get_file_paths(
        DIRS.RAW_DATA, [ext]) for ext in ['.wav', '.xml']]

    files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)

.. code-block:: python
   :linenos:

    segment_files_parallel(
        files_to_segment,
        DIRS,
        resample=None,
        parser_func=parse_sonic_visualiser_xml,
        min_duration=.5,
        min_freqrange=200,
        labels_to_ignore=["NOISE"]
    )