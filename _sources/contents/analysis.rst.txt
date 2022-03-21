Vocalisation analysis
=====================


Pykanto is designed to provide a platform that enables researchers to run and
store any analyses they might need with ease and in a reproducible way. The
precise nature of these analyses will vary greatly, so pykanto's aim is not to
provide functions or methods to, for example, extract audio features—there already are
other, much better libraries for that.

These are some examples that show one way in which you can extract and store
features from vocalisations in a dataset created with pykanto:



.. code-block:: python
    :linenos:

centroid, bandwidth = spec_centroid_bandwidth(dataset, key=key, plot=True)
minfreqs, maxfreqs = approximate_minmax_frequency(dataset, key=key, plot=True)



# Add output