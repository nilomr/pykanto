

Vocalisation segmentation
=========================


Using :meth:`~pykanto.dataset.SongDataset.segment_into_units` finding and
segmenting 20.000 units takes around 16 seconds in a desktop 16-core machine,
and in my tests segmenting a dataset with over half a million units (556.472)
took just 132 seconds on a 48-core compute node.



Function 'segment_into_units' took 16.7364 sec.