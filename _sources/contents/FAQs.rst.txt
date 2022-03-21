FAQs
=============================


Ray workers crash
-----------------

If your workers crash and/or your machine freezes it is likely that you have run
out of RAM. Try using fewer cpus when running in parallel. You can do this by,
for example, setting `parameters.num_cpus` to a number lower than your available
CPUS. Example: If your SongDataset object is called dataset, then:

.. code-block:: python
    :linenos:

    dataset.parameters.update(num_cpus=1)
