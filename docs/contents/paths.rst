.. highlight:: python
   :linenothreshold: 1
   
Working with paths and directories
==================================

pykanto provies a convenient way to store all paths under a single roof
so that you can keep track of stuff

.. highlight:: python
   :linenothreshold: 1


ProjDirs()
then you can .add(NEWPATH, Path(...))
Peek it with , or if in dataset object ``print(dataset.DIRS)``

.. code-block:: python

    from pykanto.utils.paths import ProjDirs

    print(DIRS) # Print the directories currently held in DIRS
    print(dataset.DIRS)

If you need to upload your raw or segmented data and you have lots of small files
you may want to consider creating a ``tar`` file to reduce overhead. 

.. code-block:: python

    from pykanto.utils.write import make_tarfile 
    out_dir = DIRS.WAVFILES / 'JSON.tar.gz'
    in_dir = DIRS.WAVFILES / 'JSON'
    make_tarfile(out_dir, in_dir)

If you need to change the location of your raw data before you create a dataset,
you can use the following method:

.. code-block:: python

    DIRS.update_json_locs(PROJECT_DIR)

You can safely try this even if you have not moved your data; you will just get 
a message ``Files exist: no need to update paths. You can force update by setting `overwrite_dataset = True`.``.


If you need to change the location of your dataset,

.. code-block:: python

