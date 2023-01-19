# FAQs & known issues

## Known issues:

- If using the autoreload ipython magic, you might get the following error when
  saving a `KantoData` dataset:
  ```
  Can't pickle <class 'pykanto.dataset.KantoData'>: it's not the same object as pykanto.dataset.KantoData
  ```
  Fix: restart your kernel without the autoreload extension.

- Ray throws a fork support INFO message when using grpcio==1.44.0
  ```Fork support is only compatible with the epoll1 and poll polling
  strategies
  ```
  Fix:: see ray issue [#22518](https://github.com/ray-project/ray/issues/22518)

- Ray workers crash:  
If your workers crash and/or your machine freezes it is likely that you have run
out of RAM. You could try using fewer cpus when eecuting paralell processes. You
can do this by, for example, setting `parameters.num_cpus` to a number lower
than your available CPUS. Example: If your KantoData object is called dataset,
then run `dataset.parameters.update(num_cpus=1)`

## System dependencies:
  
- `hdbscan` requires `gcc` to be installed. If you get an error when `hdbscan` is
being installed, then you might need to install `gcc` first, e.g. by running
`sudo apt-get install gcc`, and then try installing pykanto again.