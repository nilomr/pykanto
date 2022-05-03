# FAQs & known issues

# Known issues:

- If using the autoreload ipython magic, might get the following error:
`Can't pickle <class 'pykanto.dataset.KantoData'>: it's not the same object as pykanto.dataset.KantoData`
Fix: restart your kernel without the autoreload extension.

- Ray throws a fork support INFO message `Fork support is only compatible with the epoll1 and poll polling strategies` with grpcio==1.44.0. See ray issue [#22518](https://github.com/ray-project/ray/issues/22518)

- Ray workers crash:  
If your workers crash and/or your machine freezes it is likely that you have run
out of RAM. Try using fewer cpus when running in parallel. You can do this by,
for example, setting `parameters.num_cpus` to a number lower than your available
CPUS. Example: If your KantoData object is called dataset, then run `dataset.parameters.update(num_cpus=1)`

