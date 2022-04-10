

# `pykanto` & HPC

## Introduction

Many of the tasks that `pykanto` carries out are computationally intensive,
such as calculating spectrograms and running dimensionality reduction and clustering algorithms. High-level, interpreted languages—like
R or Python—are notoriously slow: where possible, I have optimised performance
by both a) translating functions to optimized machine code at runtime using
[Numba](https://numba.pydata.org/) and b) parallelising tasks using
[Ray](https://www.ray.io/), a platform for distributed
computing. As an example, the `segment_into_units()` function can find
and segment 20.000 discrete acoustic units in approximately 16 seconds on a
desktop 8-core machine; a dataset with over half a million (556.472) units takes
~132 seconds on a standard 48-core compute node.

`pykanto` works in any (well, hopefully) personal machine, but for most real-world
applications you will probably want to run it on a compute cluster. This can be a
daunting task, so I have packaged some tools that should make it a little bit
easier—at least they do for me!


[Slurm](https://slurm.schedmd.com/documentation.html) is still the most popular
job scheduler used in compute clusters, so the following instructions and tips
refer to it.

## Using `pykanto` in a HPC cluster

This library uses [Ray](https://www.ray.io/) for parallel/distributed computation. Ray provides tools to 'go from a single CPU to multi-core, multi-GPU or multi-node'. Submitting jobs that use multiple nodes or multiple GPUs is slightly more involved than using single-core or multi-core jobs. This might be overkill for some users, but if you need it—for example if you are training large models, or if you have a truly large dataset—

````{admonition} Tip: testing your code on your local machine first
:class: tip, dropdown

Before you run a large job on a compute cluster you might want to test your
parameters (e.g., for spectrogramming and segmentation) on a local machine. To
do this, you can test-build a `pykanto` dataset from a subset of your data -
large enough to be representative, but small enough to run quickly on your
machine. There are two ways to do this in `pykanto`:

To use a random subset,

```{code-block} python
:linenos:

dataset = SongDataset(... , random_subset=200)
```
To use a slice of the data:

```{code-block} python
:linenos:

params = Parameters(... , subset=(100, 300))
dataset = SongDataset(... , parameters=params)
```
````


See [source code by Peng Zhenghao](https://github.com/pengzhenghao/use-ray-with-slurm). Also see [ray instructions](https://docs.ray.io/en/ray-1.1.0/cluster/slurm.html)


1. Add this to the top of the script you want to run, right after any imports:

```{code-block} python
:linenos:

redis_password = sys.argv[1]
ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
print(ray.cluster_resources())
```

2. Request resources the same way you would normally do, e.g. an interactive session in one node with an NVIDIA v100 GPU with `srun -p interactive --x11 --pty --gres=gpu:v100:1 --mem=90000 /bin/bash`. How you do this exactly will depend on which particular system you are using.

3. You can run `pykanto-slaunch --help` to see which arguments you can pass to pykanto-slaunch.
   
4. A sumbission will look something like this:
   ```{code-block} bash
   pykanto-slaunch --exp greti2021 --p short --time 00:30:00 -n 1 --memory 40000 --gpu 1 --c "python 0.0_build-dataset.py"
   ```
   This will create a bash (.sh) file and a  log (.log) file in a `/logs` folder within the directory from which you are calling the script.

1. Check the logfile for errors!


````{admonition} Tip: uploading data to your cluster storage area
:class: tip, dropdown

If you need to upload your raw or segmented data to use in a HPC cluster and you
have lots of small files you may want to consider creating a `.tar` file to
reduce overhead. `pykanto` has a simple wrapper function to do this:

```{code-block} python
:linenos:

from pykanto.utils.write import make_tarfile 
out_dir = DIRS.SEGMENTED / 'JSON.tar.gz'
in_dir = DIRS.SEGMENTED / 'JSON'
make_tarfile(out_dir, in_dir)
```
````