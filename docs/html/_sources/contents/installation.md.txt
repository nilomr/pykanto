# Installing `pykanto`

### Basic installation

To install `pykanto` using `pip`, run:

```bash
pip install pykanto
```

````{admonition} Tip: avoid a [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell)!
:class: tip

It is possible to use pip and install pykanto outside of a virtual environment, but this is not advised:

```{epigraph}
Virtual environments create a clean Python environment that does not interfere with any existing system installation, can be easily removed, and contain only the package versions your application needs. They help avoid a common challenge known as dependency hell.

--[ scikit-image docs](https://github.com/scikit-image/scikit-image/blob/main/INSTALL.rst)

I highly recommend that you create a new environment, for example with conda: 
```bash
conda create -n pykanto-env python==3.8
conda activate pykanto-env      
```
And only then install `pykanto`.
````

### Installing GPU/ML libraries

Under the hood, `pykanto` uses some libraries and algorithms (like UMAP) that
can run much faster on a GPU. If you have a CUDA-supported GPU and install
[RAPIDS](https://rapids.ai/)'s [cuML](https://github.com/rapidsai/cuml) in the
same environment, `pykanto` will detect it and switch to the faster GPU
implementation.

I recommend that, if this is the case, you first create a fresh environment with
conda:

```bash
conda create -n pykanto-env python==3.8             
```

And then install cuML & `pykanto`, in that order.

```bash
conda install -c rapidsai -c nvidia -c conda-forge cuml 
pip install pykanto
```
(Also see the [rapids release
selector](https://rapids.ai/start.html#rapids-release-selector).)

The same is true if you want to install [pytorch](https://pytorch.org/); the
preferred order would be:
```bash
# pytorch and cuML installation via conda
conda install -c rapidsai -c nvidia -c conda-forge cuml
conda install -c pytorch pytorch torchvision   
pip install pykanto
```

### Developer installation

```bash
git clone https://github.com/nilomr/pykanto.git
cd pykanto
pip install -e .'[dev]'
```
This will install extra dependencies, such as `pytest`, `nox` or `sphinx`, necessary for testing and generating the documentation.