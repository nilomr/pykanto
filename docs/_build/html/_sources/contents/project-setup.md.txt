
# Setting up a project


## Working with paths and directories

`pykanto` provides a convenient way to store all paths pointing to directories
and files in your project together: this makes it easier to access them, and
promotes standardisation among your projects.


```{code-block} python
---
caption: First, import any dependencies
---

from pathlib import Path
from pykanto.utils.paths import link_project_data, ProjDirs

```

## 1. Link your data

Find your project's root directory. You can do this in any number of ways, as
long as you do it programmatically. For example, the code below assumes that you
are doing version control with [git](https://git-scm.com/) and simply gets the root of your repository:

```{code-block} python

project_root = Path(
    git.Repo(".", search_parent_directories=True).working_tree_dir
)

```

It is common to have your raw data on a large external drive or remote server
(for example I use a RAID system). If this is the case for you, you probably
want to link the actual location of your raw data to an otherwise empty `/data`
folder in your project for ease of access and clarity. Pykanto includes a
function to do just that:

```{code-block} python

external_data = Path('path/to/your/data/drive')
link_project_data(external_data, project_root / 'data')

```

```{admonition} Tip: freeze your raw data and only work on programmatically derived datasets
:class: tip

You wil likely create different derived datasets from the same raw data, and that is why pykanto lets your (raw) data live wherever you want.
I **strongly** recommend that you make its directory read-only and
never ever touch it.
```

## 2. Set up project directories

Next, tell `pykanto` where the raw data for your project live, 

```{code-block} python
dataset_name = 'BIGBIRD_2021'
data_dir = project_root / "data" / "raw" / dataset_name
```

and build the project's directory tree:
 

```{code-block} python

DIRS = ProjDirs(project_root, data_dir, dataset_name,  mkdir=True)
print(DIRS)
```
See the
{py:class}`~pykanto.utils.paths.ProjDirs` docs for more information.
<br>
**Now you are ready to import and segment your raw data (see next section).**