1. Clone the repository with `git clone https://github.com/CreanzaLab/chipper.git`

2. Create a new conda environment: `conda create -n chipper_env python=3.7 -y`

3. Activate the conda environment: `conda activate chipper_env` and avigate to `./chipper`.

4. `pip install --no-binary kivy kivy`. 

> **Note:** Installing kivy from pypi directly by `pip install -r requirements.txt` resulted in the audio not working. Installing ffpyplayer resulted in the audio playing only once (might be the same issue reported [here](https://github.com/kivy/kivy/issues/3845#issuecomment-555725947)).

5. Run `pip install -r requirements.txt`

> **Note:** `conda install -r requirements.txt` is not a valid conda command. I tried `conda install -c conda-forge anaconda --file requirements.txt` instead, but `soundfile` is not available in any conda channels.

>`conda install pypiwin32 kivy.deps.sdl2 kivy.deps.glew kivy.deps.gstreamer kivy.deps.glew_dev kivy.deps.sdl2_dev kivy.deps.gstreamer_dev` or its pip alternative also dont't work—and should't be necessary given that `kivy` is listed in the requirements file. Also, I'm guessing that pypiwin32 doesn't belong there.
4. Install kivy packages

        garden install --kivy graph
        garden install --kivy filebrowser
        garden install --kivy matplotlib
        garden install --kivy progressspinner


7. Next, make the setup.py file that is missing in `./chipper`

        from setuptools import find_packages, setup

        setup(
            name='chipper',
            packages=find_packages(),
            version='1.0',
        )

    and run `python setup.py develop`

8. Navigate to `./chipper/chipper` and run `python run_chipper.py`  