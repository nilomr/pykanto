Hi, 

I have been trying chipper today, congratulations on what looks like a fantastic tool!
I run into a couple of minor issues when trying to get it to work, and I thought I would let you know in case it's useful:

**chipper_v1.0_linux**, system info [here](https://user-images.githubusercontent.com/60854434/78676196-06c70300-78de-11ea-9086-5703fc48a72d.png).

I downloaded the .zip file, extracted its contents and started chipper by running `start_chipper`, as per the `README.md`. It started normally, but the audio didn't work:

    [INFO   ] Setting up
    [INFO   ] Loading file foo.wav
    [CRITICAL] [AudioSDL2   ] Unable to open mixer: b'No such audio device'
    [WARNING] Deprecated property "<AliasProperty name=filename>" of object "<kivy.core.audio.audio_sdl2.SoundSDL2 object at 0x7f1166f84590>" was accessed, it will be removed in a future version
    [WARNING] [AudioSDL2   ] Unable to load foo.wav: b"Audio device hasn't been opened"

See the full output [here](https://github.com/CreanzaLab/chipper/files/4444506/output.txt) and a similar issue reported [here](https://github.com/kivy/kivy/issues/6536) and possible solutions [here](https://github.com/matham/ffpyplayer/issues/71). Everything else seems to work ok.

I then tried to install from source:
The instructions to install from source will not work (using conda, and for deb based Linux).

**The following worked for me:**

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

Everything works normally now, but the thresholded sonogram is not binary - i.e. it still has some red and yellow pixels, see example [here](https://user-images.githubusercontent.com/60854434/78705670-bb284f80-7905-11ea-84cd-71ef4753268e.png). It doesn't seem to affect usability, however.

I hope that is useful, thank you for putting the time to develop this tool! Will let you know if I run into any other issues.

Nilo