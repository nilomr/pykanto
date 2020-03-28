Instructions
==============


#### To segment songs into bouts

 1. If it doesn't exist, `conda env create -f 0.0_great-tit-song-segment.yml`
 2. `conda activate 0.0_great-tit-song-segment`
 3. Navigate to `/dependencies/AviaNZ-2.1`
 4. If not done already, run `pip install -r requirements.txt --user`
 5. Build Cython extensions `cd ext; python3 setup.py build_ext -i; cd ..`
 6. Run `python3 AviaNZ.py`

 #### To segment bouts into syllables

  - 
  - 
  - 