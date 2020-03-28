# Read and parse annotations (manual segmentation, labels, etc)


# Parse .data file generated by AviaNZ-2.1 -- which contains segment timestamps and other fuckery

# maybe all this should go to audio.segmentation instead

import numpy as np
import wavio
import os, json
import wave

def extractSegments(wavFile, destination, copyName, species):
    """
    [Extracts the sound segments given the annotation and the corresponding wav file
        Version 1.3 23/10/18
        Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis
        Copyright (C) 2017--2018
        Changes (c) Nilo M. Recalde]
    
    Arguments:
        wavFile {[type]} -- [path to .wav file]
        destination {[type]} -- [destination path]
        copyName {[type]} -- [not properly documented!]
        species {[type]} -- [description]
    """    
    
    datFile = wavFile+'.data'
    try:
        wavobj = wavio.read(wavFile)
        sampleRate = wavobj.rate
        data = wavobj.data
        if os.path.isfile(datFile):
            with open(datFile) as f:
                segments = json.load(f)
            cnt = 1
            for seg in segments:
                if seg[0] == -1:
                    continue
                if copyName:    # extract all - extracted sounds are saved with the same name as the corresponding segment in the annotation
                    filename = destination + '\\' + seg[4] + '.wav'
                    s = int(seg[0] * sampleRate)
                    e = int(seg[1] * sampleRate)
                    temp = data[s:e]
                    wavio.write(filename, temp.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)
                elif not species:   # extract all - extracted sounds are saved with the original file name followed by an index starting 1
                    ind = wavFile.rindex('/')
                    filename = destination + '\\' + str(wavFile[ind + 1:-4]) + '-' + str(cnt) + '.wav'
                    cnt += 1
                    s = int(seg[0] * sampleRate)
                    e = int(seg[1] * sampleRate)
                    temp = data[s:e]
                    wavio.write(filename, temp.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)
                elif species == seg[4][0]:   # extract only specific calls - extracted sounds are saved with with the original file name followed by an index starting 1
                    ind = wavFile.rindex('/')
                    ind2 = wavFile.rindex('\\')
                    filename = destination + '\\' + str(wavFile[ind2+1:ind]) + '-' + str(wavFile[ind + 1:-4]) + '-' + str(seg[4][0]) + '-' + str(cnt) + '.wav'
                    cnt += 1
                    s = int((seg[0]-1) * sampleRate)
                    e = int((seg[1]+1) * sampleRate)
                    temp = data[s:e]
                    wavio.write(filename, temp.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)
    except:
        print ("unsupported file: ", wavFile)




def extractSegments_batch(dirName, destination, copyName = True, species = None):
    """
    [Extracts the sound segments in a directory.
        Version 1.3 23/10/18
        Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis
        Copyright (C) 2017--2018
        Changes (c) Nilo M. Recalde]
    
    Arguments:
        dirName {[type]} -- [description]
        destination {[type]} -- [description]
    
    Keyword Arguments:
        copyName {bool} -- [Save with same name as segment in the annotation?] (default: {True})
        species {[type]} -- [Specify species if needed] (default: {None})
    """    
    
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav') and filename+'.data' in files:
                filename = root + '/' + filename
                extractSegments(filename, destination, copyName=copyName, species = species)



############# This now works - make a function!!

wavFile = "/home/nilomr/projects/00_gtit/data/200212-004.wav"
destination = "/home/nilomr/projects/00_gtit/data"

datFile = wavFile+'.data'
wavobj = wavio.read(wavFile)
sampleRate = wavobj.rate
data = wavobj.data

f = open(datFile) 
segments = json.load(f)[1:] # I had to remove the header - why? 
cnt = 1

for seg in segments:
    ind = wavFile.rindex('/')
    filename = destination + '/' + str(wavFile[ind + 1:-4]) + '-' + str(cnt) + '.wav'
    cnt += 1
    s = int(seg[0] * sampleRate)
    e = int(seg[1] * sampleRate)
    temp = data[s:e]
    wavio.write(filename, temp.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)


##########################################

# 1 - Make function to split wavs to processed data folder 
# (you need to create a directory in paths.py for this purpose)

# 2 - get data from the .data file, add coordinates and other information
# and make a nice, tidy .jason file following avgn format