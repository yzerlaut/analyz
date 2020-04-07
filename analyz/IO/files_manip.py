import os
import numpy as np

def get_files_with_given_exts(dir='./', EXTS=['npz','abf','bin']):
    """ get files of a given extension and sort them..."""
    FILES = []
    for ext in EXTS:
        for file in os.listdir(dir):
            if file.endswith(ext):
                FILES.append(os.path.join(dir, file))
    return np.array(FILES)

def rename_files_for_easy_sorting(dir='./'):
    """ get files and sort them..."""
    FILES = []
    LIST_OF_FILES = os.listdir(dir)
    # we first rename it, to sort them easily
    for f in LIST_OF_FILES:
        fs = f.split('_')
        for i in range(len(fs)):
            try:
                if (int(fs[i])<10) and (len(fs[i])<2):
                    fs[i] = '0'+fs[i]
            except ValueError:
                pass
        s1, s2 = os.path.join(dir,f), os.path.join(dir,'_'.join(fs))
        if s1!=s2:
            os.rename(s1, s2)
            print("RENAMED", f, '_'.join(fs))

if __name__ == '__main__':
    import sys
    foldername = sys.argv[-1]
    rename_files_for_easy_sorting(dir=foldername)
    # get_files_with_given_exts(dir=foldername, EXTS=['bin'])
