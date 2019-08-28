import time, os

def filename_with_datetime(filename, folder='./', extension=''):
    return os.path.join(folder,  filname+'__'+time.strftime("%Y_%m_%d-%H:%M:%S")+extension)
