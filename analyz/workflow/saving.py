import time, os

def filename_with_datetime(filename, folder='./', extension=''):
    if extension.startswith('.'):
        extension=extension[1:]
    if filename!='':
        filename=filename+'_'
    return os.path.join(folder, filename, time.strftime("%Y_%m_%d-%H:%M:%S")+'.'+extension)


if __name__=='__main__':

    print(filename_with_datetime('', folder='./', extension='.npy'))
    print(filename_with_datetime('', folder='./', extension='npy'))    
