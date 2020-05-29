import datetime, os

def filename_with_datetime(filename,
                           folder='./',
                           with_microseconds=False,
                           extension=''):
    if extension.startswith('.'):
        extension=extension[1:]
    if filename!='':
        filename=filename+'_'
    if with_microseconds:
        d = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S-%f")
    else:
        d = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    return os.path.join(folder, filename+d+'.'+extension)


if __name__=='__main__':

    print(filename_with_datetime('', folder='./', extension='.npy'))
    print(filename_with_datetime('', folder='./', extension='npy'))
