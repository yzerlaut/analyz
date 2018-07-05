import os

def get_files_ordered_by_creation(dirpath, return_full_path=True):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    if not return_full_path:
        return a
    else:
        return [os.path.join(dirpath, s) for s in a]

def get_directories_ordered_by_creation(dirpath, return_full_path=True):
    a = [s for s in os.listdir(dirpath)
         if not os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    if not return_full_path:
        return a
    else:
        return [os.path.join(dirpath, s) for s in a]
