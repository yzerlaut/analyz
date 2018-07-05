import os

def get_files_ordered_by_creation(dirpath,
                                  extension='',
                                  return_full_path=True):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s)) and s.endswith(extension)]
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

if __name__ == '__main__':
    print(get_directories_ordered_by_creation(os.path.join(os.getenv("HOME"), 'DATA')))
    last_dir = get_directories_ordered_by_creation(os.path.join(os.getenv("HOME"), 'DATA'))[-1]
    print(get_files_ordered_by_creation(last_dir))
    
