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


def choose_a_file_based_on_keyboard_input(dirpath, extension='', Nmax=5):

    list_of_files = get_files_ordered_by_creation(dirpath,
                                                  extension=extension)[::-1]
    print('==> list of files:')
    for i in range(min([Nmax, len(list_of_files)])):
        print('   ', str(i+1)+') ', list_of_files[i])
    print('--------------------------------')
    number= input('choose a number within this list (default = 1) \n')
    try:
        number = int(number)-1
    except ValueError:
        number = 0
    print('the chosen file is:', list_of_files[number])
    return list_of_files[number]

if __name__ == '__main__':
    print(get_directories_ordered_by_creation(os.path.join(os.getenv("HOME"), 'DATA')))
    last_dir = get_directories_ordered_by_creation(os.path.join(os.getenv("HOME"), 'DATA'))[-1]
    print(get_files_ordered_by_creation(last_dir, ''))
    
