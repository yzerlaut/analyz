import csv
import numpy as np

def transform_csv_into_array_of_cells(csvfile):

    CSV_ARRAY = []
    CELLS = [] # dictionary of each cell

    with open('data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            CSV_ARRAY.append(row)
    # extra keys that will be added
    extra_keys = np.array(CSV_ARRAY)[0,4:]

    CSV_ARRAY = np.array(CSV_ARRAY)[1:,:]
    ii=1
    i_diff_folders = np.argwhere(CSV_ARRAY[:,0]!='').flatten()
    i_diff_folders = np.concatenate([i_diff_folders, [len(CSV_ARRAY)]])
    for i1, i2 in zip(i_diff_folders[:-1], i_diff_folders[1:]):
        CSV_ARRAY2 = CSV_ARRAY[i1:i2,:]
        i_diff_days = np.argwhere(CSV_ARRAY2[:,1]!='').flatten()
        i_diff_days = np.concatenate([i_diff_days, [len(CSV_ARRAY2)]])
        for i3, i4 in zip(i_diff_days[:-1], i_diff_days[1:]):
            CSV_ARRAY3 = CSV_ARRAY2[i3:i4,:]
            i_diff_cells = np.argwhere(CSV_ARRAY3[:,2]!='').flatten()
            i_diff_cells = np.concatenate([i_diff_cells, [len(CSV_ARRAY3)]])
            for i5, i6 in zip(i_diff_cells[:-1], i_diff_cells[1:]):
                CSV_ARRAY4 = CSV_ARRAY3[i5:i6,:]
                print(CSV_ARRAY4[1:,4:])
                bd = {'day':CSV_ARRAY3[0,1], 'files':CSV_ARRAY4[1:,3], 'folder':CSV_ARRAY2[0,0], 'n':ii}
                for i in range(len(extra_keys)):
                    bd[extra_keys[i]] = CSV_ARRAY4[1:,4+i]
                CELLS.append(bd)
                ii+=1
    return CELLS

def List_cells_and_choose_one(csvfile, force_n=0):
    CELLS = transform_csv_into_array_of_cells(csvfile)
    print(
        """
        ==================================================
        Dataset for this protocol
        ==================================================
        """)    
    for i in range(len(CELLS)):
        print(""" Cell {n} ==> Day {day}, Files {files} """.format(**CELLS[i]))
    print(
        """
        ==================================================
        ==================================================
        """)
    if not force_n:
        n=int(input('Choose the cell you want to visualize: '))
    else:
        n=force_n # to shunt this step for debugging
    if n<=0 or n>len(CELLS):
        print('/!\ Not a valid cell number !!')
    else:
        return CELLS[n-1], n
