import os
import numpy as np
from analyz.IO.npz import load_dict

class slurm_script:

    def __init__(self, jobname,
                 partition='bigmem',
                 time='02:00:00',
                 mem='5G',
                 load = ['python/3.6', 'gcc'],
                 ntasks=1, nodes=8):

        self.jobname = jobname
        self.script = self.create_new_slurm_script(jobname, partition,
                                                   time, mem, ntasks, nodes, load)
        
    def create_new_slurm_script(self, jobname, partition,
                                time, mem, ntasks, nodes, load):
        S = '#!/bin/bash'
        
        S = """#!/bin/bash
        #SBATCH --partition={partition}
        #SBATCH --time={time}
        #SBATCH --mem={mem}
        #SBATCH --nodes={nodes}
        #SBATCH --chdir=.
        #SBATCH --output={name}_%j.out 
        #SBATCH --error={name}_%j.err
        #SBATCH --job-name={name}
        #SBATCH --mail-user=yann.zerlaut@icm-institute.org   # your mail
        #SBATCH --mail-type=ALL # type of notifications you want to receive
        # # # SBATCH --ntasks={ntasks}
        # # # SBATCH --cpus-per-task=1
        # # # SBATCH --gres=gpu:1
        """.format(**{'name':jobname,
                      'partition':partition,
                      'time':time,
                      'mem':mem,
                      'ntasks':ntasks,
                      'nodes':nodes}).replace('        ', '')
        for key in load:
            S += 'module load %s\n' % key

        return S

    def write(self, filename=None, folder='.'):
        if filename is None:
           filename = self.jobname+'.sh'

        with open(os.path.join(folder, filename), 'w') as f:
            f.write(self.script)

    def append_instruction(self, instruction):
        if instruction.endswith('\n'):
            self.script += instruction
        else:
            self.script += instruction+'\n'

            
class bash_script:

    def __init__(self, jobname):
        self.jobname = jobname
        self.script = '#!/bin/bash\n'

    def write(self, filename=None, folder='.'):
        if filename is None:
           filename = self.jobname+'.sh'

        with open(os.path.join(folder, filename), 'w') as f:
            f.write(self.script)

    def append_instruction(self, instruction):
        if instruction.endswith('\n'):
            self.script += instruction
        else:
            self.script += instruction+'\n'


            
class GridSimulation:
    """
    type of dict elements should be numpy arrays !!
    """
    
    def __init__(self, GRID):

        if type(GRID) is str:
            GRID = load_dict(GRID)

        for key in GRID: # forcing to have a set of numpy arrays
            if type(GRID[key])!=np.ndarray:
                GRID[key] = np.array([GRID[key]])

        self.N = np.product([len(GRID[key]) for key in GRID.keys()])
        self.Ns = [len(GRID[key]) for key in list(GRID.keys())]
        self.nkeys = len(list(GRID.keys()))
        self.GRID = GRID
        self.dtypes = [GRID[key].dtype for key in list(GRID.keys())]

        # useful to split indices
        self.cumNprod = [int(np.product([len(self.GRID[key]) for key in list(self.GRID.keys())[i:]])) for i in range(len(self.GRID.keys())+1)]
        
    def update_dict_from_GRID_and_index(self, i, dict_to_fill):

        Is = self.compute_indices(i)

        for k, key in enumerate(self.GRID.keys()):
            dict_to_fill[key] = self.GRID[key][Is[k]]


    def params_filename(self, i,
                        formatting=None):
        """
        i can be either the fill index 
        print(sim.params_filename(i, formatting=['%.1f', '%.2f', '%.2f', '%i', '%.0f']))
        """
        
        if isinstance(i, (int, np.int32, np.int64)):
            Is = self.compute_indices(i)
        elif isinstance(i, (list, np.array)) and (len(i)==self.nkeys):
            Is = np.array(i) # should be the array of indices
        else:
            print('argument not recognized')
            Is = []

        if formatting is None:
            formatting = []
            for dtype in self.dtypes:
                if dtype==int:
                    formatting.append('%i')
                elif dtype==float:
                    formatting.append('%f')
                else:
                    formatting.append('%s')

        filename = ''
        for k, key in enumerate(self.GRID.keys()):
            filename += ('%s_'+formatting[k]+'--') % (key, self.GRID[key][Is[k]])

        return filename[:-2]

    def build_script(self, base_instruction,
                     base_script=None,
                     simultaneous_runs=0):
        """
        """

        if base_script is None:
            script=''
        else:
            script=base_script
            
        for i in range(self.N):
            if (simultaneous_runs>0) and (i%simultaneous_runs<(simultaneous_runs-1)):
                script+='%s %i &\n' % (base_instruction, i)
            else:
                script+='%s %i\n' % (base_instruction, i)

        if script.endswith(' &\n'):
            return script[:-3]
        else:
            return script
    

    def compute_indices(self, i):

        Is = np.zeros(self.nkeys, dtype=int) 

        Is[self.nkeys-1] = i % self.Ns[self.nkeys-1]
        for ii in range(self.nkeys-1):
            Is[ii] = int((i-np.sum(np.dot(Is, self.cumNprod[1:])))/np.product(self.Ns[ii+1:]))

        return Is


if __name__=='__main__':

    #ss = slurm_script('calib-passive', partition='bigmem', mem='10G', nodes=1)
    
    GRID = {'x':np.arange(3),
            'y':np.linspace(0, 1, 5),
            'z':np.array(['kjsdhf', 'ksjdhf'])}

    sim = GridSimulation(GRID)
    print(sim.params_filename(3))
    print(sim.params_filename([2, 4, 1]))
    bs = bash_script('test')
    
    # print(sim.params_filename(2))
    bs.script = sim.build_script('python -c "print(3)"',
                                 base_script=bs.script,
                                 simultaneous_runs=100)
    # print(bs.script)
