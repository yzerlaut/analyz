import os

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

    def __init__(self, GRID):

        self.N = np.product([len(GRID[key]) for key in GRID.keys()])
        self.Ns = [len(GRID[key]) for key in list(GRID.keys())]
        self.nkeys = len(list(GRID.keys()))
        self.GRID = GRID

        # useful to split indices
        self.cumNprod = [int(np.product([len(self.GRID[key]) for key in list(self.GRID.keys())[i:]])) for i in range(len(self.GRID.keys())+1)]
        
    def update_dict_from_GRID_and_index(self, i, dict_to_fill):

        if i>=self.N:
            print('/!\ index to high given the GRID /!\ ')

        Is = self.compute_indices(i)

        for k, key in enumerate(GRID.keys()):
            dict_to_fill[key] = self.GRID[key][Is[k]]


    def params_filename(self, i,
                        formatting=None):
        """
        print(sim.params_filename(i, formatting=['%.1f', '%.2f', '%.2f', '%i', '%.0f']))
        """
        if i>=self.N:
            print('/!\ index to high given the GRID /!\ ')

        Is = self.compute_indices(i)

        if formatting is None:
            formatting = ['%.3f' for n in range(self.nkeys)]
            
        filename = ''
        for k, key in enumerate(GRID.keys()):
            filename += ('%s_'+formatting[k]+'--') % (key, self.GRID[key][Is[k]])

        return filename[:-2]
            

    def compute_indices(self, i):

        Is = np.zeros(self.nkeys, dtype=int) #

        Is[self.nkeys-1] = i % Ns[self.nkeys-1]
        for ii in range(self.nkeys-1):
            Is[ii] = int((i-np.sum(np.dot(Is, self.cumNprod[1:])))/np.product(self.Ns[ii+1:]))

        return Is
