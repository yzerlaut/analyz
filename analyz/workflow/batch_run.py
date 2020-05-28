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

    def __init__(self):
        
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
            
