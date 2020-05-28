class slurm_script:

    def __init__(self, jobname,
                 partition='gpu',
                 time='02:00:00',
                 mem='20G',
                 ntasks=1):

        self.jobname = jobname
        self.script = self.create_new_slurm_script(jobname, partition, time, mem, ntasks)
        
    def create_new_slurm_script(self, jobname, partition, time, mem, ntasks):
        S = """#!/bin/bash
        #SBATCH --partition={partition}
        #SBATCH --time={time}
        #SBATCH --mem={mem}
        #SBATCH --ntasks={ntasks}
        #SBATCH --cpus-per-task=1
        #SBATCH --gres=gpu:1
        #SBATCH --nodes=1
        #SBATCH --workdir=.
        #SBATCH --output={name}_%j.out 
        #SBATCH --error={name}_%j.err
        #SBATCH --job-name={name}
        module load gcc
        module load python/3.6
        """.format(**{'name':jobname,
                      'partition':partition,
                      'time':time,
                      'mem':mem,
                      'ntasks':ntasks}).replace('        ', '')
        return S

    def write(self, filename=None):
        if filename is None:
           filename = self.jobname+'.sh'

           with open(filename, 'w') as f:
               f.write(self.script)

    def append_instruction(self, instruction):
        if instruction.endswith('\n'):
            self.script += instruction
        else:
            self.script += instruction+'\n'
