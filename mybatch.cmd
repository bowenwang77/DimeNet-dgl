#!/bin/bash
#SBATCH --job-name=TEST
#SBATCH --mail-user=bwwang@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept8/gds/bwwang/phd2/DimeNet-dgl/ouput.txt 
#SBATCH --gres=gpu:1


@REM #!/bin/bash
@REM #SBATCH --job-name=TEST
@REM #SBATCH --mail-user=bwwang@cse.cuhk.edu.hk
@REM #SBATCH --mail-type=ALL
@REM #SBATCH --output=/research/dept8/gds/bwwang/phd2/ocp/ouput4.txt 
@REM #SBATCH --qos pheng_gpu
@REM #SBATCH --account pheng_gpu
@REM #SBATCH --gres=gpu:1


## Below is the commands to run , for this example,
## Create a sample helloworld.py and Run the sample python file
## Result are stored at your defined --output location

python main.py --model-cnf config/dimenet.yaml
