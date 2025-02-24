### exit on error
set -e
### set conda hook
eval "$(conda.shell.bash hook)"

### create enviroment
conda create -n RAM_jit -y python=3.9
conda activate RAM_jit

### install package
pip install -r requirements.txt
python setup.py develop
