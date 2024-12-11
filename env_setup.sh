conda create -y -n masterarbeit
conda activate masterarbeit
conda config --append channels conda-forge
conda install -y --file .\requirements.txt
pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl
