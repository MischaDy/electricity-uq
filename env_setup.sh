conda create -y -n masterarbeit
conda activate masterarbeit
conda config --append channels conda-forge
conda install -y --file .\requirements.txt
pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl
xpip install git+https://github.com/cornellius-gp/gpytorch.git@fe4619b546f0bc41f801b9b5ac57c424c25bd3fe
xpip install git+https://github.com/aladagemre/django-notification.git@2927346f4c513a217ac8ad076e494dd1adbf70e1