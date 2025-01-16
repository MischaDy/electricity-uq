import logging
import os.path

from helpers.io_helper import IO_Helper

logging.basicConfig(level=logging.INFO)

model_filenames = [
    'base_model_nn_n35136_it100_nh2_hs50.pth',
]
state_dict_infix = 'dict'

io_helper = IO_Helper()
for model_filename in model_filenames:
    model = io_helper.load_model(filename=model_filename)
    root, ext = os.path.splitext(model_filename)
    statedict_filename = f'{root}_{state_dict_infix}{ext}'
    logging.info(f'saving model {model_filename} as {statedict_filename}')
    io_helper.save_torch_model_statedict(model, filename=statedict_filename)
