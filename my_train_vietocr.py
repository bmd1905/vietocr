import matplotlib.pyplot as plt
from PIL import Image
import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')

dataset_params = {
    'name':'hw',
    'data_root':'./data_line/',
    'train_annotation':'train_line_annotation.txt',
    'valid_annotation':'test_line_annotation.txt'
}

params = {
         'print_every':1,
         'valid_every':15*200,
          'iters':10_000,
          'pretrained': '../pretrained/transformer_10000iters_kaggle_3.pth',
          'checkpoint': '../pretrained/transformer_10000iters_kaggle_3.pth',    
          'export':'./weights/22_1_2023.pth',
          'metrics': 10000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)

device = torch.device('mps')
config['device'] = device

# config['transformer']['num_encoder_layers'] = 3
# config['transformer']['num_decoder_layers'] = 3

config['dataloader']['num_workers'] = 0
config['trainer']['batch_size'] = 64



trainer = Trainer(config, pretrained=False)


trainer.config.save('config.yml')

trainer.train()

