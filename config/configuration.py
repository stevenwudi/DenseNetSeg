import os
import sys
from datetime import datetime
from utils.logger import Logger
import importlib.machinery
import time

class Logger_std(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



class Configuration:
    def __init__(self):
        loader = importlib.machinery.SourceFileLoader('config', './config/config_densenet_seg.py')
        config = loader.load_module()

        # Experiment dir
        model_save_suffix = "crop_size_" + str(config.crop_size)
        model_save_suffix += "_out_channels_num_" + str(config.out_channels_num)
        model_save_suffix += "_ppl_out_channels_num_" + str(config.ppl_out_channels_num)
        config.model_save_suffix = model_save_suffix + '.pth.tar'
        if not os.path.exists('./logs'):
            os.mkdir('./logs')

        config.exp_dir = os.path.join('./logs', model_save_suffix + '_' * 5 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(config.exp_dir):
            os.mkdir(config.exp_dir)

        config.logger_tf = Logger('./logs')

        # If we load from a pretrained model
        # Enable log file
        sys.stdout = Logger_std(os.path.join(config.exp_dir, "logfile.log"))
        #sys.stdout = open(os.path.join(config.exp_dir, "logfile.log"), 'w')
        print(help(config), flush=True)

        self.config = config

