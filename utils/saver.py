'''
Copyright (c) R. Mineo, 2022-2024. All rights reserved.
This code was developed by R. Mineo in collaboration with PerceiveLab and other contributors.
For usage and licensing requests, please contact the owner.
'''

from cProfile import label
from datetime import datetime
import os
from os.path import getmtime
import sys
import ast
import torch
from pathlib import Path
from time import time
from typing import Union
from torch.utils.tensorboard import SummaryWriter
import torchvision
import socket
import threading
import webbrowser
from matplotlib import pyplot as plt
from matplotlib import figure as fgr
import numpy as np

class Saver(object):
    """
    Saver allows for saving and restore networks.
    """
    def __init__(self, base_output_dir: Path, args: dict, sub_dirs=('trainingSet', 'validationSet'), tag=''):

        # Create experiment directory
        timestamp_str = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')
        if isinstance(tag, str) and len(tag) > 0:
            # Append tag
            timestamp_str += f"_{tag}"
        self.path = base_output_dir / f'{timestamp_str}'
        self.path.mkdir(parents=True, exist_ok=True)

        # TB logs
        self.args = args
        self.writer = SummaryWriter(str(self.path))

        # Create checkpoint sub-directory
        self.ckpt_path = self.path / 'ckpt'
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        # Create output sub-directories
        self.sub_dirs = sub_dirs
        self.output_path = {}

        for s in self.sub_dirs:
            self.output_path[s] = self.path / 'output' / s

        for d in self.output_path.values():
            d.mkdir(parents=True, exist_ok=False)

        # Dump experiment hyper-params
        with open(self.path / 'hyperparams.txt', mode='wt') as f:
            args_str = [f'{a}: {v}\n' for a, v in self.args.items()]
            args_str.append(f'exp_name: {timestamp_str}\n')
            f.writelines(sorted(args_str))
        
        if self.args['start_tensorboard_server']:
            self.start_tensorboard_daemon(self.path, args['tensorboard_port'])
        
        # Dump command
        with open(self.path / 'command.txt', mode='wt') as f:
            cmd_args = ' '.join(sys.argv)
            f.write(cmd_args)
            f.write('\n')

    def close(self):
        self.writer.close()
    
    def save_model(self, net: torch.nn.Module, name: str, epoch: int):
        """
        Save model parameters in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Copy to CPU
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        # Save
        torch.save(state_dict, str(self.ckpt_path) + '/' + f'{name}_{epoch:05d}.pth')

    def save_model_stats(self, net: torch.nn.Module, stats: dict, name: str, epoch: int):
        """
        Save model parameters and stats in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Copy to CPU
        for k,v in state_dict.items():
            state_dict[k] = v.cpu()
        # Save
        torch.save({'state_dict':state_dict,'stats':stats}, self.ckpt_path / f'{name}_{epoch:05d}.pth')

    def save_figure(self, figure_data_array: dict, step, split, filename: str):
        ''' Save figure data array in experiment folder '''
        out_path = self.output_path[split] / f'{filename}_{step:05d}.pt'
        torch.save(figure_data_array, out_path)

    def save_image(self, tensor_img: torch.Tensor, step, split, filename: str):
        ''' Save image from tensor to file '''
        out_path = self.output_path[split] / f'{filename}_{step:05d}.png'
        torchvision.utils.save_image(tensor_img.cpu(), out_path, normalize=True)

    def log_scalar(self, name: str, value: float, iter_n: int):
        '''
        Log scalar to Tensorboard
        '''
        self.writer.add_scalar(name, value, iter_n)

    def log_figure(self, name: str, figure_obj: fgr.Figure, iter_n: int):
        '''
        Log figure to Tensorboard
        '''
        self.writer.add_figure(name, figure_obj, iter_n)
        
        plt.close(figure_obj)

    def log_images(self, name: str, images_vector: torch.Tensor, iter_n: int):
        '''
        Log images to Tensorboard
        image_vector.shape = (CH,M,N)
        '''
        img_grid = torchvision.utils.make_grid(images_vector.unsqueeze(0), normalize=True, nrow=10)
        self.writer.add_image(name, img_grid, iter_n)

    @staticmethod
    def load_hyperparams(hyperparams_path):
        """
        Load hyperparams from file. Tries to convert them to best type.
        """
        # Process input
        hyperparams_path = Path(hyperparams_path)
        if not hyperparams_path.exists():
            raise OSError('Please provide a valid checkpoints path')
        if hyperparams_path.is_dir():
            hyperparams_path = os.path.join(hyperparams_path, 'hyperparams.txt')
        else:
            hyperparams_path = os.path.join(hyperparams_path.parent.parent, 'hyperparams.txt')
        # Prepare output
        output = {}
        # Read file
        with open(hyperparams_path) as file:
            # Read lines
            for l in file:
                # Remove new line
                l = l.strip()
                # Separate name from value
                toks = l.split(':')
                name = toks[0]
                value = ':'.join(toks[1:]).strip()
                # Parse value
                try:
                    value = ast.literal_eval(value)
                except:
                    pass
                # Add to output
                output[name] = value
        # Return
        return output

    @staticmethod
    def load_model(model_path: Union[str, Path], verbose: bool = True, return_epoch: bool = False):
        """
        Load state dict from pre-trained checkpoint. In case a directory is
          given as `model_path`, the last checkpoint is loaded.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise OSError('Please provide a valid path for restore checkpoint.')

        if model_path.is_dir():
            # Check there are files in that directory
            file_list = sorted(model_path.glob('*.pth'), key=getmtime)
            if len(file_list) == 0:
                # Check there are files in the 'ckpt' subdirectory
                model_path = model_path / 'ckpt'
                file_list = sorted(model_path.glob('*.pth'), key=getmtime)
                if len(file_list) == 0:
                    raise OSError("Couldn't find pth file.")
            checkpoint = file_list[-1]
            if verbose:
                print(f'Last checkpoint found: {checkpoint} .')
        elif model_path.is_file():
            checkpoint = model_path

        if verbose:
            print(f'Loading pre-trained weight from {checkpoint}...')

        if return_epoch:
            return torch.load(checkpoint), int(str(checkpoint).split("_")[-1][:-4])
        else:
            return torch.load(checkpoint)

    @staticmethod
    def load_checkpoint(model_path: Union[str, Path], verbose: bool = True, return_epoch: bool = False):
        """
        Load state dict e stats from pre-trained checkpoint. In case a directory is
          given as `model_path`, the best (minor loss) checkpoint is loaded.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise OSError('Please provide a valid path for restore checkpoint.')

        if model_path.is_dir():
            # Check there are files in that directory
            file_list = sorted(model_path.glob('*.pth'), key=getmtime)
            if len(file_list) == 0:
                # Check there are files in the 'ckpt' subdirectory
                model_path = model_path / 'ckpt'
                file_list = sorted(model_path.glob('*.pth'), key=getmtime)
                if len(file_list) == 0:
                    raise OSError("Couldn't find pth file.")
            # Chose best checkpoint based on minor loss
            if verbose:
                print(f'Search best checkpoint (minor loss)...')
            loss = torch.load(file_list[0])['stats']['mse_loss']
            checkpoint = file_list[0]
            for i in range(1,len(file_list)):
                loss_tmp = torch.load(file_list[i])['stats']['mse_loss']
                if loss_tmp < loss:
                    loss = loss_tmp
                    checkpoint = file_list[i]
            if verbose:
                print(f'Best checkpoint found: {checkpoint} (loss: {loss}).')
        elif model_path.is_file():
            checkpoint = model_path

        if verbose:
            print(f'Loading pre-trained weight from {checkpoint}...')

        if return_epoch:
            return torch.load(checkpoint)['state_dict'], int(str(checkpoint).split("_")[-1][:-4])
        else:
            return torch.load(checkpoint)['state_dict']

    @staticmethod
    def saveLogsError(filename, true_labels_in, predicted_labels_in, predicted_scores_in, dict_other_info_in, split, epoch, txt_file=False):
        true_labels, predicted_labels, predicted_scores = np.array(true_labels_in), np.array(predicted_labels_in), np.array(predicted_scores_in)
        dict_other_info = {key:np.array(dict_other_info_in[key]) for key in dict_other_info_in}

        mask_error = true_labels!=predicted_labels
        
        Saver.saveLogs(filename, true_labels[mask_error], predicted_labels[mask_error], predicted_scores[mask_error], {k: dict_other_info[k][mask_error] for k in dict_other_info}, split, epoch, txt_file)
    
    @staticmethod
    def saveLogs(filename, true_labels_in, predicted_labels_in, predicted_scores_in, dict_other_info_in, split, epoch, txt_file=False):
        true_labels, predicted_labels, predicted_scores = np.array(true_labels_in), np.array(predicted_labels_in), np.array(predicted_scores_in)
        dict_other_info = {key: np.array(dict_other_info_in[key]) for key in dict_other_info_in}

        if txt_file:
            with open(str(filename)+'.txt', "w") as file1:
                file1.write("true_label;predicted_label;predicted_scores")
                for x in dict_other_info:
                    file1.write(";")
                    file1.write(str(x))
                file1.write("\n")
            
                for i in range(len(true_labels)):
                    file1.write(str(true_labels[i]))
                    file1.write(";")
                    file1.write(str(predicted_labels[i]))
                    file1.write(";")
                    file1.write(str(predicted_scores[i]))
                    for x in dict_other_info:
                        file1.write(";")
                        file1.write(str(dict_other_info[x][i]))
                    file1.write("\n")
        else:
            dict_save= {
                'epoch': epoch,
                'split': split,
                'true_labels': true_labels,
                'predicted_labels': predicted_labels,
            }
            
            for k in dict_other_info:
                dict_save[k] = dict_other_info[k]
            
            torch.save(dict_save, str(filename)+'.pt')
        
    @staticmethod
    def saveMetrics(filename, dict_general_info_in, split, epoch, txt_file=False):
        dict_general_info = {key: np.array(dict_general_info_in[key]) for key in dict_general_info_in}

        if txt_file:
            with open(str(filename)+'.txt', "w") as file1:
                file1.write("epoch;split")
                for x in dict_general_info:
                    file1.write(";")
                    file1.write(str(x))
                file1.write("\n")
            
                file1.write(str(epoch))
                file1.write(";")
                file1.write(split)
                for x in dict_general_info:
                    file1.write(";")
                    file1.write(str(dict_general_info[x]))
        else:
            dict_save= {
                'epoch': epoch,
                'split': split,
            }
            
            for k in dict_general_info:
                dict_save[k] = dict_general_info[k]
            
            torch.save(dict_save, str(filename)+'.pt')
    
    @staticmethod
    def printLogsError(true_labels_in, predicted_labels_in, predicted_scores_in, dict_other_info_in, split, epoch):
        true_labels, predicted_labels, predicted_scores = np.array(true_labels_in), np.array(predicted_labels_in), np.array(predicted_scores_in)
        dict_other_info = {key:np.array(dict_other_info_in[key]) for key in dict_other_info_in}

        mask_error = true_labels!=predicted_labels
        
        print(split, epoch, "epoch - Error logs")
        print("true_label", "predicted_label", "predicted_score", *(str(x) for x in dict_other_info), sep=';')
        for i in range(len(true_labels[mask_error])):
            print(str(true_labels[mask_error][i]), str(predicted_labels[mask_error][i]), str(predicted_scores[mask_error][i]), *(str(dict_other_info[x][mask_error][i]) for x in dict_other_info), sep=';')

    @staticmethod
    def start_tensorboard_daemon(path, tensorboard_port):
        # Start TensorBoard Daemon to visualize data
        i = 0
        while(True):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                i += 1
                if s.connect_ex(('localhost', tensorboard_port)) == 0: # check if port is busy
                    tensorboard_port = tensorboard_port + 1
                else:
                    break
                if i > 100:
                    raise RuntimeError('Can not find free port at +100 from your chosen port!')
        t = threading.Thread(target=lambda: os.system('tensorboard --logdir=' + str(path) + ' --port=' + str(tensorboard_port)))
        t.start()
        webbrowser.open('http://localhost:' + str(tensorboard_port) + '/', new=1)
