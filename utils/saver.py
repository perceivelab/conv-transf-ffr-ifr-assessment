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

    def save_checkpoint(self, net: torch.nn.Module, stats: dict, name: str, epoch: int):
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

    def dump_line(self, line, step, split, name, fmt=''):
        """
        Dump line as matplotlib figure into folder and tb

        """
        assert split in self.sub_dirs
        # Plot line
        fig = plt.figure()
        if isinstance(line, tuple):
            line_x, line_y = line
            plt.plot(line_x.cpu().detach().numpy(), line_y.cpu().detach().numpy(), fmt)
        else:
            plt.plot(line.cpu().detach().numpy(), fmt)
        out_path = self.output_path[split] / f'line_{step:08d}_{name}.jpg'
        plt.savefig(out_path)
        self.writer.add_figure(f'{split}/{name}', fig, step)

    def dump_histogram(self, tensor: torch.Tensor, epoch: int, desc: str):
        try:
            self.writer.add_histogram(desc, tensor.contiguous().view(-1), epoch)
        except:
            print('Error writing histogram')
    
    def dump_metric(self, value: float, epoch: int, *tags):
        self.writer.add_scalar('/'.join(tags), value, epoch)

    def save_data(self,data,name:str):
        ''' Save generic data in experiment folder '''
        torch.save(data, self.path / f'{name}.pth')

    def log_scalar(self, name: str, value: float, iter_n: int):
        '''
        Log loss to TB and comet_ml
        '''
        self.writer.add_scalar(name, value, iter_n)

    def log_images(self, name: str, images_vector: torch.Tensor, iter_n: int, split, filename, save_file):
        '''
        Log images to Tensorboard
        image_vector.shape = (CH,M,N)
        '''
        img_grid = torchvision.utils.make_grid(images_vector.unsqueeze(0), normalize=True, nrow=10)
        self.writer.add_image(name, img_grid, iter_n)
        #pil_img = transforms.ToPILImage()(img_grid.cpu())
        if save_file:
            self.save_image(img_grid, filename, split, iter_n)

    def save_image(self, tensor_img: torch.Tensor, filename: str, split, step):
        ''' Save image from tensor to file '''
        out_path = self.output_path[split] / f'{filename}_{step:05d}.png'
        torchvision.utils.save_image(tensor_img.cpu(), out_path, normalize=True)

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

    def saveLogsError(self, true_labels_in, predicted_labels_in, predicted_scores_in, dict_other_info_in, split, epoch):
        true_labels, predicted_labels, predicted_scores = np.array(true_labels_in), np.array(predicted_labels_in), np.array(predicted_scores_in)
        dict_other_info = {key:np.array(dict_other_info_in[key]) for key in dict_other_info_in}

        mask_error = true_labels!=predicted_labels

        if (len(predicted_scores[mask_error]) > 0):
            if (len(predicted_scores[mask_error][0]) != 2):
                raise RuntimeError("Correct len of predicted_score's list in the file1.write (a few lines below in this code).")

        with open(self.output_path[split] / f'errors_{epoch:05d}.txt', "w") as file1:
            file1.write("true_label;predicted_label;predicted_score")
            for x in dict_other_info:
                file1.write(";")
                file1.write(str(x))
            file1.write("\n")

            for i in range(len(true_labels[mask_error])):
                file1.write(str(true_labels[mask_error][i]))
                file1.write(";")
                file1.write(str(predicted_labels[mask_error][i]))
                file1.write(";")
                file1.write("[" + str(predicted_scores[mask_error][i][0]) + "," + str(predicted_scores[mask_error][i][1]) + "]")
                for x in dict_other_info:
                    file1.write(";")
                    file1.write(str(dict_other_info[x][mask_error][i]))
                file1.write("\n")
    
    @staticmethod
    def saveLogs(logdir, true_labels_in, predicted_labels_in, predicted_scores_in, dict_other_info_in, split, epoch):
        true_labels, predicted_labels, predicted_scores = np.array(true_labels_in), np.array(predicted_labels_in), np.array(predicted_scores_in)
        dict_other_info = {key:np.array(dict_other_info_in[key]) for key in dict_other_info_in}

        if (len(predicted_scores[0]) != 2):
            raise RuntimeError("Correct len of predicted_score's list in the file1.write (a few lines below in this code).")

        with open(logdir + f'/{split}_logs_{epoch:05d}.txt', "w") as file1:
            file1.write("true_label;predicted_label;predicted_score")
            for x in dict_other_info:
                file1.write(";")
                file1.write(str(x))
            file1.write("\n")

            for i in range(len(true_labels)):
                file1.write(str(true_labels[i]))
                file1.write(";")
                file1.write(str(predicted_labels[i]))
                file1.write(";")
                file1.write("[" + str(predicted_scores[i][0]) + "," + str(predicted_scores[i][1]) + "]")
                for x in dict_other_info:
                    file1.write(";")
                    file1.write(str(dict_other_info[x][i]))
                file1.write("\n")
    
    @staticmethod
    def printLogsError(true_labels_in, predicted_labels_in, predicted_scores_in, dict_other_info_in, split, epoch):
        true_labels, predicted_labels, predicted_scores = np.array(true_labels_in), np.array(predicted_labels_in), np.array(predicted_scores_in)
        dict_other_info = {key:np.array(dict_other_info_in[key]) for key in dict_other_info_in}

        mask_error = true_labels!=predicted_labels
        
        print(split, epoch, "epoch - Error logs")
        print("true_label", "predicted_label", "predicted_score", *(str(x) for x in dict_other_info), sep=';')
        for i in range(len(true_labels[mask_error])):
            print(str(true_labels[mask_error][i]), str(predicted_labels[mask_error][i]), str(predicted_scores[mask_error][i]), *(str(dict_other_info[x][mask_error][i]) for x in dict_other_info), sep=';')

    def close(self):
        self.writer.close()
