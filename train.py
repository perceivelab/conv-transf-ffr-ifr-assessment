from multiprocessing.sharedctypes import Value
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import argparse
from pathlib import Path
from tqdm import tqdm
from utils.dataset import get_loader
from utils.models.models import get_model
from utils.trainer import Trainer
from utils.saver import Saver
import glob
from utils import utils
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
#torch.backends.cudnn.benchmark = True # good if input sizes for net not change
import GPUtil
import platform

#torch.autograd.set_detect_anomaly(False) non è detto che se c'è quale nan bisogna bloccare subito l'allenamento

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, help='dataset folder path')
    parser.add_argument('--split_path', type=Path, help='json dataset metadata for MONAI')
    parser.add_argument('--num_fold', type=int, help='test fold for nested cross-validation', default=0)
    parser.add_argument('--inner_loop', type=int, help='validation fold for nested cross-validation', default=0)
    parser.add_argument('--cache_rate', type=float, help='fraction of dataset to be cached in RAM', default=1.0)

    parser.add_argument('--dataset3d', type=int, help='use 3D dataset', choices=[0,1], default=1)
    parser.add_argument('--dataset2d', type=int, help='use 2D dataset', choices=[0,1], default=0)
    parser.add_argument('--resize', type=int, help='(T,H,W) resize dimensions', nargs=3, default=[-1,256,256])
    parser.add_argument('--pad', type=int, help='(T,H,W) padding; use -1 to not modify that dimension', nargs=3, default=[60,-1,-1])
    parser.add_argument('--datasetGrid', type=int, help='trasform 3D dataset to 2D image with all frames in a grid', choices=[0,1], default=0)
    parser.add_argument('--datasetGridPatches', type=int, help='if using datasetGrid, number of patches in the grid (preferred squared number)', default=16)
    parser.add_argument('--datasetGridStride', type=int, help='if using datasetGrid, temporal stride', default=4)
    parser.add_argument('--mean', type=float, help='(ch1,ch2,ch3) normalization mean', nargs=3, default=[0.0,0.0,0.0])
    parser.add_argument('--std', type=float, help='(ch1,ch2,ch3) normalization standard deviation', nargs=3, default=[1.0,1.0,1.0])
    parser.add_argument('--ffr_threshold', type=float, help='if not None, threshold for FFR-based labeling [0.0-1.0]', default=None)
    parser.add_argument('--ifr_threshold', type=float, help='if not None, if not None, threshold for FFR-based labeling [0.0-1.0]', default=None)
    parser.add_argument('--inputChannel', type=int, help='number of input channels (1: grayscale, 3: RGB)', default=1)
    parser.add_argument('--doubleViewInput', type=int, help='return two views from dataset', choices=[0,1], default=0)

    parser.add_argument('--model', type=str, help='model (ResNet3D, ResNet3D_pretrained)', default='ResNet3D_pretrained')

    parser.add_argument('--enable_multibranch_ffr', type=int, help='enable FFR regression output branch', choices=[0,1], default=1)
    parser.add_argument('--enable_multibranch_ifr', type=int, help='enable iFR regression output branch', choices=[0,1], default=1)
    parser.add_argument('--multibranch_dropout_label', type=float, help='dropout for classification branch [0.0-1.0]', default=0.0)
    parser.add_argument('--multibranch_dropout_FFR', type=float, help='dropout for FFR regression branch [0.0-1.0]', default=0.0)
    parser.add_argument('--multibranch_dropout_iFR', type=float, help='dropout of iFR regression branch [0.0-1.0]', default=0.0)
    parser.add_argument('--multibranch_loss_weight_label', type=float, help='classification loss weight', default=1.0)
    parser.add_argument('--multibranch_loss_weight_FFR', type=float, help='FFR regression loss weight', default=10.0)
    parser.add_argument('--multibranch_loss_weight_iFR', type=float, help='iFR regression loss weight', default=10.0)
    parser.add_argument('--multibranch_error_regression_threshold', type=float, help='regression threshold for accuracy', default=0.05)
    parser.add_argument('--enable_clinicalData', type=int, help='use clinical data in the model', choices=[0,1], default=0)
    parser.add_argument('--len_clinicalData', type=int, help='if using clinical data, set dimensionality')
    parser.add_argument('--enable_doubleView', type=int, help='use second view in the model', choices=[0,1], default=0)
    parser.add_argument('--input3d', type=int, help='use 3D view in the model', choices=[0,1], default=1)
    parser.add_argument('--input2d', type=int, help='use 2D view in the model', choices=[0,1], default=0)
    parser.add_argument('--reduceInChannel', type=int, help='change model input channel', choices=[0,1], default=1)
    parser.add_argument('--freezeBackbone', type=int, help='if use backbone pretrained, freeze backbone weight', choices=[0,1], default=0)
    parser.add_argument('--enableNonLocalBlock', type=int, help='enable non-local block', choices=[0,1], default=0)
    parser.add_argument('--enableTemporalNonLocalBlock', type=int, help='enable temporal non-local block', choices=[0,1], default=0)
    parser.add_argument('--enableSpatioTemporalNonLocalBlock', type=int, help='enable spatio-temporal non-local block', choices=[0,1], default=0)
    parser.add_argument('--numNonLocalBlock', type=int, help='if using any type of non-local block, number of blocks', default=4)
    parser.add_argument('--enableGlobalMultiHeadAttention', type=int, help='enable global attention', choices=[0,1], default=0)
    parser.add_argument('--enableTemporalMultiHeadAttention', type=int, help=' enable temporal attention', choices=[0,1], default=0)
    parser.add_argument('--numHeadMultiHeadAttention', type=int, help='if using any type of attention, number of heads', default=8)
    parser.add_argument('--enableTemporalGru', type=int, help='enable temporal GRU', choices=[0,1], default=0)
    parser.add_argument('--numLayerGru', type=int, help='if using temporal GRU, number of layers', default=4)
    parser.add_argument('--enableTemporalLstm', type=int, help='enable temporal LSTM', choices=[0,1], default=0)
    parser.add_argument('--numLayerLstm', type=int, help='if using temporal LSTM, number of layers', default=4)
    parser.add_argument('--enableGlobalTransformerEncoder', type=int, help='enable global transformer', choices=[0,1], default=0)
    parser.add_argument('--enableTemporalTransformerEncoder', type=int, help='enable temporal transformer', choices=[0,1], default=0)
    parser.add_argument('--enableSpatialTemporalTransformerEncoder', type=int, help='enable factorized spatio-temporal transformer', choices=[0,1], default=1)
    parser.add_argument('--numLayerTransformerEncoder', type=int, help='if using transformer module, number of layers for each module', default=2)
    parser.add_argument('--numHeadGlobalTransformer', type=int, help='if using global transformer, number of heads', default=4)
    parser.add_argument('--numHeadSpatialTransformer', type=int, help='if using spatial transformer, number of heads', default=4)
    parser.add_argument('--numHeadTemporalTransformer', type=int, help='if using temporal transformer, number of heads', default=4)
    parser.add_argument('--transformerNormFirst', type=int, help='if using transformer, move normalization before MSA', default=0)

    parser.add_argument('--gradient_clipping_value', type=int, help='gradient clipping value', default=0)
    parser.add_argument('--optimizer', type=str, help='optimizer (SGD, Adam, AdamW, RMSprop, LBFGS)', choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'LBFGS'], default='AdamW')
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization weight', default=5e-4)
    parser.add_argument('--enable_scheduler', type=int, help='enable learning rate scheduler', choices=[0,1], default=0)
    parser.add_argument('--scheduler_factor', type=float, help='if using scheduler, factor of increment/redution', default=8e-2)
    parser.add_argument('--scheduler_threshold', type=float, help='if using scheduler, threshold for learning rate update', default=1e-2)
    parser.add_argument('--scheduler_patience', type=int, help='if using scheduler, number of epochs before updating the learning rate', default=5)

    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=300)
    parser.add_argument('--experiment', type=str, help='experiment name (in None, default is timestamp_modelname)', default=None)
    parser.add_argument('--logdir', type=str, help='log directory path', default='./logs')
    parser.add_argument('--start_tensorboard_server', type=int, help='start tensorboard server', choices=[0,1], default=0)
    parser.add_argument('--tensorboard_port', type=int, help='if starting tensorboard server, port (if unavailable, try the next ones)', default=6006)
    parser.add_argument('--saveLogsError', type=int, help='save detailed logs of error prediction/scores', default=0)
    parser.add_argument('--saveLogs', type=int, help='save detailed logs of prediction/scores', default=1)
    parser.add_argument('--ckpt_every', type=int, help='checkpoint saving frequenct (in epochs); -1 saves only best-validation and best-test checkpoints', default=-1)
    parser.add_argument('--resume', type=str, help='if not None, checkpoint path to resume', default=None)
    parser.add_argument('--save_array_file', type=int, help='save all plots as array', default=1)
    parser.add_argument('--save_image_file', type=int, help='save all plots as image', default=0)

    parser.add_argument('--enable_cudaAMP', type=int, help='enable CUDA amp', choices=[0,1], default=1)
    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda[number])', default='cuda')
    parser.add_argument('--distributed', type=int, help='enable distribuited trainining', choices=[0,1], default=1)
    parser.add_argument('--dist_url', type=str, help='if using distributed training, other process path (ex: "env://" if same none)', default='env://')

    args = parser.parse_args()

    # Convert boolean (as integer) args to boolean type
    if args.dataset3d == 0:
        args.dataset3d = False
    else:
        args.dataset3d = True
    if args.dataset2d == 0:
        args.dataset2d = False
    else:
        args.dataset2d = True
    if args.datasetGrid == 0:
        args.datasetGrid = False
    else:
        args.datasetGrid = True
    if args.doubleViewInput == 0:
        args.doubleViewInput = False
    else:
        args.doubleViewInput = True
    if args.enable_multibranch_ffr == 0:
        args.enable_multibranch_ffr = False
    else:
        args.enable_multibranch_ffr = True
    if args.enable_multibranch_ifr == 0:
        args.enable_multibranch_ifr = False
    else:
        args.enable_multibranch_ifr = True
    if args.enable_clinicalData == 0:
        args.enable_clinicalData = False
    else:
        args.enable_clinicalData = True
    if args.enable_doubleView == 0:
        args.enable_doubleView = False
    else:
        args.enable_doubleView = True
    if args.input3d == 0:
        args.input3d = False
    else:
        args.input3d = True
    if args.input2d == 0:
        args.input2d = False
    else:
        args.input2d = True
    if args.reduceInChannel == 0:
        args.reduceInChannel = False
    else:
        args.reduceInChannel = True
    if args.freezeBackbone == 0:
        args.freezeBackbone = False
    else:
        args.freezeBackbone = True
    if args.enableNonLocalBlock == 0:
        args.enableNonLocalBlock = False
    else:
        args.enableNonLocalBlock = True
    if args.enableTemporalNonLocalBlock == 0:
        args.enableTemporalNonLocalBlock = False
    else:
        args.enableTemporalNonLocalBlock = True
    if args.enableSpatioTemporalNonLocalBlock == 0:
        args.enableSpatioTemporalNonLocalBlock = False
    else:
        args.enableSpatioTemporalNonLocalBlock = True
    if args.enableGlobalMultiHeadAttention == 0:
        args.enableGlobalMultiHeadAttention = False
    else:
        args.enableGlobalMultiHeadAttention = True
    if args.enableTemporalMultiHeadAttention == 0:
        args.enableTemporalMultiHeadAttention = False
    else:
        args.enableTemporalMultiHeadAttention = True
    if args.enableTemporalGru == 0:
        args.enableTemporalGru = False
    else:
        args.enableTemporalGru = True
    if args.enableTemporalLstm == 0:
        args.enableTemporalLstm = False
    else:
        args.enableTemporalLstm = True
    if args.enableGlobalTransformerEncoder == 0:
        args.enableGlobalTransformerEncoder = False
    else:
        args.enableGlobalTransformerEncoder = True
    if args.enableTemporalTransformerEncoder == 0:
        args.enableTemporalTransformerEncoder = False
    else:
        args.enableTemporalTransformerEncoder = True
    if args.enableSpatialTemporalTransformerEncoder == 0:
        args.enableSpatialTemporalTransformerEncoder = False
    else:
        args.enableSpatialTemporalTransformerEncoder = True
    if args.transformerNormFirst == 0:
        args.transformerNormFirst = False
    else:
        args.transformerNormFirst = True
    if args.enable_scheduler == 0:
        args.enable_scheduler = False
    else:
        args.enable_scheduler = True
    if args.start_tensorboard_server == 0:
        args.start_tensorboard_server = False
    else:
        args.start_tensorboard_server = True
    if args.saveLogsError == 0:
        args.saveLogsError = False
    else:
        args.saveLogsError = True
    if args.saveLogs == 0:
        args.saveLogs = False
    else:
        args.saveLogs = True
    if args.save_array_file == 0:
        args.save_array_file = False
    else:
        args.save_array_file = True
    if args.save_image_file == 0:
        args.save_image_file = False
    else:
        args.save_image_file = True
    if args.enable_cudaAMP == 0:
        args.enable_cudaAMP = False
    else:
        args.enable_cudaAMP = True
    if args.distributed == 0:
        args.distributed = False
    else:
        args.distributed = True
    
    # Create some dict
    args.multibranch_dropout = {'label_dropout':args.multibranch_dropout_label, 'FFR_dropout':args.multibranch_dropout_FFR, 'iFR_dropout':args.multibranch_dropout_iFR}
    del args.multibranch_dropout_label
    del args.multibranch_dropout_FFR
    del args.multibranch_dropout_iFR
    args.multibranch_loss_weight = {'label_weight':args.multibranch_loss_weight_label, 'FFR_weight':args.multibranch_loss_weight_FFR, 'iFR_weight':args.multibranch_loss_weight_iFR}
    del args.multibranch_loss_weight_label
    del args.multibranch_loss_weight_FFR
    del args.multibranch_loss_weight_iFR

    # Generate experiment tags if not defined
    if args.experiment == None:
        args.experiment = args.model

    # Define pads automatically
    if args.pad[1] == -1:
        args.pad = [args.pad[0],args.resize[1],args.pad[2]]
    if args.pad[2] == -1:
        args.pad = [args.pad[0],args.pad[1],args.resize[2]]
    
    '''if (args.normalization_clinicalData != None):
        with open(os.path.join(args.normalization_clinicalData)) as fp:
            normalizations = json.load(fp)
        args.means_clinicalData = normalizations["means"]
        args.stds_clinicalData = normalizations["stds"]
    else:
        args.means_clinicalData = None
        args.stds_clinicalData = None'''
    
    if args.enable_doubleView and not args.doubleViewInput:
        raise RuntimeError("Multibranch input require multi views dataset.")
    
    if args.model == 'None':
        if (args.dataset3d) or (args.dataset2d) or (args.doubleViewInput) or (args.enable_multibranch_ffr) or (args.enable_multibranch_ifr) or (args.enable_doubleView) or (args.input3d) or (args.input2d) or (args.reduceInChannel) or (args.freezeBackbone) or (args.enableGlobalMultiHeadAttention) or (args.enableTemporalMultiHeadAttention) or (args.enableTemporalLstm) or (args.enableGlobalTransformerEncoder) or (args.enableTemporalTransformerEncoder) or (args.enableSpatialTemporalTransformerEncoder):
            raise RuntimeError("To use None model, all model/dataset parameters (dataset3d, dataset2d, doubleViewInput, enable_multibranch_ffr, enable_multibranch_ifr, enable_doubleView, input3d, input2d, reduceInChannel, freezeBackbone, enableGlobalMultiHeadAttention, enableTemporalMultiHeadAttention, enableTemporalLstm, enableGlobalTransformerEncoder, enableTemporalTransformerEncoder, enableSpatialTemporalTransformerEncoder) must be False.")
    
    return args


# disable printing when not in master process
import builtins as __builtin__
builtin_print = __builtin__.print
def print_mod(*args, **kwargs):
    force = kwargs.pop('force', False)
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
    else:
        RuntimeError("No RANK found!")
    if (rank==0) or force:
        builtin_print(*args, **kwargs)
def no_print(*args, **kwargs):
    pass


def main():
    args = parse()

    # choose device
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            raise RuntimeError("Can't use distributed mode! Check if you don't run correct command: 'torchrun --master_addr=localhost --nproc_per_node=NUMBER_GPUS train.py'")
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'gloo' if ((platform.system() == 'Windows') or (args.device == 'cpu')) else 'nccl'
        print('| ' + args.dist_backend + ': distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        device = torch.device(args.gpu)
        # disable printing when not in master process
        #if args.rank != 0:
        #    __builtin__.print = no_print
    else:
        if args.device == 'cuda': # choose the most free gpu
            #mem = [(torch.cuda.memory_allocated(i)+torch.cuda.memory_reserved(i)) for i in range(torch.cuda.device_count())]
            mem = [gpu.memoryUtil for gpu in GPUtil.getGPUs()]
            args.device = 'cuda:' + str(mem.index(min(mem)))
        device = torch.device(args.device)
        print('Using device', args.device)

    # Dataset e Loader
    print("Dataset: balanced nested cross-validation use fold (test-set) " + str(args.num_fold) + " and inner_loop (validation-set) " + str(args.inner_loop) + ".")
    loaders, samplers, loss_weights = get_loader(args)

    # Model
    model = get_model(num_classes=2,
                        model_name=args.model,
                        enable_multibranch_ffr=args.enable_multibranch_ffr,
                        enable_multibranch_ifr=args.enable_multibranch_ifr,
                        multibranch_dropout=args.multibranch_dropout,
                        enable_clinicalData=args.enable_clinicalData,
                        in_dim_clinicalData=args.len_clinicalData if args.enable_clinicalData else None,
                        enable_doubleView=args.enable_doubleView,
                        enable_keyframe=(args.dataset3d and args.dataset2d),
                        reduceInChannel=args.reduceInChannel,
                        freezeBackbone = args.freezeBackbone,
                        enableNonLocalBlock = args.enableNonLocalBlock,
                        enableTemporalNonLocalBlock = args.enableTemporalNonLocalBlock,
                        enableSpatioTemporalNonLocalBlock = args.enableSpatioTemporalNonLocalBlock,
                        numNonLocalBlock= args.numNonLocalBlock,
                        enableGlobalMultiHeadAttention=args.enableGlobalMultiHeadAttention,
                        enableTemporalMultiHeadAttention=args.enableTemporalMultiHeadAttention,
                        numHeadMultiHeadAttention=args.numHeadMultiHeadAttention,
                        enableTemporalGru=args.enableTemporalGru,
                        numLayerGru=args.numLayerGru,
                        enableTemporalLstm=args.enableTemporalLstm,
                        numLayerLstm=args.numLayerLstm,
                        enableGlobalTransformerEncoder=args.enableGlobalTransformerEncoder,
                        enableTemporalTransformerEncoder=args.enableTemporalTransformerEncoder,
                        enableSpatialTemporalTransformerEncoder=args.enableSpatialTemporalTransformerEncoder,
                        numLayerTransformerEncoder=args.numLayerTransformerEncoder,
                        numHeadGlobalTransformer=args.numHeadGlobalTransformer,
                        numHeadSpatialTransformer=args.numHeadSpatialTransformer,
                        numHeadTemporalTransformer=args.numHeadTemporalTransformer,
                        transformerNormFirst=args.transformerNormFirst,
                        loss_weights=loss_weights,
                        batch_size=args.batch_size,
                        input_size=args.pad)
    if args.resume is not None:
        model.load_state_dict(Saver.load_model(args['resume']), strict=True)
    model.to(device)

    # Enable model distribuited if it is
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:', args.model, '(number of params:', n_parameters, ')')

    # Create optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model_without_ddp.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model_without_ddp.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS(params=model_without_ddp.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Optimizer chosen not implemented!")
    
    # Create scheduler
    if args.enable_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='min',
                                                                factor=args.scheduler_factor,
                                                                patience=args.scheduler_patience,
                                                                threshold=args.scheduler_threshold,
                                                                threshold_mode='rel',
                                                                cooldown=0,
                                                                min_lr=0,
                                                                eps=1e-08,
                                                                verbose=True)

    if args.enable_cudaAMP:
        # Creates GradScaler for CUDA AMP
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Trainer
    class_trainer = Trainer(net=model,
                            class_weights=torch.Tensor(loss_weights).to(device),
                            optim=optimizer,
                            gradient_clipping_value=args.gradient_clipping_value,
                            enable_multibranch_ffr=args.enable_multibranch_ffr,
                            enable_multibranch_ifr=args.enable_multibranch_ifr,
                            multibranch_loss_weight=args.multibranch_loss_weight,
                            enable_clinicalData=args.enable_clinicalData,
                            doubleViewInput=args.doubleViewInput,
                            input3d=args.input3d,
                            input2d=args.input2d,
                            scaler=scaler)

    # Saver
    if (not args.distributed) or (args.distributed and (args.rank==0)):
        saver = Saver(Path(args.logdir),
                        vars(args),
                        sub_dirs=list(loaders.keys()),
                        tag=args.experiment)
    else:
        saver = None

    tot_predicted_labels_last = {split:{} for split in loaders}
    if (saver is not None) and (args.ckpt_every <= 0):
        max_validation_accuracy_balanced = 0
        max_test_accuracy_balanced = 0
        save_this_epoch = False
    for epoch in range(args.epochs):
        try:
            for split in loaders:
                if args.distributed:
                    samplers[split].set_epoch(epoch)

                data_loader = loaders[split]

                epoch_metrics = {}
                tot_true_labels = []
                tot_true_FFRs = []
                tot_true_iFRs = []
                tot_predicted_labels = []
                tot_predicted_scores = []
                if args.enable_multibranch_ffr:
                    tot_predicted_FFRs = []
                if args.enable_multibranch_ifr:
                    tot_predicted_iFRs = []
                tot_image_paths = []
                for batch in tqdm(data_loader, desc=f'{split}, {epoch}/{args.epochs}'):
                    labels, image_paths, FFRs, iFRs = batch['label'], batch['image'], batch['FFR'], batch['iFR']
                    
                    images_3d = None
                    doubleView_3d = None
                    if args.dataset3d:
                        images_3d = batch['image_3d']
                        if args.doubleViewInput:
                            doubleView_3d = batch['image2_3d']

                    images_2d = None
                    doubleView_2d = None
                    if args.dataset2d:
                        images_2d = batch['image_2d']
                        if args.doubleViewInput:
                            doubleView_2d = batch['image2_2d']
                    
                    clinicalData = None
                    if args.enable_clinicalData:
                        clinicalData = torch.cat((batch['age_array'], batch['sex'].unsqueeze(1), batch['ckd'].unsqueeze(1), batch['ef_array'], batch['stemi'].unsqueeze(1), batch['nstemi'].unsqueeze(1), batch['ua'].unsqueeze(1), batch['stable_angima'].unsqueeze(1), batch['positive_stress_test'].unsqueeze(1)), dim=1)
                    
                    tot_true_labels.extend(labels.tolist())
                    tot_image_paths.extend(image_paths)
                    tot_true_FFRs.extend(FFRs.tolist())
                    tot_true_iFRs.extend(iFRs.tolist())

                    if args.dataset3d:
                        images_3d = images_3d.to(device)
                        if args.doubleViewInput:
                            doubleView_3d = doubleView_3d.to(device)
                    if args.dataset2d:
                        images_2d = images_2d.to(device)
                        if args.doubleViewInput:
                            doubleView_2d = doubleView_2d.to(device)
                    
                    if args.enable_clinicalData:
                        clinicalData = clinicalData.to(device)

                    labels = labels.to(device)
                    FFRs = FFRs.to(device)
                    iFRs = iFRs.to(device)

                    returned_values = class_trainer.forward_batch(images_3d, images_2d, labels, FFRs, iFRs, clinicalData, doubleView_3d, doubleView_2d, split)
                    if args.enable_multibranch_ffr and args.enable_multibranch_ifr:
                        metrics_dict, (predicted_labels, predicted_scores, predicted_FFRs, predicted_iFRs) = returned_values
                        tot_predicted_FFRs.extend(predicted_FFRs.tolist())
                        tot_predicted_iFRs.extend(predicted_iFRs.tolist())
                    elif args.enable_multibranch_ffr:
                        metrics_dict, (predicted_labels, predicted_scores, predicted_FFRs) = returned_values
                        tot_predicted_FFRs.extend(predicted_FFRs.tolist())
                    elif args.enable_multibranch_ifr:
                        metrics_dict, (predicted_labels, predicted_scores, predicted_iFRs) = returned_values
                        tot_predicted_iFRs.extend(predicted_iFRs.tolist())
                    else:
                        metrics_dict, (predicted_labels, predicted_scores) = returned_values
                    
                    tot_predicted_labels.extend(predicted_labels.tolist())
                    tot_predicted_scores.extend(predicted_scores.tolist())
                    
                    for k, v in metrics_dict.items():
                        epoch_metrics[k]= epoch_metrics[k] + [v] if k in epoch_metrics else [v]
                
                # Run scheduler
                if args.enable_scheduler and split=="train":
                    scheduler.step(sum(epoch_metrics['loss'])/len(epoch_metrics['loss']))

                # Print metrics
                for k, v in epoch_metrics.items():
                    avg_v = sum(v)/len(v)
                    if args.distributed:
                        torch.distributed.barrier()
                        avg_v_output = [None for _ in range(args.world_size)]
                        torch.distributed.all_gather_object(avg_v_output, avg_v)
                        avg_v = sum(avg_v_output)/len(avg_v_output)
                    if saver is not None:
                        saver.log_scalar("Classifier Epoch/"+k+"_"+split, avg_v, epoch)
                
                if args.distributed:
                    torch.distributed.barrier()

                    tot_true_labels_output = [None for _ in range(args.world_size)]
                    tot_true_FFRs_output = [None for _ in range(args.world_size)]
                    tot_true_iFRs_output = [None for _ in range(args.world_size)]
                    tot_predicted_labels_output = [None for _ in range(args.world_size)]
                    tot_predicted_scores_output = [None for _ in range(args.world_size)]
                    if args.enable_multibranch_ffr:
                        tot_predicted_FFRs_output = [None for _ in range(args.world_size)]
                    if args.enable_multibranch_ifr:
                        tot_predicted_iFRs_output = [None for _ in range(args.world_size)]
                    tot_image_paths_output = [None for _ in range(args.world_size)]

                    torch.distributed.all_gather_object(tot_true_labels_output, tot_true_labels)
                    torch.distributed.all_gather_object(tot_true_FFRs_output, tot_true_FFRs)
                    torch.distributed.all_gather_object(tot_true_iFRs_output, tot_true_iFRs)
                    torch.distributed.all_gather_object(tot_predicted_labels_output, tot_predicted_labels)
                    torch.distributed.all_gather_object(tot_predicted_scores_output, tot_predicted_scores)
                    if args.enable_multibranch_ffr:
                        torch.distributed.all_gather_object(tot_predicted_FFRs_output, tot_predicted_FFRs)
                    if args.enable_multibranch_ifr:
                        torch.distributed.all_gather_object(tot_predicted_iFRs_output, tot_predicted_iFRs)
                    torch.distributed.all_gather_object(tot_image_paths_output, tot_image_paths)

                    tot_true_labels=[]
                    tot_true_FFRs=[]
                    tot_true_iFRs=[]
                    tot_predicted_labels=[]
                    tot_predicted_scores=[]
                    if args.enable_multibranch_ffr:
                        tot_predicted_FFRs=[]
                    if args.enable_multibranch_ifr:
                        tot_predicted_iFRs=[]
                    tot_image_paths=[]
                    for i in range(len(tot_true_labels_output)):
                        tot_true_labels.extend(tot_true_labels_output[i])
                        tot_true_FFRs.extend(tot_true_FFRs_output[i])
                        tot_true_iFRs.extend(tot_true_iFRs_output[i])
                        tot_predicted_labels.extend(tot_predicted_labels_output[i])
                        tot_predicted_scores.extend(tot_predicted_scores_output[i])
                        if args.enable_multibranch_ffr:
                            tot_predicted_FFRs.extend(tot_predicted_FFRs_output[i])
                        if args.enable_multibranch_ifr:
                            tot_predicted_iFRs.extend(tot_predicted_iFRs_output[i])
                        tot_image_paths.extend(tot_image_paths_output[i])
                
                if saver is not None:
                    # Accuracy Balanced classification
                    accuracy_balanced = utils.calc_accuracy_balanced_classification(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch/accuracy_balanced_"+split, accuracy_balanced, epoch)
                    if (saver is not None) and (args.ckpt_every <= 0):
                        if (split == "validation") and (accuracy_balanced >= max_validation_accuracy_balanced):
                            max_validation_accuracy_balanced = accuracy_balanced
                            save_this_epoch = True
                        if (split == "test") and (accuracy_balanced >= max_test_accuracy_balanced):
                            max_test_accuracy_balanced = accuracy_balanced
                            save_this_epoch = True
                    
                    # Accuracy classification
                    accuracy = utils.calc_accuracy_classification(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch "+split+"/accuracy_"+split, accuracy, epoch)

                    # Precision
                    precision = utils.calc_precision(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Precision", precision, epoch)

                    # Recall
                    recall = utils.calc_recall(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Recall", recall, epoch)

                    # Specificity
                    specificity = utils.calc_specificity(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Specificity", specificity, epoch)

                    # F1 Score
                    f1score = utils.calc_f1(tot_true_labels, tot_predicted_labels)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"F1 Score", f1score, epoch)

                    # AUC
                    auc = utils.calc_auc(tot_true_labels, tot_predicted_scores)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"AUC", auc, epoch)

                    # Precision-Recall Score
                    prc_score = utils.calc_aps(tot_true_labels, tot_predicted_scores)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"PRscore", prc_score, epoch)
            
                    # Calibration: Brier Score
                    brier_score = utils.calc_brierScore(tot_true_labels, tot_predicted_scores)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Brier Score", brier_score, epoch)

                    # ROC Curve
                    falsePositiveRate_array, truePositiveRate_array, _ = utils.calc_rocCurve(tot_true_labels, tot_predicted_scores)
                    rocCurve_figure, rocCurve_image = utils.plot_rocCurve(falsePositiveRate_array, truePositiveRate_array)
                    if args.save_array_file:
                        saver.save_figure({'type':'rocCurve', 'falsePositiveRate_array':falsePositiveRate_array, 'truePositiveRate_array':truePositiveRate_array}, epoch, split, "ROCcurve")
                    if args.save_image_file:
                        saver.save_image(rocCurve_image, epoch, split, "ROCcurve")
                    saver.log_figure("Classifier Epoch "+split+"/"+"ROC Curve", rocCurve_figure, epoch)

                    # Precision-Recall Curve
                    precision_array, recall_array, _ = utils.calc_precisionRecallCurve(tot_true_labels, tot_predicted_scores)
                    precisionRecallCurve_figure, precisionRecallCurve_image = utils.plot_precisionRecallCurve(precision_array, recall_array)
                    if args.save_array_file:
                        saver.save_figure({'type':'precisionRecallCurve', 'precision_array':precision_array, 'recall_array':recall_array}, epoch, split, "PrecisionRecallCurve")
                    if args.save_image_file:
                        saver.save_image(precisionRecallCurve_image, epoch, split, "PrecisionRecallCurve")
                    saver.log_figure("Classifier Epoch "+split+"/"+"Precision-Recall Curve", precisionRecallCurve_figure, epoch)
                    
                    # Prediction Agreement Rate: same-sample evaluation agreement between current and previous epoch
                    predictionAgreementRate, tot_predicted_labels_last[split] = utils.calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last[split], tot_image_paths)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Prediction Agreement Rate", predictionAgreementRate, epoch)
        
                    # Confusion Matrix
                    confusionMatrix_array = utils.calc_confusionMatrix(tot_true_labels, tot_predicted_labels)
                    confusionMatrix_figure, confusionMatrix_image = utils.plot_confusionMatrix(confusionMatrix_array, ['negative', 'positive'], "Confusion Matrix")
                    if args.save_array_file:
                        saver.save_figure({'type':'confusionMatrix', 'confusionMatrix_array':confusionMatrix_array}, epoch, split, "ConfusionMatrix")
                    if args.save_image_file:
                        saver.save_image(confusionMatrix_image, epoch, split, "ConfusionMatrix")
                    saver.log_figure("Classifier Epoch "+split+"/"+"Confusion Matrix", confusionMatrix_figure, epoch)

                    # Histograms FFR of FN and FP
                    tot_FFRs_FP, tot_FFRs_FN = utils.calc_FN_FP_histograms(tot_true_labels, tot_predicted_labels, tot_true_FFRs)
                    histogramFFRs_figure, histogramFFRs_image = utils.plot_FN_FP_histograms(tot_FFRs_FP, tot_FFRs_FN, "FFR")
                    if args.save_array_file:
                        saver.save_figure({'type':'histogramFFRs', 'tot_FFRs_FP':tot_FFRs_FP, 'tot_FFRs_FN':tot_FFRs_FN}, epoch, split, "HistogramFFRs")
                    if args.save_image_file:
                        saver.save_image(histogramFFRs_image, epoch, split, "HistogramFFRs")
                    saver.log_figure("Classifier Epoch "+split+"/"+"Histogram FFRs", histogramFFRs_figure, epoch)

                    # Histograms iFR of FN and FP
                    tot_iFRs_FP, tot_iFRs_FN = utils.calc_FN_FP_histograms(tot_true_labels, tot_predicted_labels, tot_true_iFRs)
                    histogramiFRs_figure, histogramiFRs_image = utils.plot_FN_FP_histograms(tot_iFRs_FP, tot_iFRs_FN, "iFR")
                    if args.save_array_file:
                        saver.save_figure({'type':'histogramiFRs', 'tot_iFRs_FP':tot_iFRs_FP, 'tot_iFRs_FN':tot_iFRs_FN}, epoch, split, "HistogramiFRs")
                    if args.save_image_file:
                        saver.save_image(histogramiFRs_image, epoch, split, "HistogramiFRs")
                    saver.log_figure("Classifier Epoch "+split+"/"+"Histogram iFRs", histogramiFRs_figure, epoch)
                    
                    if args.enable_multibranch_ffr:
                        # Accuracy regression
                        accuracy_FFRs = utils.calc_accuracy_regression(tot_true_FFRs, tot_predicted_FFRs, args.multibranch_error_regression_threshold)
                        saver.log_scalar("Classifier Epoch "+split+"/accuracy_FFRs_"+split, accuracy_FFRs, epoch)

                        # Accuracy balanced labels da regression FFR
                        accuracyBalalnced_labels_FFRs = utils.calc_accuracyBalanced_regression_labels(tot_true_labels, tot_predicted_FFRs, 0.8)
                        saver.log_scalar("Classifier Epoch "+split+"/accuracy_balanced_labelsDaFFRs_"+split, accuracyBalalnced_labels_FFRs, epoch)

                        # Accuracy labels da regression FFR
                        accuracy_labels_FFRs = utils.calc_accuracy_regression_labels(tot_true_labels, tot_predicted_FFRs, 0.8)
                        saver.log_scalar("Classifier Epoch "+split+"/accuracy_labelsDaFFRs_"+split, accuracy_labels_FFRs, epoch)
                
                        # MAE FFR
                        mae_FFRs = utils.calc_mae(tot_true_FFRs, tot_predicted_FFRs)
                        saver.log_scalar("Classifier Epoch "+split+"/mae_FFRs_"+split, mae_FFRs, epoch)
                        
                        # MSE FFR
                        mse_FFRs = utils.calc_mse(tot_true_FFRs, tot_predicted_FFRs)
                        saver.log_scalar("Classifier Epoch "+split+"/mse_FFRs_"+split, mse_FFRs, epoch)
                
                        # Skewness FFR
                        skewness_FFRs = utils.calc_skewness(tot_true_FFRs, tot_predicted_FFRs)
                        saver.log_scalar("Classifier Epoch "+split+"/skewness_FFRs_"+split, skewness_FFRs, epoch)

                        # Prediction FFR error: histogram of FFR values ​​with wrong prediction
                        predictionErrorFFR_array = utils.calc_predictionError_histograms(tot_true_FFRs, tot_predicted_FFRs, args.multibranch_error_regression_threshold)
                        predictionError_FFRs_figure, predictionError_FFRs_image = utils.plot_predictionError_histograms(predictionErrorFFR_array, "FFR")
                        if args.save_array_file:
                            saver.save_figure({'type':'predictionError_FFRs', 'predictionErrorFFR_array':predictionErrorFFR_array}, epoch, split, "HistogramPredictionErrorFFRs")
                        if args.save_image_file:
                            saver.save_image(predictionError_FFRs_image, epoch, split, "HistogramPredictionErrorFFRs")
                        saver.log_figure("Classifier Epoch "+split+"/"+"Prediction error FFRs", predictionError_FFRs_figure, epoch)
                    
                    if args.enable_multibranch_ifr:
                        # Accuracy regression
                        accuracy_iFRs = utils.calc_accuracy_regression(tot_true_iFRs, tot_predicted_iFRs, args.multibranch_error_regression_threshold)
                        saver.log_scalar("Classifier Epoch "+split+"/accuracy_iFRs_"+split, accuracy_iFRs, epoch)

                        # Accuracy balanced labels da regression iFR
                        accuracyBalalnced_labels_iFRs = utils.calc_accuracyBalanced_regression_labels(tot_true_labels, tot_predicted_iFRs, 0.89)
                        saver.log_scalar("Classifier Epoch "+split+"/accuracy_balanced_labelsDaiFRs_"+split, accuracyBalalnced_labels_iFRs, epoch)

                        # Accuracy labels da regression iFR
                        accuracy_labels_iFRs = utils.calc_accuracy_regression_labels(tot_true_labels, tot_predicted_iFRs, 0.89)
                        saver.log_scalar("Classifier Epoch "+split+"/accuracy_labelsDaiFRs_"+split, accuracy_labels_iFRs, epoch)
                
                        # MAE iFR
                        mae_iFRs = utils.calc_mae(tot_true_iFRs, tot_predicted_iFRs)
                        saver.log_scalar("Classifier Epoch "+split+"/mae_iFRs_"+split, mae_iFRs, epoch)
                        
                        # MSE iFR
                        mse_iFRs = utils.calc_mse(tot_true_iFRs, tot_predicted_iFRs)
                        saver.log_scalar("Classifier Epoch "+split+"/mse_iFRs_"+split, mse_iFRs, epoch)
                
                        # Skewness iFR
                        skewness_iFRs = utils.calc_skewness(tot_true_iFRs, tot_predicted_iFRs)
                        saver.log_scalar("Classifier Epoch "+split+"/skewness_iFRs_"+split, skewness_iFRs, epoch)

                        # Prediction iFR error: histogram of FFR values ​​with wrong prediction
                        predictionErroriFR_array = utils.calc_predictionError_histograms(tot_true_iFRs, tot_predicted_iFRs, args.multibranch_error_regression_threshold)
                        predictionError_iFRs_figure, predictionError_iFRs_image = utils.plot_predictionError_histograms(predictionErroriFR_array, "iFR")
                        if args.save_array_file:
                            saver.save_figure({'type':'predictionError_iFRs', 'predictionErroriFR_array':predictionErroriFR_array}, epoch, split, "HistogramPredictionErroriFRs")
                        if args.save_image_file:
                            saver.save_image(predictionError_iFRs_image, epoch, split, "HistogramPredictionErroriFRs")
                        saver.log_figure("Classifier Epoch "+split+"/"+"Prediction error iFRs", predictionError_iFRs_figure, epoch)

                    # Error/Percentual error prediction per hospital
                    hospitals_list, counts_list, counts_percentual_list = utils.calc_predictionErrorHospital_histogram(tot_true_labels, tot_predicted_labels, tot_image_paths)
                    histogramPredictionErrorHospital_figure, histogramPredictionErrorHospital_image = utils.plot_predictionErrorHospital_histogram(hospitals_list, counts_list)
                    histogramPredictionPercentualErrorHospital_figure, histogramPredictionPercentualErrorHospital_image = utils.plot_predictionPercentualErrorHospital_histogram(hospitals_list, counts_percentual_list)
                    if args.save_array_file:
                        saver.save_figure({'type':'predictionErrorHospital_histogram', "hospitals_list":hospitals_list, "counts_list":counts_list}, epoch, split, "HistogramPredictionErrorHospital")
                        saver.save_figure({'type':'predictionPercentualErrorHospital_histogram', "hospitals_list":hospitals_list, "counts_percentual_list":counts_percentual_list}, epoch, split, "HistogramPredictionPercentualErrorHospital")
                    if args.save_image_file:
                        saver.save_image(histogramPredictionErrorHospital_image, epoch, split, "HistogramPredictionErrorHospital")
                        saver.save_image(histogramPredictionPercentualErrorHospital_image, epoch, split, "HistogramPredictionPercentualErrorHospital")
                    saver.log_figure("Classifier Epoch "+split+"/"+"Histogram prediction error hospital", histogramPredictionErrorHospital_figure, epoch)
                    saver.log_figure("Classifier Epoch "+split+"/"+"Histogram prediction percentual error hospital", histogramPredictionPercentualErrorHospital_figure, epoch)

                    # Save logs of error
                    dict_other_info = {'image_path':tot_image_paths, 'ffr':tot_true_FFRs, 'ifr':tot_true_iFRs}
                    if args.enable_multibranch_ffr:
                        dict_other_info['ffr_predicted'] = tot_predicted_FFRs
                    if args.enable_multibranch_ifr:
                        dict_other_info['ifr_predicted'] = tot_predicted_iFRs
                    if args.saveLogsError:
                        saver.saveLogsError(saver.output_path[split]/f'{split}_logs_error_{epoch:05d}', tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
                
                    # Save logs
                    if args.saveLogs:
                        Saver.saveLogs(saver.output_path[split]/f'{split}_logs_{epoch:05d}', tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
                        
                # Save checkpoint
                if args.distributed:
                    torch.distributed.barrier()
                if saver is not None:
                    if args.ckpt_every > 0:
                        if (split ==  "train") and (epoch % args.ckpt_every == 0):
                            saver.save_model(model_without_ddp, args.experiment, epoch)
                    else: # args.ckpt_every <= 0
                        if save_this_epoch:
                            for filename in glob.glob(str(saver.ckpt_path / (args.experiment+"_best_"+split+"_*"))):
                                os.remove(filename)
                            saver.save_model(model_without_ddp, args.experiment+"_best_"+split, epoch)
                        save_this_epoch = False
        except KeyboardInterrupt:
            print('Caught Keyboard Interrupt: exiting...')
            break

    # Save last checkpoint
    if args.distributed:
        torch.distributed.barrier()
    if saver is not None:
        if args.ckpt_every > 0:
            saver.save_model(model_without_ddp, args.experiment, epoch)
        saver.close()

    if args.start_tensorboard_server:
        print("Finish (Press CTRL+C to close tensorboard and quit)")
    else:
        print("Finish")

    if args.distributed:
        torch.distributed.destroy_process_group()
        

if __name__ == '__main__':
    main()