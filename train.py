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
import GPUtil

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, help='dataset folder path')
    parser.add_argument('--split_path', type=Path, help='json dataset metadata for MONAI')
    parser.add_argument('--num_fold', type=int, help='test fold for nested cross-validation', default=0)
    parser.add_argument('--inner_loop', type=int, help='validation fold for nested cross-validation', default=0)
    parser.add_argument('--cache_rate', type=float, help='fraction of dataset to be cached in RAM', default=1.0)

    parser.add_argument('--resize', type=int, help='(T,H,W) resize dimensions', nargs=3, default=[-1,256,256])
    parser.add_argument('--pad', type=int, help='(T,H,W) padding; use -1 to not modify that dimension', nargs=3, default=[60,-1,-1])
    parser.add_argument('--mean', type=float, help='normalization mean', default=0.0)
    parser.add_argument('--std', type=float, help='normalization standard deviation', default=1.0)

    parser.add_argument('--model', type=str, help='model', default='resnet3d_pretrained')

    parser.add_argument('--enable_multibranch_ffr', type=int, help='enable FFR regression output branch', choices=[0,1], default=1)
    parser.add_argument('--enable_multibranch_ifr', type=int, help='enable iFR regression output branch', choices=[0,1], default=1)
    parser.add_argument('--multibranch_dropout_label', type=float, help='dropout for classification branch [0.0-1.0]', default=0.0)
    parser.add_argument('--multibranch_dropout_FFR', type=float, help='dropout for FFR regression branch [0.0-1.0]', default=0.0)
    parser.add_argument('--multibranch_dropout_iFR', type=float, help='dropout of iFR regression branch [0.0-1.0]', default=0.0)
    parser.add_argument('--multibranch_loss_weight_label', type=float, help='classification loss weight', default=1.0)
    parser.add_argument('--multibranch_loss_weight_FFR', type=float, help='FFR regression loss weight', default=10.0)
    parser.add_argument('--multibranch_loss_weight_iFR', type=float, help='iFR regression loss weight', default=10.0)
    parser.add_argument('--multibranch_error_regression_threshold', type=float, help='regression threshold for accuracy', default=0.05)
    parser.add_argument('--enableGlobalMultiHeadAttention', type=int, help='enable global attention', choices=[0,1], default=0)
    parser.add_argument('--enableTemporalMultiHeadAttention', type=int, help=' enable temporal attention', choices=[0,1], default=0)
    parser.add_argument('--enableSpatialTemporalTransformerEncoder', type=int, help='enable factorized spatio-temporal transformer', choices=[0,1], default=1)
    parser.add_argument('--numLayerTransformerEncoder', type=int, help='if using factorized spatio-temporal transformer, number of layers', default=2)
    parser.add_argument('--numHeadMultiHeadAttention', type=int, help='if using any type of attention/transformer, number of heads', default=4)

    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization weight', default=5e-4)

    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=300)
    parser.add_argument('--experiment', type=str, help='experiment name (in None, default is timestamp_modelname)', default=None)
    parser.add_argument('--logdir', type=str, help='log directory path', default='./logs')
    parser.add_argument('--start_tensorboard_server', type=int, help='start tensorboard server', choices=[0,1], default=0)
    parser.add_argument('--tensorboard_port', type=int, help='if starting tensorboard server, port (if unavailable, try the next ones)', default=6006)
    parser.add_argument('--saveLogs', type=int, help='save detailed logs of prediction/scores', default=1)
    parser.add_argument('--ckpt_every', type=int, help='checkpoint saving frequenct (in epochs); -1 saves only best-validation and best-test checkpoints', default=-1)
    parser.add_argument('--resume', type=str, help='if not None, checkpoint path to resume', default=None)
    parser.add_argument('--save_image_file', type=int, help='save all plots', default=0)

    parser.add_argument('--enable_cudaAMP', type=int, help='enable CUDA amp', choices=[0,1], default=1)
    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda[number])', default='cuda')
    parser.add_argument('--distributed', type=int, help='enable distribuited trainining', choices=[0,1], default=0)
    parser.add_argument('--dist_url', type=str, help='if using distributed training, other process path (ex: "env://" if same none)', default='env://')

    args = parser.parse_args()

    if args.resize != [-1,256,256]:
        raise RuntimeError("To best performance, the dataset resize size should be [-1,256,256].")
    
    # Convert boolean (as integer) args to boolean type
    if args.enable_multibranch_ffr == 0:
        args.enable_multibranch_ffr = False
    else:
        args.enable_multibranch_ffr = True
    if args.enable_multibranch_ifr == 0:
        args.enable_multibranch_ifr = False
    else:
        args.enable_multibranch_ifr = True
    if args.enableGlobalMultiHeadAttention == 0:
        args.enableGlobalMultiHeadAttention = False
    else:
        args.enableGlobalMultiHeadAttention = True
    if args.enableTemporalMultiHeadAttention == 0:
        args.enableTemporalMultiHeadAttention = False
    else:
        args.enableTemporalMultiHeadAttention = True
    if args.enableSpatialTemporalTransformerEncoder == 0:
        args.enableSpatialTemporalTransformerEncoder = False
    else:
        args.enableSpatialTemporalTransformerEncoder = True
    if args.start_tensorboard_server == 0:
        args.start_tensorboard_server = False
    else:
        args.start_tensorboard_server = True
    if args.saveLogs == 0:
        args.saveLogs = False
    else:
        args.saveLogs = True
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
            raise RuntimeError("Can't use distributed mode! Check if you don't run correct command: 'python -m torch.distributed.launch --nproc_per_node=NUMBER_GPUS --use_env train.py'")
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'gloo'
        print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        device = torch.device(args.gpu)
        # disable printing when not in master process
        __builtin__.print = print_mod
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
                        enableGlobalMultiHeadAttention=args.enableGlobalMultiHeadAttention,
                        enableTemporalMultiHeadAttention=args.enableTemporalMultiHeadAttention,
                        enableSpatialTemporalTransformerEncoder=args.enableSpatialTemporalTransformerEncoder,
                        numLayerTransformerEncoder=args.numLayerTransformerEncoder,
                        numHeadMultiHeadAttention=args.numHeadMultiHeadAttention,
                        loss_weights=loss_weights)
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
    optimizer = torch.optim.AdamW(params=model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    if args.enable_cudaAMP:
        # Creates GradScaler for CUDA AMP
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Trainer
    class_trainer = Trainer(net=model,
                            class_weights=torch.Tensor(loss_weights).to(device),
                            optim=optimizer,
                            enable_multibranch_ffr=args.enable_multibranch_ffr,
                            enable_multibranch_ifr=args.enable_multibranch_ifr,
                            multibranch_loss_weight=args.multibranch_loss_weight,
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
                    
                    images_3d = batch['image_3d']
                    
                    tot_true_labels.extend(labels.tolist())
                    tot_image_paths.extend(image_paths)
                    tot_true_FFRs.extend(FFRs.tolist())
                    tot_true_iFRs.extend(iFRs.tolist())

                    images_3d = images_3d.to(device)

                    labels = labels.to(device)
                    FFRs = FFRs.to(device)
                    iFRs = iFRs.to(device)

                    returned_values = class_trainer.forward_batch(images_3d, labels, FFRs, iFRs, split)
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
                    rocCurve_image = utils.calc_rocCurve(tot_true_labels, tot_predicted_scores)
                    saver.log_images("Classifier Epoch "+split+"/"+"ROC Curve", rocCurve_image, epoch, split, "ROCcurve", args.save_image_file)

                    # Precision-Recall Curve
                    precisionRecallCurve_image = utils.calc_precisionRecallCurve(tot_true_labels, tot_predicted_scores)
                    saver.log_images("Classifier Epoch "+split+"/"+"Precision-Recall Curve", precisionRecallCurve_image, epoch, split, "PrecisionRecallCurve", args.save_image_file)
                    
                    # Prediction Agreement Rate: same-sample evaluation agreement between current and previous epoch
                    predictionAgreementRate, tot_predicted_labels_last[split] = utils.calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last[split], tot_image_paths)
                    saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Prediction Agreement Rate", predictionAgreementRate, epoch)
        
                    # Confusion Matrix
                    cm_image = utils.plot_confusion_matrix(tot_true_labels, tot_predicted_labels, ['negative', 'positive'], title="Confusion matrix "+split)
                    saver.log_images("Classifier Epoch "+split+"/"+"Confusion Matrix", cm_image, epoch, split, "ConfMat", args.save_image_file)

                    # Histograms FFR/iFR of FN and FP
                    histFFRs_image = utils.calc_FN_FP_histograms(tot_true_labels, tot_predicted_labels, tot_true_FFRs, "FFR", split)
                    saver.log_images("Classifier Epoch "+split+"/"+"Histogram FFRs", histFFRs_image, epoch, split, "HistFFRs", args.save_image_file)
                    histiFRs_image = utils.calc_FN_FP_histograms(tot_true_labels, tot_predicted_labels, tot_true_iFRs, "iFR", split)
                    saver.log_images("Classifier Epoch "+split+"/"+"Histogram iFRs", histiFRs_image, epoch, split, "HistiFRs", args.save_image_file)
                    
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
                        predictionError_FFRs = utils.calc_predictionError_histograms(tot_true_FFRs, tot_predicted_FFRs, args.multibranch_error_regression_threshold, "FFR", split)
                        saver.log_images("Classifier Epoch "+split+"/"+"Prediction error FFRs", predictionError_FFRs, epoch, split, "HistPredErrorFFRs", args.save_image_file)
                    
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
                        predictionError_iFRs = utils.calc_predictionError_histograms(tot_true_iFRs, tot_predicted_iFRs, args.multibranch_error_regression_threshold, "iFR", split)
                        saver.log_images("Classifier Epoch "+split+"/"+"Prediction error iFRs", predictionError_iFRs, epoch, split, "HistPredErroriFRs", args.save_image_file)

                    # Error prediction per hospital
                    histPredictionErrorHospital = utils.calc_predictionErrorHospital_histogram(tot_true_labels, tot_predicted_labels, tot_image_paths, split)
                    saver.log_images("Classifier Epoch "+split+"/"+"Histogram prediction error hospital", histPredictionErrorHospital, epoch, split, "HistPredErrHospital", args.save_image_file)
                    # Percentual error prediction per hospital
                    histPercentualPredictionErrorHospital = utils.calc_predictionPercentualErrorHospital_histogram(tot_true_labels, tot_predicted_labels, tot_image_paths, split)
                    saver.log_images("Classifier Epoch "+split+"/"+"Histogram percentual prediction error hospital", histPercentualPredictionErrorHospital, epoch, split, "HistPercPredErrHospital", args.save_image_file)

                    # Save logs of error
                    dict_other_info = {'image_path':tot_image_paths, 'ffr':tot_true_FFRs, 'ifr':tot_true_iFRs}
                    if args.enable_multibranch_ffr:
                        dict_other_info['ffr_predicted'] = tot_predicted_FFRs
                    if args.enable_multibranch_ifr:
                        dict_other_info['ifr_predicted'] = tot_predicted_iFRs
                    saver.saveLogsError(tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
                
                    # Save logs
                    if args.saveLogs:
                        Saver.saveLogs(args.logdir, tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
                        
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