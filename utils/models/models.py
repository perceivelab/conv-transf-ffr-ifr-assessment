'''
Copyright (c) R. Mineo, 2022-2024. All rights reserved.
This code was developed by R. Mineo in collaboration with PerceiveLab and other contributors.
For usage and licensing requests, please contact the owner.
'''

import torch
import torchvision
from torch import nn
from collections import OrderedDict

from utils.models import modules

from utils.models import i3d, s3d #, Angio3DNet

from utils.models import lesion_classification, directQuantificationAttention, directQuantification, mvcnn, gvcnn, vit, fullLeftVentricleQuantification, swin_transformer

def get_model(num_classes,
              model_name,
              enable_multibranch_ffr, enable_multibranch_ifr, multibranch_dropout,
              enable_clinicalData, in_dim_clinicalData,
              enable_doubleView,
              enable_keyframe,
              reduceInChannel,
              freezeBackbone,
              enableNonLocalBlock, enableTemporalNonLocalBlock, enableSpatioTemporalNonLocalBlock, numNonLocalBlock,
              enableGlobalMultiHeadAttention, enableTemporalMultiHeadAttention, numHeadMultiHeadAttention,
              enableTemporalGru, numLayerGru, enableTemporalLstm, numLayerLstm,
              enableGlobalTransformerEncoder, enableTemporalTransformerEncoder, enableSpatialTemporalTransformerEncoder, numLayerTransformerEncoder, numHeadGlobalTransformer, numHeadSpatialTransformer, numHeadTemporalTransformer, transformerNormFirst,
              loss_weights,
              batch_size,
              input_size): # [time, height, width]
    ''' returns the classifier '''

    enable_mod_ffr_ifr = False
    if enable_clinicalData or enable_doubleView or enable_keyframe:
        if enable_clinicalData and (in_dim_clinicalData is None):
            raise RuntimeError("in_dim_clinicalData must be specified.")
        enable_mod_ffr_ifr = True
        fusion_dim = 256
        num_classes_in = num_classes
        num_classes = fusion_dim
    
    if (model_name == 'ResNet3D') or (model_name == 'ResNet3D_pretrained'):
        model = torchvision.models.video.r3d_18(pretrained=(model_name=='ResNet3D_pretrained'), progress=True)
        model.avgpool = modules.RaiseExceptionShape()
        output_shape_batch, output_shape_channel, output_shape_time, output_shape_height, output_shape_width = modules.calculateOutputSize(input_size, model) # torch.Size([batch, channel, time, height, width]) ex. input BxCx60x256x256 - output Bx512x8x16x16
        if freezeBackbone:
            if not "pretrained" in model_name:
                raise RuntimeError("Freeze backbone is available only in pretrained models.")
            for _, param in model.named_parameters():
                param.requires_grad = False
        if enableNonLocalBlock:
            nl_blocks = []
            for _ in range(numNonLocalBlock):
                nl_blocks.append(modules.NonLocalBlock(in_channels=output_shape_channel, dimension=3, inter_channels=output_shape_channel//2, sub_sample=False, bn_layer=True))
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Non-Local Blocks
                *nl_blocks,
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableTemporalNonLocalBlock:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(1,1,1)), # torch.Size([batch, channel/4, time, height, width])
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(1,1,1)), # torch.Size([batch, channel/4/8, time, height, width])
                # Temporal Non Local Block
                modules.TemporalNonLocalBlock(in_channels=output_shape_channel//4//8, inter_channels=output_shape_channel//4//8//2, numblocks=numNonLocalBlock), # torch.Size([batch, channel/4/8, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2),# torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableSpatioTemporalNonLocalBlock:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Spatial Non local block
                modules.SpatialNonLocalBlock(numblocks=numNonLocalBlock, in_channels=output_shape_channel, inter_channels=output_shape_channel//2), # torch.Size([batch, channel, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(1,1,1)), # torch.Size([batch, channel/4, time, height, width])
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(1,1,1)), # torch.Size([batch, channel/4/8, time, height, width])
                # Temporal Non Local Block
                modules.TemporalNonLocalBlock(numblocks=numNonLocalBlock, in_channels=output_shape_channel//4//8, inter_channels=output_shape_channel//4//8//2), # torch.Size([batch, channel/4/8, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableGlobalMultiHeadAttention:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Global attention
                modules.GlobalMultiHeadAttention(embed_dim=output_shape_channel, num_heads=numHeadMultiHeadAttention), # torch.Size([batch, channel, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4, time/2, height/2, width/2]),
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableTemporalMultiHeadAttention:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(1,1,1)), # torch.Size([batch, channel/4, time, height, width])
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(1,1,1)), # torch.Size([batch, channel/4/8, time, height, width])
                # Temporal attention
                modules.TemporalMultiHeadAttention(embed_dim=(output_shape_channel//4//8) * output_shape_height * output_shape_width, num_heads=numHeadMultiHeadAttention), # torch.Size([batch, channel/4/8, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableTemporalGru:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(1,1,1)), # torch.Size([batch, channel/4, time, height, width])
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(1,1,1)), # torch.Size([batch, channel/4/8, time, height, width])
                # Temporal GRU
                modules.TemporalGru(input_size=(output_shape_channel//4//8) * output_shape_height * output_shape_width, hidden_size=(output_shape_channel//4//8) * output_shape_height * output_shape_width, num_layers=numLayerGru, batch_size=batch_size), # torch.Size([batch, channel/4/8, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableTemporalLstm:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(1,1,1)), # torch.Size([batch, channel/4, time, height, width])
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(1,1,1)), # torch.Size([batch, channel/4/8, time, height, width])
                # Temporal LSTM
                modules.TemporalLSTM(input_size=(output_shape_channel//4//8) * output_shape_height * output_shape_width, hidden_size=(output_shape_channel//4//8) * output_shape_height * output_shape_width, num_layers=numLayerLstm, batch_size=batch_size), # torch.Size([batch, channel/4/8, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableGlobalTransformerEncoder:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Global transformer
                modules.GlobalTransformer(d_model=output_shape_channel, num_heads=numHeadGlobalTransformer, norm_first=transformerNormFirst, num_layers=numLayerTransformerEncoder), # torch.Size([batch, channel, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableTemporalTransformerEncoder:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(1,1,1)), # torch.Size([batch, channel/4, time, height, width])
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(1,1,1)), # torch.Size([batch, channel/4/8, time, height, width])
                # Temporal attention
                modules.TemporalTransformer(d_model=(output_shape_channel//4//8) * output_shape_height * output_shape_width, num_heads=numHeadTemporalTransformer, norm_first=transformerNormFirst, num_layers=numLayerTransformerEncoder), # torch.Size([batch, channel/4/8, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        elif enableSpatialTemporalTransformerEncoder:
            model.avgpool = nn.Sequential( # torch.Size([batch, channel, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel, out_channels=output_shape_channel//4, kernel_size=(1,1,1)), # torch.Size([batch, channel/4, time, height, width])
                # Spatial Transformer
                modules.SpatialTransformer(d_model=output_shape_channel//4, num_heads=numHeadSpatialTransformer, norm_first=transformerNormFirst, num_layers=numLayerTransformerEncoder), # torch.Size([batch, channel/4, time, height, width])
                # Reduce feature
                nn.Conv3d(in_channels=output_shape_channel//4, out_channels=output_shape_channel//4//8, kernel_size=(1,1,1)), # torch.Size([batch, channel/4/8, time, height, width])
                # Temporal Transformer
                modules.TemporalTransformer(d_model=(output_shape_channel//4//8) * output_shape_height * output_shape_width, num_heads=numHeadTemporalTransformer, norm_first=transformerNormFirst, num_layers=numLayerTransformerEncoder), # torch.Size([batch, channel/4/8, time, height, width])
                # Reduce dims
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2, height/2, width/2])
                nn.ReLU(),
                nn.Conv3d(in_channels=output_shape_channel//4//8, out_channels=output_shape_channel//4//8, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)), # torch.Size([batch, channel/4/8, time/2/2, height/2/2, width/2/2])
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2), # torch.Size([batch, channel/4/8 * time/2/2 * height/2/2 * width/2/2])
            )
            output_feature_dim = (output_shape_channel//4//8) * (output_shape_time//2//2) * (output_shape_height//2//2) * (output_shape_width//2//2)
        else:
            output_feature_dim = 512
        model.fc = modules.LastLayer(enable_multibranch_ffr,enable_multibranch_ifr,('linear', 'linear'),output_feature_dim,{'label':num_classes, 'FFR':num_classes if enable_mod_ffr_ifr else 1, 'iFR':num_classes if enable_mod_ffr_ifr else 1},multibranch_dropout)
    else:
        raise RuntimeError('Model name not found!')

    if reduceInChannel:
        if (model_name == 'ResNet3D') or (model_name == 'ResNet3D_pretrained'):
            modules.reduceInputChannel(model)
        else:
            raise NotImplementedError("reduceInputChannel not implemented in chosen model.")

    if enable_clinicalData or enable_doubleView or enable_keyframe:
        model_additional = None
        if enable_keyframe:
            model_additional = torchvision.models.resnext50_32x4d(pretrained=True, progress=True)
            if reduceInChannel:
                weight2 = model_additional.conv1.weight.clone()
                model_additional.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    model_additional.conv1.weight = nn.Parameter(weight2.mean(dim=1).unsqueeze(1))
            model_additional.fc = modules.LastLayer(enable_multibranch_ffr,enable_multibranch_ifr,('linear', 'linear'),False,0,512*4,{'label':num_classes, 'FFR':num_classes if enable_mod_ffr_ifr else 1, 'iFR':num_classes if enable_mod_ffr_ifr else 1},multibranch_dropout)
        
        model = modules.doubleView_clinicalData_keyframe_LateFusion(enable_multibranch_ffr=enable_multibranch_ffr, enable_multibranch_ifr=enable_multibranch_ifr, enable_doubleView=enable_doubleView, enable_clinicalData=enable_clinicalData, enable_keyframe=enable_keyframe, net_img_3d=model, net_img_2d=model_additional, fusion_dim=fusion_dim, in_dim_clinicalData=in_dim_clinicalData, num_classes=num_classes_in)

    return model