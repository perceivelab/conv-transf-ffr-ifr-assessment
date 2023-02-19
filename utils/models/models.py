import torchvision
from torch import nn

from utils.models import modules

def get_model(num_classes, model_name, enable_multibranch_ffr, enable_multibranch_ifr, multibranch_dropout, enableGlobalMultiHeadAttention, enableTemporalMultiHeadAttention, enableSpatialTemporalTransformerEncoder, numLayerTransformerEncoder, numHeadMultiHeadAttention, loss_weights):
    ''' returns the classifier '''
    
    if model_name == 'resnet3d_pretrained':
        model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        if enableGlobalMultiHeadAttention:
            model.avgpool = nn.Sequential(
                # Global attention
                modules.GlobalMultiHeadAttention(embed_dim=512, num_heads=numHeadMultiHeadAttention, unflatten=(8,16,16)),
                # Reduce dims
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2),
            )
            output_feature_dim = 512
        elif enableTemporalMultiHeadAttention:
            model.avgpool = nn.Sequential(
                # Reduce feature
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(1,1,1)),
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(1,1,1)),
                # Temporal attention
                modules.TemporalMultiHeadAttention(embed_dim=16*16*16, num_heads=numHeadMultiHeadAttention, unflatten=(16,16,16)),
                # Reduce dims
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.ReLU(),
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2),
            )
            output_feature_dim = 512
        elif enableSpatialTemporalTransformerEncoder:
            model.avgpool = nn.Sequential(
                # Reduce feature
                nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(1,1,1)),
                # Spatial Transformer
                modules.SpatialTransformer(d_model=128, num_heads=numHeadMultiHeadAttention, num_layers=numLayerTransformerEncoder, unflatten1=(16,16), unflatten0=(-1,8)),
                # Reduce feature
                nn.Conv3d(in_channels=128, out_channels=16, kernel_size=(1,1,1)),
                # Temporal Transformer
                modules.TemporalTransformer(d_model=16*16*16, num_heads=numHeadMultiHeadAttention, num_layers=numLayerTransformerEncoder, unflatten=(16,16,16)),
                # Reduce dims
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.ReLU(),
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.ReLU(),
                # Linearize dims
                modules.Reshape(2),
            )
            output_feature_dim = 512
        else:
            output_feature_dim = 512
        model.fc = modules.LastLayer(enable_multibranch_ffr,enable_multibranch_ifr,('linear', 'linear'),output_feature_dim,{'label':num_classes, 'FFR':1, 'iFR':1},multibranch_dropout)
    else:
        raise RuntimeError('Model name not found!')
    
    return model