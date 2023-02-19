import torch 
import torch.nn as nn
import torch.nn.functional as F
import typing


class Reshape(nn.Module):
    def __init__(self, output_num_dims):
        super(Reshape, self).__init__()
        self.output_num_dims = output_num_dims
        
    def forward(self,io):
        if (self.output_num_dims == 4):
            io = io.reshape(io.shape[0],io.shape[1],io.shape[2], -1)
        elif (self.output_num_dims == 3):
            io = io.reshape(io.shape[0],io.shape[1], -1)
        elif (self.output_num_dims == 2):
            io = io.reshape(io.shape[0], -1)
        elif (self.output_num_dims == 1):
            io = io.reshape(-1)
        else:
            raise RuntimeError("Reshape module accept only output_num_dims 1/2/3/4")
        return io


class Permute(nn.Module):
    def __init__(self, dims_list : typing.Union[tuple, list]):
        self.dims_list = dims_list
        super(Permute, self).__init__()
        
    def forward(self,i):
        o = i.permute(self.dims_list)
        return o


class PrintInShape(nn.Module):
    def __init__(self, stop=True):
        self.stop = stop
        super(PrintInShape, self).__init__()
        
    def forward(self,io):
        print(io.shape)
        if self.stop:
            raise RuntimeError("Stop to view shape.")
        return io


class ClassificationLayer(nn.Module):
    def __init__(self, type, in_dim, out_dim, enable_batchnorm=False, dropout=0.5):
        super(ClassificationLayer, self).__init__()
        self.enable_batchnorm = enable_batchnorm

        if type[0] == 'linear':
            self.fc1 = nn.Linear(in_features=in_dim, out_features=in_dim//2)
        elif type[0] == 'convolutional':
            self.fc1 = nn.Conv3d(in_dim, in_dim//2, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            raise NotImplementedError('You insert ' + type + ' as ClassificationLayer type, but only linear and convolutional type are implemented.')
        
        if self.enable_batchnorm:
            self.bn = nn.BatchNorm1d(num_features=in_dim//2)
        self.a = nn.ReLU()
        self.do = nn.Dropout(p=dropout)

        if type[1] == 'linear':
            self.fc2 = nn.Linear(in_features=in_dim//2, out_features=out_dim)
        elif type[1] == 'convolutional':
            self.fc2 = nn.Conv3d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            raise NotImplementedError('You insert ' + type + ' as ClassificationLayer type, but only linear and convolutional type are implemented.')
        
    def forward(self,i):
        tmp = self.fc1(i)
        if self.enable_batchnorm:
            tmp = self.bn(tmp)
        tmp = self.a(tmp)
        tmp = self.do(tmp)
        o = self.fc2(tmp)
        return o


class MultiheadAttentionMod(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super(MultiheadAttentionMod, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
    
    def forward(self,i):
        o,_ = self.attention(i,i,i)
        return o


class GlobalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, unflatten):
        super(GlobalMultiHeadAttention, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Global Attention
            nn.Flatten(1,3), # torch.Size([batch, time*height*width, channel])
            MultiheadAttentionMod(embed_dim=embed_dim, num_heads=num_heads, batch_first=True), # torch.Size([batch, time*height*width, channel])
            nn.Unflatten(1,unflatten), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, unflatten):
        super(TemporalMultiHeadAttention, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Temporal Attention
            nn.Flatten(2,4), # torch.Size([batch, time, height*width*channel])
            MultiheadAttentionMod(embed_dim=embed_dim, num_heads=num_heads, batch_first=True), # torch.Size([batch, time, height*width*channel])
            nn.Unflatten(2,unflatten), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, unflatten1, unflatten0):
        super(SpatialMultiHeadAttention, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Spatial Attention
            nn.Flatten(2,3), # torch.Size([batch, time, height*width, channel])
            nn.Flatten(0,1), # torch.Size([batch*time, height*width, channel])
            MultiheadAttentionMod(embed_dim=embed_dim, num_heads=num_heads, batch_first=True), # torch.Size([batch*time, height*width, channel])
            nn.Unflatten(1,unflatten1), # torch.Size([batch*time, height, width, channel])
            nn.Unflatten(0,unflatten0), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, unflatten):
        super(TemporalTransformer, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Temporal Attention
            nn.Flatten(2,4), # torch.Size([batch, time, height*width*channel])
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=num_heads*64, dropout=0.1, batch_first=True),
                                    num_layers=num_layers), # torch.Size([batch, time, height*width*channel])
            nn.Unflatten(2,unflatten), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class SpatialTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, unflatten1, unflatten0):
        super(SpatialTransformer, self).__init__()
        self.net = nn.Sequential( # torch.Size([batch, channel, time, height, width])
            # Move feature to last dim
            Permute((0,2,3,4,1)), # torch.Size([batch, time, height, width, channel])
            # Temporal Attention
            nn.Flatten(2,3), # torch.Size([batch, time, height*width, channel])
            nn.Flatten(0,1), # torch.Size([batch*time, height*width, channel])
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=num_heads*64, dropout=0.1, batch_first=True),
                                    num_layers=num_layers), # torch.Size([batch*time, height*width, channel])
            nn.Unflatten(1,unflatten1), # torch.Size([batch*time, height, width, channel])
            nn.Unflatten(0,unflatten0), # torch.Size([batch, time, height, width, channel])
            # Move feature to 2nd dim
            Permute((0,4,1,2,3)), # torch.Size([batch, channel, time, height, width])
        )
    
    def forward(self,i):
        o = self.net(i)
        return o


class LastLayer(nn.Module):
    def __init__(self, enable_multibranch_ffr, enable_multibranch_ifr, type, in_dim, out_dim={'label':2, 'FFR':1, 'iFR':1}, dropout={'label_dropout':0.5, 'FFR_dropout':0.5, 'iFR_dropout':0.5}):
        super(LastLayer, self).__init__()
        self.enable_multibranch_ffr = enable_multibranch_ffr
        self.enable_multibranch_ifr = enable_multibranch_ifr

        self.labelClassification = ClassificationLayer(type=type,in_dim=in_dim,out_dim=out_dim['label'],dropout=dropout['label_dropout'])
        if self.enable_multibranch_ffr:
            self.ffrRegression = ClassificationLayer(type=type,in_dim=in_dim,out_dim=out_dim['FFR'],dropout=dropout['FFR_dropout'])
        if self.enable_multibranch_ifr:
            self.ifrRegression = ClassificationLayer(type=type,in_dim=in_dim,out_dim=out_dim['iFR'],dropout=dropout['iFR_dropout'])
        
    def forward(self,i):
        o_label = self.labelClassification(i)
        if self.enable_multibranch_ffr:
            o_ffr = self.ffrRegression(i)
        if self.enable_multibranch_ifr:
            o_ifr = self.ifrRegression(i)
        
        if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
            return o_label, o_ffr, o_ifr
        elif self.enable_multibranch_ffr:
            return o_label, o_ffr
        elif self.enable_multibranch_ifr:
            return o_label, o_ifr
        else:
            return o_label