import torch 
import torch.nn as nn
import torch.nn.functional as F
import typing


class ShapeException(Exception):
    def __init__(self, shape_to_pass):
        self.shape_to_pass = shape_to_pass

    def __str__(self):
        return repr(self.shape_to_pass)
    
    def getShape(self):
        return self.shape_to_pass

class RaiseExceptionShape(nn.Module):
    def __init__(self):
        super(RaiseExceptionShape, self).__init__()
        
    def forward(self,input):
        raise ShapeException(input.shape)

def calculateOutputSize(input_dim : list, model : nn.Module) -> torch.Size:
    try:
        model(torch.rand(1, 3, *input_dim))
    except ShapeException as se:
        return se.getShape()

    raise SyntaxError("The model must raise ShapeException")


class Unsqueeze(nn.Module):
    def __init__(self, dim_list):
        super(Unsqueeze, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        for dim in self.dim_list:
            io = io.unsqueeze(dim)
        return io


class Squeeze(nn.Module):
    def __init__(self, dim_list):
        super(Squeeze, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        for dim in self.dim_list:
            io = io.squeeze(dim)
        return io


class View(nn.Module):
    def __init__(self, output_num_dims):
        super(View, self).__init__()
        self.output_num_dims = output_num_dims
        
    def forward(self,io):
        if (self.output_num_dims == 4):
            io = io.view(io.shape[0],io.shape[1],io.shape[2], -1)
        elif (self.output_num_dims == 3):
            io = io.view(io.shape[0],io.shape[1], -1)
        elif (self.output_num_dims == 2):
            io = io.view(io.shape[0], -1)
        elif (self.output_num_dims == 1):
            io = io.view(-1)
        else:
            raise RuntimeError("View module accept only output_num_dims 1/2/3/4")
        return io


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


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self,i):
        o = F.interpolate(i, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        return o


class Squeeze_3out(nn.Module):
    def __init__(self, dim_list):
        super(Squeeze_3out, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        io1, io2, io3 = io
        for dim in self.dim_list:
            io1 = io1.squeeze(dim)
            io2 = io2.squeeze(dim)
            io3 = io3.squeeze(dim)
        return io1, io2, io3


class Interpolate_3out(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate_3out, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self,i):
        i1, i2, i3 = i
        o1 = F.interpolate(i1, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        o2 = F.interpolate(i2, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        o3 = F.interpolate(i3, self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        return o1,o2,o3


class Pad3D(nn.Module):
    def __init__(self, kernel_shape=(1, 1, 1), stride=(1, 1, 1)):
        super(Pad3D, self).__init__()
        self._kernel_shape = kernel_shape
        self._stride = stride

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        return x


class Mean_3out(nn.Module):
    def __init__(self):
        super(Mean_3out, self).__init__()
        
    def forward(self,i):
        i1, i2, i3 = i

        o1 = torch.mean(i1.view(i1.size(0), i1.size(1), i1.size(2)), 2)
        o2 = torch.mean(i2.view(i2.size(0), i2.size(1), i2.size(2)), 2)
        o3 = torch.mean(i3.view(i3.size(0), i3.size(1), i3.size(2)), 2)

        return o1,o2,o3


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
    
    
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, dimension, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3] # 1D, 2D, 3D

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z
    

class SpatialNonLocalBlock(nn.Module):
    def __init__(self, numblocks, in_channels, inter_channels, sub_sample, bn_layer):
        super(SpatialNonLocalBlock, self).__init__()
        nl_blocks = []
        for _ in range(numblocks):
            nl_blocks.append(NonLocalBlock(in_channels=in_channels, dimension=1, inter_channels=inter_channels, sub_sample=sub_sample, bn_layer=bn_layer))
        self.nl_blocks = nn.Sequential(*nl_blocks)

    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature after time dim
        i = i.permute((0,2,1,3,4)) # torch.Size([batch, time, channel, height, width])

        # Flatten dims 3-4 e 0-1
        o = i.flatten(3,4) # torch.Size([batch, time, channel, height*width])
        o = o.flatten(0,1) # torch.Size([batch*time, channel, height*width])

        # Spatial Non Local Block
        o = self.nl_blocks(o) # torch.Size([batch*time, channel, height*width])

        # Unflatten dims 3-4 e 0-1
        o = o.unflatten(2, i.shape[3:]) # torch.Size([batch*time, channel, height, width])
        o = o.unflatten(0, i.shape[0:2]) # torch.Size([batch, time, channel, height, width])

        # Move feature to 2nd dim
        o = o.permute((0,2,1,3,4)) # torch.Size([batch, channel, time, height, width])

        return o

class TemporalNonLocalBlock(nn.Module):
    def __init__(self, numblocks, in_channels, inter_channels, sub_sample, bn_layer):
        super(TemporalNonLocalBlock, self).__init__()
        nl_blocks = []
        for _ in range(numblocks):
            nl_blocks.append(NonLocalBlock(in_channels=in_channels, dimension=1, inter_channels=inter_channels, sub_sample=sub_sample, bn_layer=bn_layer))
        self.nl_blocks = nn.Sequential(*nl_blocks)

    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature after time dim
        i = i.permute((0,2,1,3,4)) # torch.Size([batch, time, channel, height, width])

        # Flatten dims 2-4
        o = i.flatten(2,4) # torch.Size([batch, time, height*width*channel])

        # Temporal Non Local Block
        o = self.nl_blocks(o)  # torch.Size([batch, time, height*width*channel])

        # Unflatten dims 2-4
        o = o.unflatten(2, i.shape[2:5]) # torch.Size([batch, time, channel, height, width])

        # Move feature to 2nd dim
        o = o.permute((0,2,1,3,4)) # torch.Size([batch, channel, time, height, width])

        return o


class MultiheadAttentionMod(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super(MultiheadAttentionMod, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
    
    def forward(self,i):
        o,_ = self.attention(i,i,i)
        return o


class GlobalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GlobalMultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])
        
        # Flatten dims 1-3
        o = i.flatten(1,3) # torch.Size([batch, time*height*width, channel])
        
        # Global Attention
        o,_ = self.mha(o,o,o) # torch.Size([batch, time*height*width, channel])
        
        # Unflatten dims 1-3
        o = o.unflatten(1, i.shape[1:4]) # torch.Size([batch, time, height, width, channel])
        
        # Move feature to 2nd dim
        o = i.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])
        return o


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TemporalMultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True) # torch.Size([batch, time, height*width*channel])
    
    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])
        
        # Flatten dims 2-4
        o = i.flatten(2,4) # torch.Size([batch, time, height*width*channel])
        
        # Temporal Attention
        o,_ = self.mha(o,o,o) # torch.Size([batch, time, height*width*channel])
        
        # Unflatten dims 2-4
        o = o.unflatten(2, i.shape[2:5]) # torch.Size([batch, time, height, width, channel])
        
        # Move feature to 2nd dim
        o = o.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])
        
        return o


class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SpatialMultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])
        
        # Flatten dims 2-3 e 0-1
        o = i.flatten(2,3) # torch.Size([batch, time, height*width, channel])
        o = o.flatten(0,1) # torch.Size([batch*time, height*width, channel])
        
        # Spatial Attention
        o,_ = self.mha(o,o,o) # torch.Size([batch*time, height*width, channel])
        
        # Unflatten dims 2-3 e 0-1
        o = o.unflatten(1, i.shape[2:4]) # torch.Size([batch*time, height, width, channel])
        o = o.unflatten(0, i.shape[0:2]) # torch.Size([batch, time, height, width, channel])
        
        # Move feature to 2nd dim
        o = o.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])

        return o
    

class TemporalGru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(TemporalGru, self).__init__()
        self.h0 = torch.nn.Parameter(torch.zeros(num_layers, batch_size, hidden_size))
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])

        # Flatten dims 2-4
        o = i.flatten(2,4) # torch.Size([batch, time, height*width*channel])

        # solve problem if dataloader has drop_last=False
        if o.shape[0] != self.h0.shape[1]:
            o = o.repeat(torch.ceil(torch.tensor(self.h0.shape[1]/o.shape[0])).int().item(), 1, 1)[:self.h0.shape[1]]
            if len(o.shape) == 2:
                o = o.unsqueeze(0)

        # Temporal GRU
        o,_ = self.gru(o, self.h0) # torch.Size([batch, time, height*width*channel])

        # solve problem if dataloader has drop_last=False
        if i.shape[0] != self.h0.shape[1]:
            o = o[:i.shape[0]]
            if len(o.shape) == 2:
                o = o.unsqueeze(0)

        # Unflatten dims 2-4
        o = o.unflatten(2, i.shape[2:5]) # torch.Size([batch, time, height, width, channel])

        # Move feature to 2nd dim
        o = o.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])

        return o


class TemporalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(TemporalLSTM, self).__init__()
        self.h0 = torch.nn.Parameter(torch.zeros(num_layers, batch_size, hidden_size))
        self.c0 = torch.nn.Parameter(torch.zeros(num_layers, batch_size, hidden_size))
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])
        
        # Flatten dims 2-4
        o = i.flatten(2,4) # torch.Size([batch, time, height*width*channel])
        
        # solve problem if dataloader has drop_last=False
        if o.shape[0] != self.h0.shape[1]:
            o = o.repeat(torch.ceil(torch.tensor(self.h0.shape[1]/o.shape[0])).int().item(), 1, 1)[:self.h0.shape[1]]
            if len(o.shape) == 2:
                o = o.unsqueeze(0)
        
        # Temporal LSTM
        o,_ = self.lstm(o, (self.h0,self.c0)) # torch.Size([batch, time, height*width*channel])
        
        # solve problem if dataloader has drop_last=False
        if i.shape[0] != self.h0.shape[1]:
            o = o[:i.shape[0]]
            if len(o.shape) == 2:
                o = o.unsqueeze(0)
        
        # Unflatten dims 2-4
        o = o.unflatten(2, i.shape[2:5]) # torch.Size([batch, time, height, width, channel])
        
        # Move feature to 2nd dim
        o = o.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])
        
        return o


class GlobalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, norm_first, num_layers):
        super(GlobalTransformer, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=num_heads*64, dropout=0.1, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    
    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])
        
        # Flatten dims 1-3
        o = i.flatten(1,3) # torch.Size([batch, time*height*width, channel])
        
        # Global Attention
        o = self.encoder(o) # torch.Size([batch, time*height*width, channel])
        
        # Unflatten dims 1-3
        o = o.unflatten(1, i.shape[1:4]) # torch.Size([batch, time, height, width, channel])
        
        # Move feature to 2nd dim
        o = i.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])
        return o


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, norm_first, num_layers):
        super(TemporalTransformer, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=num_heads*64, dropout=0.1, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    
    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])
        
        # Flatten dims 2-4
        o = i.flatten(2,4) # torch.Size([batch, time, height*width*channel])
        
        # Temporal Attention
        o = self.encoder(o)  # torch.Size([batch, time, height*width*channel])
        
        # Unflatten dims 2-4
        o = o.unflatten(2, i.shape[2:5]) # torch.Size([batch, time, height, width, channel])
        
        # Move feature to 2nd dim
        o = o.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])
        
        return o


class SpatialTransformer(nn.Module):
    def __init__(self, d_model, num_heads, norm_first, num_layers):
        super(SpatialTransformer, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=num_heads*64, dropout=0.1, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    
    def forward(self,i): # torch.Size([batch, channel, time, height, width])
        # Move feature to last dim
        i = i.permute((0,2,3,4,1)) # torch.Size([batch, time, height, width, channel])
        
        # Flatten dims 2-3 e 0-1
        o = i.flatten(2,3) # torch.Size([batch, time, height*width, channel])
        o = o.flatten(0,1) # torch.Size([batch*time, height*width, channel])
        
        # Spatial Attention
        o = self.encoder(o) # torch.Size([batch*time, height*width, channel])
        
        # Unflatten dims 2-3 e 0-1
        o = o.unflatten(1, i.shape[2:4]) # torch.Size([batch*time, height, width, channel])
        o = o.unflatten(0, i.shape[0:2]) # torch.Size([batch, time, height, width, channel])
        
        # Move feature to 2nd dim
        o = o.permute((0,4,1,2,3)) # torch.Size([batch, channel, time, height, width])

        return o


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


class doubleView_clinicalData_keyframe_LateFusion(nn.Module):
    def __init__(self, enable_multibranch_ffr, enable_multibranch_ifr, enable_doubleView, enable_clinicalData, enable_keyframe, net_img_3d, net_img_2d, fusion_dim, in_dim_clinicalData, num_classes):
        super(doubleView_clinicalData_keyframe_LateFusion, self).__init__()
        self.enable_multibranch_ffr = enable_multibranch_ffr
        self.enable_multibranch_ifr = enable_multibranch_ifr
        self.enable_doubleView = enable_doubleView
        self.enable_clinicalData = enable_clinicalData
        self.enable_keyframe = enable_keyframe
        #self.enable_batchnorm = enable_batchnorm

        self.net_img_3d = net_img_3d

        if self.enable_keyframe:
            self.net_img_2d = net_img_2d

        if self.enable_clinicalData:
            self.fc_clinicalData = nn.Linear(in_dim_clinicalData, fusion_dim)
            #if self.enable_batchnorm:
            #   self.bn = nn.BatchNorm1d(fusion_dim)

        if self.net_img_3d is not None:
            self.n = 1
        else:
            self.n = 0
        if self.enable_doubleView:
            self.n += 1
        if self.enable_clinicalData:
            self.n += 1
        if self.enable_keyframe:
            self.n += 1
            if self.enable_doubleView:
                self.n += 1

        self.classifier_label = nn.Linear(self.n*fusion_dim, self.n*fusion_dim//2)
        self.classifier2_label = nn.Linear(self.n*fusion_dim//2, num_classes)
        if self.enable_multibranch_ffr:
            self.classifier_ffr = nn.Linear(self.n*fusion_dim, self.n*fusion_dim//2)
            self.classifier2_ffr = nn.Linear(self.n*fusion_dim//2, 1)
        if self.enable_multibranch_ifr:
            self.classifier_ifr = nn.Linear(self.n*fusion_dim, self.n*fusion_dim//2)
            self.classifier2_ifr = nn.Linear(self.n*fusion_dim//2, 1)
        
    def forward(self, inputs):
        if self.enable_doubleView:
            if self.enable_clinicalData:
                if self.enable_keyframe:
                    if self.net_img_3d is not None:
                        imgs_3d, doubleView_3d, clinicalData, imgs_2d, doubleView_2d = inputs
                    else:
                        raise RuntimeError("Can't use doubleView without 3D input branch")
                else:
                    if self.net_img_3d is not None:
                        imgs_3d, doubleView_3d, clinicalData = inputs
                    else:
                        raise RuntimeError("Can't use doubleView without 3D input branch")
            else:
                if self.enable_keyframe:
                    if self.net_img_3d is not None:
                        imgs_3d, doubleView_3d, imgs_2d, doubleView_2d = inputs
                    else:
                        raise RuntimeError("Can't use doubleView without 3D input branch")
                else:
                    if self.net_img_3d is not None:
                        imgs_3d, doubleView_3d = inputs
                    else:
                        raise RuntimeError("Can't use doubleView without 3D input branch")
        else:
            if self.enable_clinicalData:
                if self.enable_keyframe:
                    if self.net_img_3d is not None:
                        imgs_3d, clinicalData, imgs_2d = inputs
                    else:
                        clinicalData, imgs_2d = inputs
                        current_batch_size = clinicalData.shape[0]
                        current_device = clinicalData.device
                else:
                    if self.net_img_3d is not None:
                        imgs_3d, clinicalData = inputs
                    else:
                        clinicalData = inputs
                        current_batch_size = clinicalData.shape[0]
                        current_device = clinicalData.device
            else:
                if self.enable_keyframe:
                    if self.net_img_3d is not None:
                        imgs_3d, imgs_2d = inputs
                    else:
                        imgs_2d = inputs
                        current_batch_size = imgs_2d.shape[0]
                        current_device = imgs_2d.device
                else:
                    #imgs = inputs
                    raise RuntimeError("All optional input branch disabled, so don't use this module.")
        
        if self.net_img_3d is not None:
            output = self.net_img_3d(imgs_3d)
            if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
                o_img_label, o_img_ffr, o_img_ifr = output
            elif self.enable_multibranch_ffr:
                o_img_label, o_img_ffr = output
            elif self.enable_multibranch_ifr:
                o_img_label, o_img_ifr = output
            else:
                o_img_label = output
            
            union_label = o_img_label
            if self.enable_multibranch_ffr:
                union_ffr = o_img_ffr
            if self.enable_multibranch_ifr:
                union_ifr = o_img_ifr
        else:
            union_label = torch.zeros(size=(current_batch_size, 0), device=current_device)
            if self.enable_multibranch_ffr:
                union_ffr = torch.zeros(size=(current_batch_size, 0), device=current_device)
            if self.enable_multibranch_ifr:
                union_ifr = torch.zeros(size=(current_batch_size, 0), device=current_device)

        if self.enable_doubleView:
            output_doubleView = self.net_img_3d(doubleView_3d)
            if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
                o_doubleView_label, o_doubleView_ffr, o_doubleView_ifr = output_doubleView
            elif self.enable_multibranch_ffr:
                o_doubleView_label, o_doubleView_ffr = output_doubleView
            elif self.enable_multibranch_ifr:
                o_doubleView_label, o_doubleView_ifr = output_doubleView
            else:
                o_img_label = output_doubleView
            
            union_label = torch.cat( (union_label, o_doubleView_label) , dim=1)
            if self.enable_multibranch_ffr:
                union_ffr = torch.cat( (union_ffr, o_doubleView_ffr) , dim=1)
            if self.enable_multibranch_ifr:
                union_ifr = torch.cat( (union_ifr, o_doubleView_ifr) , dim=1)

        if self.enable_clinicalData:
            o_clinicalData = self.fc_clinicalData(clinicalData)
            #if self.enable_batchnorm:
            #    o_clinicalData = self.bn(o_clinicalData)
            union_label = torch.cat( (union_label, o_clinicalData) , dim=1)
            if self.enable_multibranch_ffr:
                union_ffr = torch.cat( (union_ffr, o_clinicalData) , dim=1)
            if self.enable_multibranch_ifr:
                union_ifr = torch.cat( (union_ifr, o_clinicalData) , dim=1)

        if self.enable_keyframe:
            output_keyframe = self.net_img_2d(imgs_2d)
            if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
                o_keyframe_label, o_keyframe_ffr, o_keyframe_ifr = output_keyframe
            elif self.enable_multibranch_ffr:
                o_keyframe_label, o_keyframe_ffr = output_keyframe
            elif self.enable_multibranch_ifr:
                o_keyframe_label, o_keyframe_ifr = output_keyframe
            else:
                o_keyframe_label = output_keyframe
            
            union_label = torch.cat( (union_label, o_keyframe_label) , dim=1)
            if self.enable_multibranch_ffr:
                union_ffr = torch.cat( (union_ffr, o_keyframe_ffr) , dim=1)
            if self.enable_multibranch_ifr:
                union_ifr = torch.cat( (union_ifr, o_keyframe_ifr) , dim=1)
            
            if self.enable_doubleView:
                output_keyframe = self.net_img_2d(doubleView_2d)
                if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
                    o_doubleView_keyframe_label, o_doubleView_keyframe_ffr, o_doubleView_keyframe_ifr = output_keyframe
                elif self.enable_multibranch_ffr:
                    o_doubleView_keyframe_label, o_doubleView_keyframe_ffr = output_keyframe
                elif self.enable_multibranch_ifr:
                    o_doubleView_keyframe_label, o_doubleView_keyframe_ifr = output_keyframe
                else:
                    o_doubleView_keyframe_label = output_keyframe
                
                union_label = torch.cat( (union_label, o_doubleView_keyframe_label) , dim=1)
                if self.enable_multibranch_ffr:
                    union_ffr = torch.cat( (union_ffr, o_doubleView_keyframe_ffr) , dim=1)
                if self.enable_multibranch_ifr:
                    union_ifr = torch.cat( (union_ifr, o_doubleView_keyframe_ifr) , dim=1)

        o_label = self.classifier2_label(F.relu(self.classifier_label(union_label)))
        if self.enable_multibranch_ffr:
            o_ffr = self.classifier2_ffr(F.relu(self.classifier_ffr(union_ffr)))
        if self.enable_multibranch_ifr:
            o_ifr = self.classifier2_ifr(F.relu(self.classifier_ifr(union_ifr)))

        if self.enable_multibranch_ffr and self.enable_multibranch_ifr:
            return o_label, o_ffr, o_ifr
        elif self.enable_multibranch_ffr:
            return o_label, o_ffr
        elif self.enable_multibranch_ifr:
            return o_label, o_ifr
        else:
            return o_label


def reduceInputChannel(model): # only for ResNet3D
    weight = model.stem[0].weight.clone()

    model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

    with torch.no_grad():
        model.stem[0].weight = nn.Parameter(weight.mean(dim=1).unsqueeze(1))