
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pdb
import math
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.nn.functional as F

class ConvNormRelu(nn.Module):
  def __init__(self, in_channels, out_channels,
               type='1d', leaky=False,
               downsample=False, kernel_size=None, stride=None,
               padding=None, p=0, groups=1):
    super(ConvNormRelu, self).__init__()
    if kernel_size is None and stride is None:
      if not downsample:
        kernel_size = 3
        stride = 1
      else:
        kernel_size = 4
        stride = 2

    if padding is None:
      if isinstance(kernel_size, int) and isinstance(stride, tuple):
        padding = tuple(int((kernel_size - st)/2) for st in stride)
      elif isinstance(kernel_size, tuple) and isinstance(stride, int):
        padding = tuple(int((ks - stride)/2) for ks in kernel_size)
      elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
        assert len(kernel_size) == len(stride), 'dims in kernel_size are {} and stride are {}. They must be the same'.format(len(kernel_size), len(stride))
        padding = tuple(int((ks - st)/2) for ks, st in zip(kernel_size, kernel_size))
      else:
        padding = int((kernel_size - stride)/2)


    in_channels = in_channels*groups
    out_channels = out_channels*groups
    if type == '1d':
      self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm1d(out_channels)
      self.dropout = nn.Dropout(p=p)
    elif type == '2d':
      self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm2d(out_channels)
      self.dropout = nn.Dropout2d(p=p)
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=0.2)
    else:
      self.relu = nn.ReLU()

  def forward(self, x, **kwargs):
    return self.relu(self.norm(self.dropout(self.conv(x))))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, state, desc):
        # Set 1
        x = state
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # Set 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        #Flatten
        out = out.view(out.size(0), -1)

        #Dense
        out = self.fc1(out)

        #pdb.set_trace()
        return out

class UNet1D(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)
  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)
  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels
  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor
  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector
  '''
  def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super(UNet1D, self).__init__()
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv1 = nn.ModuleList([])
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth
    self.groups = groups

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    for i in range(self.max_depth):
      self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    for i in range(self.max_depth):
      self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, return_bottleneck=False, return_feats=False, feats=[]):
    input_size = x.shape[-1]
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)

    x = nn.Sequential(*self.pre_downsampling_conv)(x)

    residuals = []
    residuals.append(x)
    for i, conv1 in enumerate(self.conv1):
      x = conv1(x)
      if i < self.max_depth - 1:
        residuals.append(x)

    bn = x

    for i, conv2 in enumerate(self.conv2):
      x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv2(x)
      if return_feats:
        feats.append(x)

    if return_feats:
      return x, feats
    elif return_bottleneck:
      return x, bn
    else:
      return x



class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class StateEncoder(nn.Module):
    def __init__(self,num_classes=1, in_channels = 3):
        super().__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=in_channels,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=512,out_features=256)

    def forward(self, state):

        output = self.net(state)
        output = output.flatten(start_dim=1)
        output = self.fc(output)
        return output



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pdb.set_trace()
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DescEncoder_Transformer(nn.Module):

    def __init__(self, ntoken = 20, ninp = 50, nhead = 2, nhid = 200, nlayers = 6, dropout=0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        #
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output


class DescEncoder(nn.Module):
  '''
  input_shape:  (N, time, text_features: 50)
  output_shape: (N, 256, time)
  '''

  def __init__(self, output_feats=64, input_channels=768, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.fc1 = nn.Linear(15360, 4096)
    self.fc2 = nn.Linear(4096, 1024)
    self.fc3 = nn.Linear(1024, 256)


  def forward(self, desc):
    desc = torch.flatten(desc,  start_dim = 1)
    out = torch.relu(self.fc1(desc))
    out = torch.relu(self.fc2(out))
    out = torch.relu(self.fc3(out))

    return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class DescEncoder_OneHot(nn.Module):
  '''
  input_shape:  (N, time, text_features: 50)
  output_shape: (N, 256, time)
  '''

  def __init__(self, output_feats=64, input_channels=50, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.fc1 = nn.Linear(1000, 768)
    self.bn1 = nn.BatchNorm1d(num_features=768)
    self.fc2 = nn.Linear(768, 512)
    self.bn2 = nn.BatchNorm1d(num_features=512)
    self.fc3 = nn.Linear(512, 256)
    self.bn3 = nn.BatchNorm1d(num_features=256)


  def forward(self, desc):
    desc = torch.flatten(desc,  start_dim = 1)
    out = torch.relu(self.bn1(self.fc1(desc)))
    out = torch.relu(self.bn2(self.fc2(out)))
    out = torch.relu(self.bn3(self.fc3(out)))

    return out


class DescEncoder_BOW(nn.Module):
  '''
  input_shape:  (N, time, text_features: 50)
  output_shape: (N, 256, time)
  '''

  def __init__(self, output_feats=64, input_channels=50, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.fc1 = nn.Linear(50, 40)
    self.bn1 = nn.BatchNorm1d(num_features=40)
    self.fc2 = nn.Linear(40, 30)
    self.bn2 = nn.BatchNorm1d(num_features=30)
    self.fc3 = nn.Linear(30, 20)
    self.bn3 = nn.BatchNorm1d(num_features=20)


  def forward(self, desc):
    desc = torch.flatten(desc,  start_dim = 1)
    out = torch.relu(self.bn1(self.fc1(desc)))
    out = torch.relu(self.bn2(self.fc2(out)))
    out = torch.relu(self.bn3(self.fc3(out)))

    return out


class DescEncoder_OneHot_RNN(nn.Module):
  '''
  input_shape:  (N, time, text_features: 50)
  output_shape: (N, 256, time)
  '''

  def __init__(self, output_feats=64, input_channels=50, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    '''
    input 1: (L = sequence length, N = , H = input_size)
    input 2: (S = num_layers * num_directions, N, H_out = hidden_size)

    >>> rnn = nn.RNN(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> h0 = torch.randn(2, 3, 20)
    >>> output, hn = rnn(input, h0)

    '''

    self.n_hidden = 128
    self.n_layers = 20
    self.rnn = RNN(50, 32, 20)

  def init_hidden(self, batch_size):
    # This method generates the first hidden state of zeros which we'll use in the forward pass
    # We'll send the tensor holding the hidden state to the device we specified earlier as well
    hidden = torch.zeros(self.n_layers, batch_size, self.n_hidden)
    return hidden



  def forward(self, x):
    batch_size = x.size(0)

    x = x.permute(1,0,2)

    # Initializing hidden state for first input using method defined below
    hidden = self.init_hidden(batch_size)
    pdb.set_trace()
    # Passing in the input and hidden state into the model and obtaining outputs
    out, hidden = self.rnn(x, hidden)

    pdb.set_trace()
    
    # # Reshaping the outputs such that it can be fit into the fully connected layer
    # out = out.contiguous().view(-1, self.hidden_dim)
    # out = self.fc(out)
    
    return out, hidden

class Classifier_transformer(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.desc_enc =  DescEncoder_OneHot_RNN()
    self.state_enc = StateEncoder()
    self.fc1 = nn.Linear(656, 312)
    self.fc2 = nn.Linear(312, 1)
    self.leakyrelu = nn.LeakyReLU(0.1)
    # self.fc3 = nn.Linear(128, 64)
    # self.fc4 = nn.Linear(64, 32)
    # self.fc5 = nn.Linear(32, 1)



  def forward(self, state, desc):

    state_encoding = self.state_enc(state)
    
    state_encoding = state_encoding.flatten(start_dim=1)

    desc_encoding = self.desc_enc(desc)
    desc_encoding = desc_encoding.flatten(start_dim=1)
    joint_encoding = torch.cat((state_encoding, desc_encoding), 1)
    joint_encoding = torch.flatten(joint_encoding,  start_dim = 1)

    out = self.leakyrelu(self.fc1(joint_encoding))
    
    out = self.leakyrelu(self.fc2(out))
    # out = torch.relu(self.fc3(out))
    # out = torch.relu(self.fc4(out))
    # out = self.fc5(out)
    
    out = torch.sigmoid(out)


    return out


class Classifier(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.desc_enc =  DescEncoder_BOW()
    self.state_enc = StateEncoder()
    self.fc1 = nn.Linear(276, 256)
    self.bn1 = nn.BatchNorm1d(num_features=256)
    self.fc2 = nn.Linear(256, 128)
    self.bn2 = nn.BatchNorm1d(num_features=128)
    self.fc3 = nn.Linear(128, 64)
    self.bn3 = nn.BatchNorm1d(num_features=64)
    self.fc4 = nn.Linear(64, 32)
    self.bn4 = nn.BatchNorm1d(num_features=32)
    self.fc5 = nn.Linear(32, 2)
    self.bn5 = nn.BatchNorm1d(num_features=1)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()


  def forward(self, state, desc):
    state_encoding = self.state_enc(state)

    state_encoding = state_encoding.flatten(start_dim=1)

    desc_encoding = self.desc_enc(desc)
    desc_encoding = desc_encoding.flatten(start_dim=1)


    joint_encoding = torch.cat((state_encoding, desc_encoding), 1)
    joint_encoding = torch.flatten(joint_encoding,  start_dim = 1)


    out = self.bn1(self.relu(self.fc1(joint_encoding)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = self.bn4(self.relu(self.fc4(out)))
    out = torch.softmax(self.fc5(out), dim =1)

    return out


class ClassifierBB(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.desc_enc =  DescEncoder_BOW()
    self.state_enc = DQNImgEncoder(in_channels = 4)
    self.fc1 = nn.Linear(522, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, 2)
    self.bn4 = nn.BatchNorm1d(num_features=1)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()


  def forward(self, state, desc):
   
    state = state.permute(0,3,1,2)/256
    state_encoding = self.state_enc(state)
    state_encoding = state_encoding.flatten(start_dim=1)
    desc_encoding = desc.squeeze()

    joint_encoding = torch.cat((state_encoding, desc_encoding), 1)
    joint_encoding = torch.flatten(joint_encoding,  start_dim = 1)

    out = self.bn1(self.relu(self.fc1(joint_encoding)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = torch.softmax(self.fc4(out), dim =1)

    return out






class ClassifierMNIST(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.desc_enc =  DescEncoder_BOW()
    self.state_enc = StateEncoder()
    self.fc1 = nn.Linear(276, 256)
    self.bn1 = nn.BatchNorm1d(num_features=256)
    self.fc2 = nn.Linear(256, 128)
    self.bn2 = nn.BatchNorm1d(num_features=128)
    self.fc3 = nn.Linear(128, 64)
    self.bn3 = nn.BatchNorm1d(num_features=64)
    self.fc4 = nn.Linear(64, 32)
    self.bn4 = nn.BatchNorm1d(num_features=32)
    self.fc5 = nn.Linear(32, 1)
    self.bn5 = nn.BatchNorm1d(num_features=1)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()


  def forward(self, state, desc):

    state_encoding = self.state_enc(state)
    state_encoding = state_encoding.flatten(start_dim=1)

    desc_encoding = self.desc_enc(desc)
    desc_encoding = desc_encoding.flatten(start_dim=1)


    joint_encoding = torch.cat((state_encoding, desc_encoding), 1)
    joint_encoding = torch.flatten(joint_encoding,  start_dim = 1)


    out = self.bn1(self.relu(self.fc1(joint_encoding)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = self.bn4(self.relu(self.fc4(out)))
    out = self.relu(self.fc5(out))

    return out



class SimpleClassifier(nn.Module):
  def __init__(self, input_state_feats = 12288, input_desc_feats = 50):
    super().__init__()
    self.fc1  = nn.Linear(input_state_feats, 4096)
    self.fc2  = nn.Linear(4096, 1024)
    self.fc3  = nn.Linear(1024, 256)
    self.fc4  = nn.Linear(256, 2)

  def forward(self, state,desc):
    x = state.reshape(-1, 3*64*64)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    #x = torch.cat((x, desc), 1)
    x = self.fc4(x)
    x = torch.softmax(x, dim = 1)
    return x


class SimpleCNNClassifier(nn.Module):
  def __init__(self, input_state_feats = 12288, input_desc_feats = 50):
    super().__init__()
    self.cnn_layers = Sequential(
              # Defining a 2D convolution layer
              Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
              BatchNorm2d(4),
              ReLU(inplace=True),
              MaxPool2d(kernel_size=2, stride=2),
              # Defining another 2D convolution layer
              Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
              BatchNorm2d(4),
              ReLU(inplace=True),
              MaxPool2d(kernel_size=2, stride=2),
              Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
              BatchNorm2d(4),
              ReLU(inplace=True),
              MaxPool2d(kernel_size=2, stride=2),
          )

    self.fc1  = nn.Linear(256+ 50, 128)
    self.fc2  = nn.Linear(128, 2)



  def forward(self, state,desc):
    x = self.cnn_layers(state)
    x = x.flatten(1)
    x = torch.cat((x, desc), 1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.softmax(x, dim = 1)
    return x




class ClassifierStateBB(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.fc1 = nn.Linear(14, 8)
    self.bn1 = nn.BatchNorm1d(num_features=8)
    self.fc2 = nn.Linear(8, 2)
    self.bn2 = nn.BatchNorm1d(num_features=2)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()


  def forward(self, state, desc):
    
    state_encoding = state.flatten(start_dim=1)
    desc_encoding = desc
    joint_encoding = torch.cat((state_encoding, desc_encoding), 1)
    joint_encoding = torch.flatten(joint_encoding,  start_dim = 1)


    out = self.bn1(self.relu(self.fc1(joint_encoding)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = torch.softmax(out, dim =1)

    return out



class ClassifierBBActionResnet(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.desc_enc =  DescEncoder_BOW()
    self.state_enc = DQNImgEncoder(in_channels = 4)
    self.fc0 = nn.Linear(1034, 512)
    self.bn0 = nn.BatchNorm1d(num_features=512)
    self.fc1 = nn.Linear(522, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, 2)
    self.bn4 = nn.BatchNorm1d(num_features=2)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()
    
    #FILM Networks
    self.gamma1 = nn.Linear(4, 128)
    self.gamma2 = nn.Linear(128, 256)
    self.gamma3 = nn.Linear(256, 512)
    self.beta1 = nn.Linear(4, 128)
    self.beta2 = nn.Linear(128, 256)
    self.beta3 = nn.Linear(256, 512)

  def FiLM(self, x, gamma, beta):
      
      x = gamma * x #+ beta
      
      return x


  def forward(self, state, desc, action):
    state = state.permute(0,3,1,2)/256
    state_encoding = self.state_enc(state)
    state_encoding = state_encoding.flatten(start_dim=1)

    action_encoding = action.flatten(start_dim=1).float()
    desc_encoding = desc.squeeze()

    #FiLM conditioning
    gamma_action_encoding = self.gamma3(self.gamma2(self.gamma1(action_encoding)))
    beta_action_encoding = self.beta3(self.beta2(self.beta1(action_encoding)))
    state_act = self.FiLM(state_encoding, gamma_action_encoding, beta_action_encoding)
    #joint_encoding = torch.cat((state_encoding, desc_encoding, gamma_action_encoding), 1)
    out = torch.cat((state_act, desc_encoding), 1).float()

    #out = self.bn0(self.relu(self.fc0(joint_encoding.float())))
    out = self.bn1(self.relu(self.fc1(out)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = torch.softmax(self.fc4(out), dim =1)

    return out





class DQNImgEncoder(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)

        
    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.flatten(start_dim = 1)))
        return x



class ClassifierBBActionResnetNoMult(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.state_enc = DQNImgEncoder(in_channels = 4)
    self.fc0 = nn.Linear(1034, 512)
    self.bn0 = nn.BatchNorm1d(num_features=512)
    self.fc1 = nn.Linear(512, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, 2)
    self.bn4 = nn.BatchNorm1d(num_features=2)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()
    
    #FILM Networks
    self.gamma1 = nn.Linear(4, 128)
    self.gamma2 = nn.Linear(128, 256)
    self.gamma3 = nn.Linear(256, 512)


  def forward(self, state, desc, action):
    state = state.permute(0,3,1,2)/256
    state_encoding = self.state_enc(state)
    state_encoding = state_encoding.flatten(start_dim=1)

    action_encoding = action.flatten(start_dim=1).float()
    desc_encoding = desc.squeeze()

    #FiLM conditioning
    gamma_action_encoding = self.gamma3(self.gamma2(self.gamma1(action_encoding)))
    print("HI")
    pdb.set_trace()
    joint_encoding = torch.cat((state_encoding,  gamma_action_encoding, desc_encoding), 1)
    #out = torch.cat((state_act, desc_encoding), 1).float()
    pdb.set_trace()
    out = self.bn0(self.relu(self.fc0(joint_encoding.float())))
    out = self.bn1(self.relu(self.fc1(out)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = torch.softmax(self.fc4(out), dim =1)

    return out



class ClassifierCR(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.desc_enc =  DescEncoder_BOW()
    self.state_enc = DQNImgEncoderClever(in_channels = 3)
    self.fc1 = nn.Linear(562, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, 2)
    self.bn4 = nn.BatchNorm1d(num_features=1)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()


  def forward(self, state, desc):
    
   
    state = state.permute(0,3,1,2)/256
    state_encoding = self.state_enc(state)
    
    state_encoding = state_encoding.flatten(start_dim=1)
    desc_encoding = desc.squeeze()

    joint_encoding = torch.cat((state_encoding, desc_encoding), 1)
    joint_encoding = torch.flatten(joint_encoding,  start_dim = 1)

    out = self.bn1(self.relu(self.fc1(joint_encoding)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = torch.softmax(self.fc4(out), dim =1)
    return out

class ClassifierCRActionResnetNoMult(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.state_enc = DQNImgEncoderClever(in_channels = 3)
    self.fc0 = nn.Linear(1074, 512)
    self.bn0 = nn.BatchNorm1d(num_features=512)
    self.fc1 = nn.Linear(512, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, 2)
    self.bn4 = nn.BatchNorm1d(num_features=2)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()
    
    #FILM Networks
    self.gamma1 = nn.Linear(4, 128)
    self.gamma2 = nn.Linear(128, 256)
    self.gamma3 = nn.Linear(256, 512)


  def forward(self, state, desc, action):
    state = state.permute(0,3,1,2)/256
    state_encoding = self.state_enc(state)
    state_encoding = state_encoding.flatten(start_dim=1)

    action_encoding = action.flatten(start_dim=1).float()
    desc_encoding = desc.squeeze()
    

    #FiLM conditioning
    gamma_action_encoding = self.gamma3(self.gamma2(self.gamma1(action_encoding)))

    joint_encoding = torch.cat((state_encoding,  gamma_action_encoding, desc_encoding), 1)
    #out = torch.cat((state_act, desc_encoding), 1).float()

    out = self.bn0(self.relu(self.fc0(joint_encoding.float())))
    out = self.bn1(self.relu(self.fc1(out)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = torch.softmax(self.fc4(out), dim =1)

    return out

class ClassifierCRActionResnet(nn.Module):
  def __init__(self, output_feats=64, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.desc_enc =  DescEncoder_BOW()
    self.state_enc = DQNImgEncoderClever(in_channels = 3)
    self.fc0 = nn.Linear(562, 512)
    self.bn0 = nn.BatchNorm1d(num_features=512)
    self.fc1 = nn.Linear(512, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, 2)
    self.bn4 = nn.BatchNorm1d(num_features=2)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()
    
    #FILM Networks
    self.gamma1 = nn.Linear(4, 128)
    self.gamma2 = nn.Linear(128, 256)
    self.gamma3 = nn.Linear(256, 512)
    self.beta1 = nn.Linear(4, 128)
    self.beta2 = nn.Linear(128, 256)
    self.beta3 = nn.Linear(256, 512)

  def FiLM(self, x, gamma):
      
      x = gamma * x #+ beta
      
      return x


  def forward(self, state, desc, action):
    state = state.permute(0,3,1,2)/255
    state_encoding = self.state_enc(state)
    state_encoding = state_encoding.flatten(start_dim=1)

    action_encoding = action.flatten(start_dim=1).float()
    desc_encoding = desc.squeeze()
    #FiLM conditioning
    gamma_action_encoding = self.gamma3(self.gamma2(self.gamma1(action_encoding)))
    #beta_action_encoding = self.beta3(self.beta2(self.beta1(action_encoding)))
    state_act = self.FiLM(state_encoding, gamma_action_encoding)
    #joint_encoding = torch.cat((state_encoding, desc_encoding, gamma_action_encoding), 1)
    out = torch.cat((state_act, desc_encoding), 1).float()

    out = self.bn0(self.relu(self.fc0(out)))
    out = self.bn1(self.relu(self.fc1(out)))
    out = self.bn2(self.relu(self.fc2(out)))
    out = self.bn3(self.relu(self.fc3(out)))
    out = torch.softmax(self.fc4(out), dim =1)

    return out



class DQNImgEncoderClever(nn.Module):
  def __init__(self, in_channels):
      """
      Initialize Deep Q Network
      Args:
          in_channels (int): number of input channels
      """
      super().__init__()
      self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
      self.bn1 = nn.BatchNorm2d(32)
      self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
      self.bn2 = nn.BatchNorm2d(64)
      self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
      self.bn3 = nn.BatchNorm2d(64)
      self.fc4 = nn.Linear(4 * 4 * 64, 512)

      
  def forward(self, x):
      x = x.float()
      x = F.relu(self.bn1(self.conv1(x)))
      x = F.relu(self.bn2(self.conv2(x)))
      x = F.relu(self.bn3(self.conv3(x)))
      x = F.relu(self.fc4(x.flatten(start_dim = 1)))
      return x
