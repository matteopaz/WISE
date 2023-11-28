import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class FluxAnomalyPredictionTF(nn.Module):
  def __init__(self, stride, dropout, bn=False, features=3, residual=0, out=4):
    super().__init__()

    self.stride = stride
    self.features = features
    self.residual = residual
    self.transformer_hidden_dim = 2*features # Needed for batchnorm 3, so defined here

    # Need Dropouts, Activation fns
    self.he = lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu')
    self.av = nn.ReLU()
    self.drop = nn.Dropout(dropout)

    self.batchnorm1 = lambda x: x
    self.batchnorm2 = lambda x: x
    self.batchnorm3 = lambda x: x

    if bn:
      self.batchnorm1 = nn.BatchNorm1d(features)
      self.batchnorm2 = nn.BatchNorm1d(features)
      self.batchnorm3 = nn.BatchNorm1d(self.transformer_hidden_dim)

      self.bn1 = lambda x: torch.transpose(self.batchnorm1(torch.transpose(x,1,2)),1,2)
      self.bn2 = lambda x: torch.transpose(self.batchnorm2(torch.transpose(x,1,2)),1,2)
      self.bn3 = lambda x: self.batchnorm3(x)
    else:
      self.bn1 = lambda x: x
      self.bn2 = lambda x: x
      self.bn3 = lambda x: x


    # Step 1

    # 5 x 64 X 64 x 5 -diag-> R^5 Vector
    self.conv_kernels_64d = nn.ParameterList([self.he(torch.randn(features, 64)) for i in range(8)])
    self.conv_biases_64d = nn.ParameterList([torch.randn(features) for i in range(8)])

    self.conv_kernel_16d = nn.Parameter(self.he(torch.randn(features, 16)))
    self.conv_bias_16d = nn.Parameter(torch.randn(features))

    self.conv_kernel_8d = nn.Parameter(self.he(torch.randn(features,8)))
    self.conv_bias_8d = nn.Parameter(torch.randn(features))

    self.four_max_pool = nn.MaxPool1d(4)

    # Step 1.5
    # self.certainty_fc_1 = nn.Parameter(torch.zeros(2,3))
    # self.certainty_fc_2 = nn.Parameter(torch.zeros(3,1))

    # Step 2
    self.widechannel_conv_kernel = nn.Parameter(self.he(torch.randn(features, 3)))
    self.widechannel_conv_bias = nn.Parameter(torch.randn(features))

    self.midchannel_conv_kernel = nn.Parameter(self.he(torch.randn(features, 3)))
    self.midchannel_conv_bias = nn.Parameter(torch.randn(features))

    self.narrowchannel_conv_kernel = nn.Parameter(self.he(torch.randn(features, 3)))
    self.narrowchannel_conv_bias = nn.Parameter(torch.randn(features))


    # Step 3
    self.pair_max_pool = nn.MaxPool1d(2)
    # Then concat all vectors into v \in R^24

    # Step 3.5 Residual connection
    self.__avgpool__ = nn.AvgPool1d(4, stride=4)

    # Step 4
    self.hidden_fc_1 = nn.Linear(3*4*features, 3*2*features)
    self.hidden_fc_2 = nn.Linear(3*2*features, self.transformer_hidden_dim)

    # Step 5
    # Transformer
    self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_hidden_dim, nhead=self.features)
    self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, 2)


    # Step 6
    # Softmax

    self.to_out_1 = nn.Linear(self.transformer_hidden_dim, 7)
    self.to_out_2 = nn.Linear(7, out, bias=False)

    self.prob = nn.Softmax(dim=1)

    for param in self.parameters():
      if len(param.shape) >= 2:
        param = self.he(param)



  def forward(self, x):
    # x = Batches X Time X Channels

    N = x.shape[0] # Batches
    T = x.shape[1] # Time

    if x.shape[2] != self.features:
      raise Exception("Feature dimension mismatch")

    # Step 1
    # Cursory vision convolution
    pad_amt = 40

    pad = torch.zeros(N, pad_amt, self.features).to(device)
  

    padded_x = torch.cat((pad, torch.cat((x, pad), dim=1)), dim=1) # catting in time

    # padded_x.requires_grad = True

    window_centers = [] # AKA 64d Convolve Centers

    n=0 # strides
    while(True):
      next_center = pad_amt + 1 + n * self.stride
      if next_center > (pad_amt + T + 1): # If our center isnt in real data
        break;
      else:
        window_centers.append(next_center)
        n += 1

    window_starts = [self.start_and_end_from_center((64 - 1)/2, i)[0] for i in window_centers]
    window_ends = [self.start_and_end_from_center((64 - 1)/2, i)[1] for i in window_centers]

    midchannel_centers = []
    narrowchannel_centers = []
    for start in window_starts:
      for n in range(8): # 8 strides of 8 -> 64 units
        midchannel_centers.append(start + n * 8)

      for n in range(32): # 32 strides of 2 -> 64 units
        narrowchannel_centers.append(start + n * 2)


    wide_convs = []
    mid_convs = []
    narrow_convs = []

    for i in window_centers:
      for j in range(8): # Hard coded 8 here
        K = self.conv_kernels_64d[j]
        B = self.conv_biases_64d[j].repeat(N,1) # Repeat here
        conv = self.convolve(K, padded_x, i) # Features x T
        conv += B
        wide_convs.append(conv)

    for i in midchannel_centers:
      conv = self.convolve(self.conv_kernel_16d, padded_x, i)
      conv += self.conv_bias_16d.repeat(N,1)
      mid_convs.append(conv)

    for i in narrowchannel_centers:
      conv = self.convolve(self.conv_kernel_8d, padded_x, i)
      conv += self.conv_bias_8d.repeat(N,1)
      narrow_convs.append(conv)

    wide_convs = torch.stack(wide_convs, dim=1).to(device)
    mid_convs = torch.stack(mid_convs, dim=1).to(device)
    narrow_convs = torch.stack(narrow_convs, dim=1).to(device)

    narrow_convs = self.four_max_pool(torch.transpose(narrow_convs, 1,2)) # Inp = N x C x L now
    narrow_convs = torch.transpose(narrow_convs,1,2) # Back to N x L x C


    wide_convs = self.bn1(self.av(wide_convs))
    mid_convs = self.bn1(self.av(mid_convs))
    narrow_convs = self.bn1(self.av(narrow_convs))

    residual_vectors = self.get_residual_vectors(narrow_convs, mid_convs, wide_convs)


    wide_convs = 5 * self.drop(wide_convs)
    mid_convs = self.drop(mid_convs)
    narrow_convs = self.drop(narrow_convs)



    #####
    # STEP 2
    # Second Convolution

    pad_amt = 10
    stride = 1

    results = []

    for x in (wide_convs, mid_convs, narrow_convs):
      T = x.shape[1]
      ker = None
      bias = None
      if len(results) == 0:
        ker = self.widechannel_conv_kernel
        bias = self.widechannel_conv_bias.repeat(N, 1)

      elif len(results) == 1:
        ker = self.midchannel_conv_kernel
        bias = self.midchannel_conv_bias.repeat(N, 1)

      elif len(results) == 2:
        ker = self.narrowchannel_conv_kernel
        bias = self.narrowchannel_conv_bias.repeat(N, 1)


      pad = torch.zeros(N, pad_amt, self.features).to(device)
      padded_x = torch.cat((pad, torch.cat((x, pad), dim=1)), dim=1)


      result = []

      next = pad_amt
      while next <= (pad_amt + T - 1):
        v = bias + self.convolve(ker, padded_x, next)
        result.append(v)
        next += stride


      results.append(torch.stack(result, dim=1))


    wide_convs = self.bn2(self.av(results[0]))
    mid_convs = self.bn2(self.av(results[1]))
    narrow_convs = self.bn2(self.av(results[2]))

    wide_convs = self.drop(wide_convs)
    mid_convs = self.drop(mid_convs)
    narrow_convs = self.drop(narrow_convs)



    ####
    # Step 2.5
    # Max Pooling Pairs
    wide_convs = self.pair_max_pool(torch.transpose(wide_convs, 1,2)) # to N x C x L
    mid_convs = self.pair_max_pool(torch.transpose(mid_convs, 1,2))
    narrow_convs = self.pair_max_pool(torch.transpose(narrow_convs, 1,2))

    wide_convs = torch.transpose(wide_convs, 1,2) # to N x L x C
    mid_convs = torch.transpose(mid_convs, 1,2)
    narrow_convs = torch.transpose(narrow_convs, 1,2)


    ####
    # Step 3

    # Now each original window reigon is each corresponding 2 rows from all 3 tensors
    # 2 rows evenly divides all possible resulting lengths

    if not (wide_convs.shape[1] == mid_convs.shape[1] == narrow_convs.shape[1]):
      raise Exception("Step 3 output mismatch")


    hidden = []
    L = wide_convs.shape[1]

    for n in range(L // 4): # Now each sliding window corresponds to 2 rows
      wide = wide_convs[:, 4*n:4*n + 4].reshape(N, 4*self.features)
      mid = mid_convs[:, 4*n:4*n + 4].reshape(N, 4*self.features)
      narrow = narrow_convs[:, 4*n:4*n + 4].reshape(N, 4*self.features)

      flat = torch.cat((wide, mid, narrow), dim=1).to(device)
      if flat.shape[0] != N or flat.shape[1] != 3*4*self.features:
        raise Exception("Flat shape err")

      layer_1 = self.hidden_fc_1(flat)
      layer_1 = self.av(layer_1)

      # RESIDUAL CONNECTION !
      res = torch.autograd.Variable(self.residual * residual_vectors[n])
      layer_1 = (1-self.residual) * layer_1
      layer_1 = layer_1 + res

      layer_2 = self.hidden_fc_2(layer_1)
      layer_2 = self.av(layer_2)

      hidden.append(layer_2)

    hidden = torch.stack(hidden, dim=0) # Results in (Divided L) x N x Hidden
    hidden = self.drop(hidden)


    seq = self.transformer_encoder(hidden) # same shape as hidden



    transformed = nn.AvgPool1d(seq.shape[0])(torch.transpose(seq,0,2)) # Pooling happens on last dim
    transformed = torch.squeeze(transformed, dim=2)
    transformed = torch.transpose(transformed, 0, 1) # Should be N x Hidden

    transformed = self.av(transformed)
    transformed = self.bn3(transformed)
    transformed = self.drop(transformed)

    final_layer = self.to_out_1(transformed)
    final_layer = self.av(final_layer)
    final_layer = self.drop(final_layer)
    final_layer = self.to_out_2(final_layer)

    # classes = self.prob(final_layer) # Dont use, use CEL instead

    return F.log_softmax(final_layer, dim=1)


  def start_and_end_from_center(self, width, i):
    start = i - np.ceil(width) 
    end = i + np.floor(width) + 1
    return (int(start), int(end))



  def convolve(self, Kernel, Data, i):

    # i is center index so we take equal on either side
    T = Kernel.shape[1]
    each_side = (T - 1) / 2

    # Moves it backwards 1 if kernel is even
    start, end = self.start_and_end_from_center(each_side, i)

    adj_data = Data[:, start:end, :]

    # So now K is 5 x L and adj_data is N x L X 5


    N = adj_data.shape[0]

    m = torch.bmm(Kernel.repeat(N,1,1), adj_data) # bij, bjk -> bik Slightly faster than einsum

    # m = torch.einsum("ij, bjk -> bik", Kernel, adj_data) #identical batch matmul

    diag = torch.einsum("bii->bi", m)
    # diags = []
    # for res in m:
    #   diags.append(torch.diag(res))


    return diag

  def get_residual_vectors(self, narrow, mid, wide):
    # Should be N x 8m x Features
    # print(narrow.shape, mid.shape, wide.shape)
    N = wide.shape[0]
    L = wide.shape[1]

    apply_avg = lambda matrix: torch.transpose(self.__avgpool__(torch.transpose(matrix, 1,2)), 1,2)

    vectors = []

    if L % 8:
      raise Exception("Something went wrong, convs are not mult of 8")
    for i in range(L // 8):

      n = torch.autograd.Variable(narrow[:, 8*i:8*(i+1), :])
      m = torch.autograd.Variable(mid[:, 8*i:8*(i+1), :])
      w = torch.autograd.Variable(wide[:, 8*i:8*(i+1), :])

      n = apply_avg(n)
      m = apply_avg(m)
      w = apply_avg(w)

      n = n.reshape(N, 2*self.features)
      m = m.reshape(N, 2*self.features)
      w = w.reshape(N, 2*self.features)
      # gives N x 2 * Features

      vectors.append(torch.autograd.Variable(torch.cat((n,m,w), dim=1))) # N x 6 * features

    return vectors