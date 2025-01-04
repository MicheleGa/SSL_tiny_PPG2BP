import torch


class conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, keral,stride=1, groups=1,activation = None):
        super().__init__()

        padding = keral//2
        self.use_activation = activation
        self.conv = torch.nn.Conv1d(in_channels, out_channels, keral, stride,padding, groups=groups)
        self.bath = torch.nn.BatchNorm1d(out_channels)
        if self.use_activation == 'Relu':
            self.activation = torch.nn.ReLU6()
        elif self.use_activation == 'H_swish':
            self.activation = torch.nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        if x.size()[-1] != 1:
            x = self.bath(x)
        if self.use_activation != None:
            x = self.activation(x)
        return x
    

class SE_block(torch.nn.Module):
    def __init__(self,in_channel,ratio=1):
        super(SE_block, self).__init__()
        self.avepool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear1 = torch.nn.Linear(in_channel,in_channel//ratio)
        self.linear2 = torch.nn.Linear(in_channel//ratio,in_channel)
        self.Hardsigmoid = torch.nn.Hardsigmoid(inplace=True)
        self.Relu = torch.nn.ReLU(inplace=True)

    def forward(self,input):
        b,c,_ = input.shape
        x = self.avepool(input)
        x = x.view([b,c])
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Hardsigmoid(x)
        x = x.view([b,c,1])

        return input * x


class bottleneck(torch.nn.Module):
    """ From https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch/blob/main/MobileNetV3.py """
    def __init__(self, in_channels, keral_size, expansion_size, out_channels, use_attenton=False, activation='Relu', stride=1):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attenton = use_attenton

        self.conv = conv(in_channels,expansion_size,1,activation=activation)
        self.conv1 = conv(expansion_size,expansion_size,keral_size,stride=stride,groups=expansion_size,activation=activation)

        if self.use_attenton:
            self.attenton = SE_block(expansion_size)

        self.conv2 = conv(expansion_size,out_channels,1,activation=activation)

    def forward(self,x):

        x1 = self.conv(x)
        x1 = self.conv1(x1)
        if self.use_attenton:
            x1 = self.attenton(x1)
        x1 = self.conv2(x1)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x1 += x

        return x1
    

class MobileNetV3_small(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = torch.nn.Sequential(
            conv(in_channels,16,3,2,activation='H_swish'),
            bottleneck(16,3,16,16,True,'Relu',2),
            bottleneck(16,3,72,24,False,'Relu',2),
            bottleneck(24,3,88,24,False,'Relu',1),
            bottleneck(24,5,96,40,False,'H_swish',2),
            bottleneck(40,5,240,40,True,'H_swish',1),
            bottleneck(40,5,240,40,True,'H_swish',1),
            bottleneck(40,5,120,48,True,'H_swish',1),
            bottleneck(48,5,144,48,True,'H_swish',1),
            bottleneck(48,5,288,96,True,'H_swish',2),
            bottleneck(96,5,576,96,True,'H_swish',1),
            bottleneck(96,5,576,96,True,'H_swish',1),
            conv(96, 576, 1, 1, activation='H_swish'),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten()
        )
        self.sbp_regressor = torch.nn.Linear(576, 1)
        self.dbp_regressor = torch.nn.Linear(576, 1)
        
    def forward(self, x):
        x = self.features(x)
        sbp = self.sbp_regressor(x)
        dbp = self.dbp_regressor(x)
        return sbp, dbp