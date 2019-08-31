import math
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain

def MSRInitializer(Alpha=0, WeightScale=1):
    def Initializer(Tensor):
        _, fan_out = _calculate_fan_in_and_fan_out(Tensor)
        gain = calculate_gain('leaky_relu', Alpha)
        std = gain / math.sqrt(fan_out)
        bound = math.sqrt(3.0) * std * WeightScale
        with torch.no_grad():
             if WeightScale != 0:
                return Tensor.uniform_(-bound, bound)
             else:
                return Tensor.zero_()
    return Initializer

def XavierInitializer(Tensor):
    _, fan_out = _calculate_fan_in_and_fan_out(Tensor)
    gain = calculate_gain('sigmoid')
    std = gain * math.sqrt(1.0 / fan_out)
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return Tensor.uniform_(-a, a)

def RawConvolutionalLayer(InputChannels, OutputChannels, ReceptiveField=3, Strides=1, KernelInitializer=None, UseBias=True, Groups=1):
    KernelInitializer = MSRInitializer() if KernelInitializer is None else KernelInitializer
    ConvolutionalLayer = nn.Conv2d(InputChannels, OutputChannels, kernel_size=ReceptiveField, stride=Strides, padding=(ReceptiveField - 1) // 2, bias=UseBias, groups=Groups)
    KernelInitializer(ConvolutionalLayer.weight)
    if UseBias:
       ConvolutionalLayer.bias.data.fill_(0)
    return ConvolutionalLayer

def RawFullyConnectedLayer(InputFeatures, OutputFeatures):
    FullyConnectedLayer = nn.Linear(InputFeatures, OutputFeatures, bias=True)
    XavierInitializer(FullyConnectedLayer.weight)
    FullyConnectedLayer.bias.data.fill_(0)
    return FullyConnectedLayer

def BatchNorm(Channels):
    RawBatchNorm = nn.BatchNorm2d(Channels)
    RawBatchNorm.weight.data.fill_(1)
    RawBatchNorm.bias.data.fill_(0)
    return RawBatchNorm
    
class ConvolutionalLayer(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3, Strides=1, Activation=True, BatchNormalization=True, Junction=False, WeightScale=0, Groups=1):
          super(ConvolutionalLayer, self).__init__()
          WeightScale = WeightScale if Junction else 1
          Alpha = 0 if Activation else 1
          self.LinearLayer = RawConvolutionalLayer(InputChannels, OutputChannels, ReceptiveField, Strides, MSRInitializer(Alpha, WeightScale), not BatchNormalization, Groups)
          self.BatchNormalizationLayer = BatchNorm(OutputChannels) if BatchNormalization else None
          self.ActivationLayer = nn.ReLU(inplace=True if BatchNormalization else False) if Activation else None
      def forward(self, x, Shortcut=None):
          RawOutputFlag = self.BatchNormalizationLayer is None and self.ActivationLayer is None
          x = self.LinearLayer(x)
          x = x + Shortcut if Shortcut is not None else x
          y = self.BatchNormalizationLayer(x) if self.BatchNormalizationLayer is not None else x
          y = self.ActivationLayer(y) if self.ActivationLayer is not None else y
          return [x, y] if RawOutputFlag == False else x
            
class BottleneckConvolutionalLayer(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3, Strides=1, Activation=True, BatchNormalization=True, Junction=False, WeightScale=0):
          super(BottleneckConvolutionalLayer, self).__init__()
          self.BottleneckLayer = ConvolutionalLayer(InputChannels, OutputChannels, 1, 1, BatchNormalization=BatchNormalization)
          self.TransformationLayer = ConvolutionalLayer(OutputChannels, OutputChannels, ReceptiveField, Strides, Activation, BatchNormalization, Junction, WeightScale)
      def forward(self, x, Shortcut=None):
          _, x = self.BottleneckLayer(x)
          return self.TransformationLayer(x, Shortcut)       
        
class SequentialConvolutionalLayer(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3, Windowed=True, BatchNormalization=True, Junction=False, WeightScale=0, k=8, DropRate=0.1):
          super(SequentialConvolutionalLayer, self).__init__()
          Filters = []
          self.Windowed = Windowed
          for x in range(OutputChannels // k):
              Filters += [BottleneckConvolutionalLayer(InputChannels if Windowed else InputChannels + x * k, k, ReceptiveField=ReceptiveField, Strides=1, Activation=True, BatchNormalization=BatchNormalization, Junction=Junction, WeightScale=WeightScale)]
          self.Filters = nn.ModuleList(Filters)
          self.Dropout = nn.Dropout2d(p=DropRate) if DropRate > 0 else None
          self.k = k
      def forward(self, x, Shortcut=None):
          SequenceView = x
          RawSequence = []
          ActivatedSequence = []
          for y in range(len(self.Filters)):
              CurrentShortcutSlice = Shortcut[:, y * self.k : (y + 1) * self.k, :, :] if Shortcut is not None else None
              DropoutSequenceView = self.Dropout(SequenceView) if self.Dropout is not None else SequenceView
              RawFeature, ActivatedFeature = self.Filters[y](DropoutSequenceView, CurrentShortcutSlice)
              RawSequence += [RawFeature]
              ActivatedSequence += [ActivatedFeature]
              SequenceView = SequenceView[:, self.k:, :, :] if self.Windowed else SequenceView
              SequenceView = torch.cat([SequenceView, ActivatedFeature], 1) 
          return torch.cat(RawSequence, 1), torch.cat(ActivatedSequence, 1)     
        
class ResidualBlock(nn.Module):
      def __init__(self, InputChannels, BatchNormalization=True, WeightScale=0, k=8):
          super(ResidualBlock, self).__init__()
          self.LayerA = SequentialConvolutionalLayer(InputChannels, InputChannels, BatchNormalization=BatchNormalization, k=k)
          self.LayerB = SequentialConvolutionalLayer(InputChannels, InputChannels, BatchNormalization=BatchNormalization, Junction=True, WeightScale=WeightScale, k=k)
      def forward(self, x, Shortcut):
          _, x = self.LayerA(x)
          return self.LayerB(x, Shortcut)
      
class DownsampleLayer(nn.Module):
      def __init__(self, InputChannels, OutputChannels, BatchNormalization=True, k=8):
          super(DownsampleLayer, self).__init__()
          self.ExtensionLayer = SequentialConvolutionalLayer(InputChannels, OutputChannels - InputChannels, BatchNormalization=BatchNormalization, k=k)
          self.ShrinkingLayer = ConvolutionalLayer(OutputChannels, OutputChannels, 3, 2, BatchNormalization=BatchNormalization, Groups=OutputChannels // k)
      def forward(self, x):
          _, ActivatedFeatures = self.ExtensionLayer(x)
          x = torch.cat([x, ActivatedFeatures], 1)
          return self.ShrinkingLayer(x)
      
class ResNetCIFAR(nn.Module):
      def __init__(self, Classes=10, BlocksPerStage=[3, 3, 3], PyramidFactor=[1, 2, 4], Widening=8, Granularity=8, BatchNormalization=True, DropRate=0.1, WeightScale=0):
          super(ResNetCIFAR, self).__init__()
          Settings = dict(BatchNormalization=BatchNormalization, k=Granularity)
          Stage2Width = 16 * PyramidFactor[1] / PyramidFactor[0]
          Stage3Width = 16 * PyramidFactor[2] / PyramidFactor[0]
          self.Init = ConvolutionalLayer(3, 16, BatchNormalization=BatchNormalization)
          self.Head = SequentialConvolutionalLayer(16, 16 * Widening, Windowed=False, DropRate=0, **Settings)
          self.Stage1 = nn.ModuleList([ResidualBlock(16 * Widening, WeightScale=WeightScale, **Settings) for _ in range(BlocksPerStage[0])])
          self.Downsample1 = DownsampleLayer(16 * Widening, int(Stage2Width * Widening), **Settings)
          self.Stage2 = nn.ModuleList([ResidualBlock(int(Stage2Width * Widening), WeightScale=WeightScale, **Settings) for _ in range(BlocksPerStage[1])])
          self.Downsample2 = DownsampleLayer(int(Stage2Width * Widening), int(Stage3Width * Widening), **Settings)
          self.Stage3 = nn.ModuleList([ResidualBlock(int(Stage3Width * Widening), WeightScale=WeightScale, **Settings) for _ in range(BlocksPerStage[2])])
          self.Dropout = nn.Dropout2d(p=DropRate) if DropRate > 0 else None
          self.Blender = ConvolutionalLayer(int(Stage3Width * Widening), int(Stage3Width * Widening), 1, 1, BatchNormalization=BatchNormalization)
          self.Compress = nn.AdaptiveAvgPool2d((1, 1))
          self.Classifier = RawFullyConnectedLayer(int(Stage3Width * Widening), Classes)         
      def forward(self, x):
          def Refine(ActivatedFeatures, RawFeatures, PoolOfBlocks):
              for Block in PoolOfBlocks:
                  RawFeatures, ActivatedFeatures = Block(ActivatedFeatures, RawFeatures)
              return RawFeatures, ActivatedFeatures   
          _, ActivatedFeatures = self.Init(x)
          RawFeatures, ActivatedFeatures = self.Head(ActivatedFeatures)
          RawFeatures, ActivatedFeatures = Refine(ActivatedFeatures, RawFeatures, self.Stage1)
          RawFeatures, ActivatedFeatures = self.Downsample1(ActivatedFeatures)
          RawFeatures, ActivatedFeatures = Refine(ActivatedFeatures, RawFeatures, self.Stage2)
          RawFeatures, ActivatedFeatures = self.Downsample2(ActivatedFeatures)
          RawFeatures, ActivatedFeatures = Refine(ActivatedFeatures, RawFeatures, self.Stage3)  
          ActivatedFeatures = self.Dropout(ActivatedFeatures) if self.Dropout is not None else ActivatedFeatures
          RawFeatures, ActivatedFeatures = self.Blender(ActivatedFeatures)
          x = self.Compress(ActivatedFeatures)
          x = x.view(x.size(0), -1)
          x = self.Classifier(x)
          return x       