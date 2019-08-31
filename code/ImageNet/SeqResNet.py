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
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3, Windowed=True, BatchNormalization=True, Junction=False, WeightScale=0, k=8):
          super(SequentialConvolutionalLayer, self).__init__()
          Filters = []
          self.Windowed = Windowed
          for x in range(OutputChannels // k):
              Filters += [BottleneckConvolutionalLayer(InputChannels if Windowed else InputChannels + x * k, k, ReceptiveField=ReceptiveField, Strides=1, Activation=True, BatchNormalization=BatchNormalization, Junction=Junction, WeightScale=WeightScale)]
          self.Filters = nn.ModuleList(Filters)
          self.k = k
      def forward(self, x, Shortcut=None):
          SequenceView = x
          RawSequence = []
          ActivatedSequence = []
          for y in range(len(self.Filters)):
              CurrentShortcutSlice = Shortcut[:, y * self.k : (y + 1) * self.k, :, :] if Shortcut is not None else None
              RawFeature, ActivatedFeature = self.Filters[y](SequenceView, CurrentShortcutSlice)
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
      def __init__(self, InputChannels, OutputChannels, BatchNormalization=True, k=8, Reduce=1):
          super(DownsampleLayer, self).__init__()
          self.ExtensionLayer = SequentialConvolutionalLayer(InputChannels, OutputChannels - InputChannels, BatchNormalization=BatchNormalization, k=k)
          self.ShrinkingLayer = ConvolutionalLayer(OutputChannels, OutputChannels, 3, 2, BatchNormalization=BatchNormalization, Groups=OutputChannels // k // Reduce)
      def forward(self, x):
          _, ActivatedFeatures = self.ExtensionLayer(x)
          x = torch.cat([x, ActivatedFeatures], 1)
          return self.ShrinkingLayer(x)
      
class ResNetImageNet(nn.Module):
      def __init__(self, Classes=1000, BlocksPerStage=[1, 1, 1, 1], PyramidFactor=[1, 2, 4, 8], Widening=2, Granularity=[8, 16, 32, 64], BatchNormalization=True, WeightScale=0):
          super(ResNetImageNet, self).__init__()
          Settings = dict(BatchNormalization=BatchNormalization)
          Stage2Width = 64 * PyramidFactor[1] // PyramidFactor[0]
          Stage3Width = 64 * PyramidFactor[2] // PyramidFactor[0]
          Stage4Width = 64 * PyramidFactor[3] // PyramidFactor[0]
          self.Head0 = ConvolutionalLayer(3, 32, Strides=2, BatchNormalization=BatchNormalization)
          self.Head1 = ConvolutionalLayer(32, 32, BatchNormalization=BatchNormalization)
          self.Head2 = SequentialConvolutionalLayer(32, 64 * Widening, Windowed=False, k=Granularity[0], **Settings)
          self.ShrinkingLayer = ConvolutionalLayer(64 * Widening, 64 * Widening, 3, 2, BatchNormalization=BatchNormalization, Groups=64 * Widening // Granularity[0])
          self.Stage1 = nn.ModuleList([ResidualBlock(64 * Widening, WeightScale=WeightScale, k=Granularity[0], **Settings) for _ in range(BlocksPerStage[0])])
          self.Downsample1 = DownsampleLayer(64 * Widening, Stage2Width * Widening, k=Granularity[0], Reduce=Granularity[1] // Granularity[0], **Settings)
          self.Stage2 = nn.ModuleList([ResidualBlock(Stage2Width * Widening, WeightScale=WeightScale, k=Granularity[1], **Settings) for _ in range(BlocksPerStage[1])])
          self.Downsample2 = DownsampleLayer(Stage2Width * Widening, Stage3Width * Widening, k=Granularity[1], Reduce=Granularity[2] // Granularity[1], **Settings)
          self.Stage3 = nn.ModuleList([ResidualBlock(Stage3Width * Widening, WeightScale=WeightScale, k=Granularity[2], **Settings) for _ in range(BlocksPerStage[2])])
          self.Downsample3 = DownsampleLayer(Stage3Width * Widening, Stage4Width * Widening, k=Granularity[2], Reduce=Granularity[3] // Granularity[2], **Settings)
          self.Stage4 = nn.ModuleList([ResidualBlock(Stage4Width * Widening, WeightScale=WeightScale, k=Granularity[3], **Settings) for _ in range(BlocksPerStage[3])])    
          self.Blender = ConvolutionalLayer(Stage4Width * Widening, Stage4Width * Widening, 1, 1, BatchNormalization=BatchNormalization)
          self.Compress = nn.AdaptiveAvgPool2d((1, 1))
          self.Classifier = RawFullyConnectedLayer(Stage4Width * Widening, Classes)         
      def forward(self, x):
          def Refine(ActivatedFeatures, RawFeatures, PoolOfBlocks):
              for Block in PoolOfBlocks:
                  RawFeatures, ActivatedFeatures = Block(ActivatedFeatures, RawFeatures)
              return RawFeatures, ActivatedFeatures   
          _, ActivatedFeatures = self.Head0(x)  
          _, ActivatedFeatures = self.Head1(ActivatedFeatures)  
          _, ActivatedFeatures = self.Head2(ActivatedFeatures)
          RawFeatures, ActivatedFeatures = self.ShrinkingLayer(ActivatedFeatures)
          RawFeatures, ActivatedFeatures = Refine(ActivatedFeatures, RawFeatures, self.Stage1)
          RawFeatures, ActivatedFeatures = self.Downsample1(ActivatedFeatures)
          RawFeatures, ActivatedFeatures = Refine(ActivatedFeatures, RawFeatures, self.Stage2)
          RawFeatures, ActivatedFeatures = self.Downsample2(ActivatedFeatures)
          RawFeatures, ActivatedFeatures = Refine(ActivatedFeatures, RawFeatures, self.Stage3)
          RawFeatures, ActivatedFeatures = self.Downsample3(ActivatedFeatures)   
          RawFeatures, ActivatedFeatures = Refine(ActivatedFeatures, RawFeatures, self.Stage4)   
          RawFeatures, ActivatedFeatures = self.Blender(ActivatedFeatures)
          x = self.Compress(ActivatedFeatures)
          x = x.view(x.size(0), -1)
          x = self.Classifier(x)
          return x    
      
ModelLite = ResNetImageNet(Classes=1000, BlocksPerStage=[1, 1, 2, 1], PyramidFactor=[1, 2, 4, 8], Widening=2, Granularity=[32, 64, 64, 128], BatchNormalization=True, WeightScale=0)  
ModelLarge = ResNetImageNet(Classes=1000, BlocksPerStage=[3, 4, 5, 3], PyramidFactor=[1, 2, 4, 8], Widening=2, Granularity=[32, 64, 64, 128], BatchNormalization=True, WeightScale=0)