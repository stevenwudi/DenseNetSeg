import torch
from torch import nn
from models import densenet_local


class DenseSeg(nn.Module):
    """
    This is a preload densenet121 model initialisation
    """

    def __init__(self, model_name, classes,
                 transition_layer=(3, 5, 7, 9, 11),
                 conv_num_features=(64, 256, 512, 1024, 1024),
                 out_channels_num=(1, 2, 4, 8, 16),
                 ppl_out_channels_num=(32, 64, 128, 256),
                 pretrained=False):
        """
        :param model_name: DenseNet model name ['densenet121', 'densenet169', 'densenet201', 'densenet161']
        :param classes: number of classes for semantic segmantation (for CITYSCAPES is 19)
        :param pretrained_model: pretrained model
        :param transition_layer: layers for extracting skip connections
        :param conv_num_features: this list is dependent upon model_name and transition_layer used.
        :param pretrained: a flag for indicating whether to load base model
        """

        super(DenseSeg, self).__init__()
        model = densenet_local.__dict__.get(model_name)(pretrained=pretrained, num_classes=1000)

        # self.layer are the base densenet layers
        self.base_layer = list()
        self.base_layer.append(nn.Sequential(*list(model.features.children())[:transition_layer[0]]))

        # self.conv_layer are the newly added conv layers
        self.conv_layer = list()
        self.conv_layer.append(self._make_conv_layers(conv_num_features[0], out_channels=out_channels_num[0],
                                                      kernel_size=1, stride=1, bias=True, padding=0))

        for i in range(len(transition_layer) - 1):
            self.base_layer.append(
                nn.Sequential(*list(model.features.children())[transition_layer[i]:transition_layer[i + 1]]))
            self.conv_layer.append(
                self._make_conv_layers(in_channels=conv_num_features[i + 1], out_channels=out_channels_num[i+1],
                                       kernel_size=1, stride=1, bias=True, padding=0))

        # Pyramid Pooling Module, after this, we will have a receptive field of 512
        self.pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pyramid_pooling_layer = list()
        for i in range(len(ppl_out_channels_num)):
            self.pyramid_pooling_layer.append(self._make_conv_layers(conv_num_features[-1], ppl_out_channels_num[i], kernel_size=1, dilation=1, padding=0))

        # The following two lines are required to register to the model.state_dict()
        self.base_down = nn.ModuleList(self.base_layer)
        self.base_up = nn.ModuleList(self.conv_layer)
        self.base_ppl = nn.ModuleList(self.pyramid_pooling_layer)
        # final conv layer for combining the multiple streams
        self.seg = nn.Conv2d(sum(out_channels_num)+sum(ppl_out_channels_num), classes, kernel_size=1, bias=True)

    def forward(self, x):
        y = list()
        for i, layer in enumerate(self.base_layer):
            x = layer(x)
            x_conv = self.conv_layer[i](x)
            y.append(nn.Upsample(scale_factor=2 ** i, mode='bilinear')(x_conv))

        # pyramid pooling moduel
        resize_ratio = len(self.base_layer)
        for i, layer in enumerate(self.pyramid_pooling_layer):
            x = self.pool_layer(x)
            x_conv = layer(x)
            y.append(nn.Upsample(scale_factor=2 ** (resize_ratio+i), mode='bilinear')(x_conv))

        y = torch.cat(y, dim=1)
        y = self.seg(y)
        # The final resolution needs to upsample again
        y = nn.Upsample(scale_factor=2, mode='bilinear')(y)
        y_out = nn.LogSoftmax(dim=1)(y)

        return y_out

    def _make_conv_layers(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=False, padding=0):
        modules = []
        modules.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride, bias=bias, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)])
        return nn.Sequential(*modules).cuda()

    def optim_parameters(self):
        #TODO: different fine-tune scheme
        for layer in self.base_layer:
            for param in layer.parameters():
                yield param

        for layer in self.conv_layer:
            for param in layer.parameters():
                yield param

        for param in self.seg.parameters():
            yield param

        for layer in self.pyramid_pooling_layer:
            for param in layer.parameters():
                yield param
