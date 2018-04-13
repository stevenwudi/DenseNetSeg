import torch
from torch import nn
from torchvision.models import densenet


class DenseSeg(nn.Module):
    """
    This is a preload densenet121 model initialisation
    """

    def __init__(self, model_name, classes,
                 transition_layer=(3, 5, 7, 9, 11),
                 conv_num_features=(64, 256, 512, 1024, 1024),
                 pretrained=True):
        """
        :param model_name: DenseNet model name ['densenet121', 'densenet169', 'densenet201', 'densenet161']
        :param classes: number of classes for semantic segmantation (for CITYSCAPES is 19)
        :param pretrained_model: pretrained model
        :param transition_layer: layers for extracting skip connections
        :param conv_num_features: this list is dependent upon model_name and transition_layer used.
        :param pretrained: a flag for indicating whether to load base model
        """

        super(DenseSeg, self).__init__()
        model = densenet.__dict__.get(model_name)(pretrained=pretrained, num_classes=1000)

        # self.layer are the base densenet layers
        self.layer = list()
        self.layer.append(nn.Sequential(*list(model.features.children())[:transition_layer[0]]))

        # self.conv_layer are the newly added conv layers
        self.conv_layer = list()
        self.conv_layer.append(
            nn.Conv2d(conv_num_features[0], out_channels=1, kernel_size=1, stride=1, bias=True).cuda())

        for i in range(len(transition_layer) - 1):
            self.layer.append(
                nn.Sequential(*list(model.features.children())[transition_layer[i]:transition_layer[i + 1]]))
            self.conv_layer.append(
                nn.Conv2d(conv_num_features[i + 1], out_channels=1, kernel_size=1, stride=1, bias=True).cuda())

        # The following two lines are required to register to the model.state_dict()
        self.base_down = nn.ModuleList(self.layer)
        self.base_up = nn.ModuleList(self.conv_layer)
        # final conv layer for combining the multiple streams
        self.seg = nn.Conv2d(len(transition_layer), classes, kernel_size=1, bias=True)

    def forward(self, x):
        y = list()
        for i, layer in enumerate(self.layer):
            x = layer(x)
            x_conv = self.conv_layer[i](x)
            y.append(nn.Upsample(scale_factor=2 ** i, mode='bilinear')(x_conv))

        y = torch.cat(y, dim=1)
        y = self.seg(y)
        # The final resolution needs to upsample again
        y = nn.Upsample(scale_factor=2, mode='bilinear')(y)
        y_out = nn.LogSoftmax(dim=1)(y)

        return y_out

    def optim_parameters(self):
        for layer in self.layer:
            for param in layer.parameters():
                yield param

        for layer in self.conv_layer:
            for param in layer.parameters():
                yield param

        for param in self.seg.parameters():
            yield param
