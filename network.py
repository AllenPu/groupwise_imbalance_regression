import torch.nn as nn
import torchvision
import torch


class ResNet_regression(nn.Module):
    def __init__(self, args):
        super(ResNet_regression, self).__init__()
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = args.groups * 2
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, z


class ResNet_ordinal_regression(nn.Module):
    def __init__(self, args):
        super(ResNet_ordinal_regression, self).__init__()
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        output_dim = args.groups
        #
        fc_inputs = self.model.fc.in_features
        #
        #print(" shape is ", fc_inputs)
        #
        #print(len(list(self.model.children())[:-1]))
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.model_linear =  nn.Linear(fc_inputs, output_dim)
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.fc_layers = []
        #
        self.output_dim = args.output_dim
        #
        for i in range(self.groups):
            exec('self.FC2_{}=nn.Linear(fc_inputs,{})'.format(i, self.output_dim))
            exec('self.fc_layers.append(self.FC2_{})'.format(i))
        #
        self.softmax = nn.Softmax(dim=2)

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        out = self.softmax(self.fc_layers[0](z).unsqueeze(1))
        for i in range(1, self.groups):
            result = self.softmax(self.fc_layers[i](z).unsqueeze(1))
            out = torch.cat((out, result), 1)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, out



class ResNet_regression_sep(nn.Module):
    def __init__(self, args):
        super(ResNet_regression_sep, self).__init__()
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = args.output_dim
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat


class ResNet_regression_split(nn.Module):
    def __init__(self, args):
        super(ResNet_regression_split, self).__init__()
        #
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = args.groups
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.linear_cls =  nn.Linear(fc_inputs, output_dim)
        #
        self.linear_reg = nn.Linear(fc_inputs, output_dim)
        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.linear_cls(z)
        #
        g_hat = self.linear_reg(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, g_hat

class ResNet_regression_ddp(nn.Module):
    def __init__(self, args):
        super(ResNet_regression_ddp, self).__init__()
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = args.groups * 2
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x, g):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_predicted = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        y_chunk = torch.chunk(y_predicted, 2, dim = 1)
        #
        g_hat, y_hat_all = y_chunk[0], y_chunk[1]
        #
        y_hat = torch.gather(y_hat_all, dim = 1, index = g.to(torch.int64))
        #
        return g_hat, y_hat
