import torch.nn as nn
import torchvision
import torch


class ResNet_regression(nn.Module):
    def __init__(self, args):
        super(ResNet_regression, self).__init__()
        self.groups = args.groups
        self.model = torchvision.models.resnet18(pretrained=False)
        output_dim = args.groups*2
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Module(*list(self.model.children())[:-1])
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #
        '''
        self.model.fc = nn.Sequential(
            #nn.Linear(fc_inputs, 1024),
            #nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(fc_inputs, output_dim)
        )
        '''

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, z


class ResNet_ordinal_regression(nn.Module):
    def __init__(self, args):
        super(ResNet_regression, self).__init__()
        self.groups = args.groups
        self.model = torchvision.models.resnet18(pretrained=False)
        output_dim = args.groups
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Module(*list(self.model.children())[:-1])
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #
        self.fc_layers = []
        #
        for i in range(self.groups):
            exec('self.FC2_{}=nn.Linear(self.fc_inputs,2)'.format(i))
            exec('self.fc_layers.append(self.FC2_{})'.format(i))
        #
        self.softmax = nn.Softmax(dim=2)
        '''
        self.model.fc = nn.Sequential(
            #nn.Linear(fc_inputs, 1024),
            #nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(fc_inputs, output_dim)
        )
        '''

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #
        z = self.model_extractor(x)
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
        return y_hat, z, out
