import torch.nn as nn
import torchvision


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
        return y_hat, z
        '''
        output_len = y_hat.shape[1]
        #first G is cls
        g_hat = y_hat[: , : int(output_len/2)]
        if mode == 'train':
            g_len = int(output_len/2)
            g_index = g + g_len
            yhat = torch.gather(y_hat, dim = 1, index = g_index)
            #loss_mse = self.loss_mse(yhat, y)
            #loss = loss_mse + self.sigma *loss_ce
            return yhat, g_hat
        else:
            #if self.output_strategy == 1:
            y_hat_index = output_len/2 + torch.argmax(g_hat, dim =1).unsqueeze(-1)
            yhat_1 = torch.gather(y_hat, dim = 1, index = y_hat_index)
            yhat_2 = torch.mean(y_hat[:, int(output_len/2):], dim =1).unsqueeze(-1)
            # shape of the output is:
            #       yhat_1 : (256,1)
            #       yhat_2 : (256,1)
            #       g_hat : (256,10)
            return yhat_1, yhat_2, g_hat
        '''
