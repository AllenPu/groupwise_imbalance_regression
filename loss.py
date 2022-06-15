class LAloss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super(LAloss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        iota_list = tau * np.log(cls_probs)

        self.iota_list = torch.cuda.FloatTensor(iota_list)

    def forward(self, x, target):
        output = x + self.iota_list

        return F.cross_entropy(output, target)



def forward(self, x, y, g):
    # x shape is N, W,H,C
    # y shape is N,1
    # g shape is N,1
    "output of model dim is 2G"
    y_hat = self.model(x)
    output_len = len(y_hat)
    #first G is cls
    g_hat = y_hat[: output_len/2]
    # add one to the cls num
    #self.cls_num_list[g] += 1
    # this is a preprocessed cls num list
    # add MSE
    if self.mode == 'train':
        loss_la = LAloss(self.cls_num_list, tau=1.0)
        # compute the group cls loss
        loss_ce = loss_la(g_hat, g)
        y_hat_index = output_len/2 + g
        yhat = y_hat[y_hat_index]
    else:
        loss_la = LAloss(self.cls_num_list, tau=0)
        # compute the group cls loss
        loss_ce = loss_la(g_hat, g)
        y_hat_index = output_len/2 + torch.argmax(g_hat, dim =1)
        if result_sum == True:
            yhat = y_hat[y_hat_index]
        else:
            #加权
        
    loss_mse = nn.MSELoss(yhat, y)
    # total loss
    loss = loss_ce + loss_mse
    return yhat, g_hat, loss
