import torch
import torch.nn as nn

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# only copy backbone
def load_pretrained_weight(model, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = model.backbone


    # use pth
    if model_path.endswith('pth'):
        #model_dict = model.state_dict()
        model_dict = backbone.state_dict()

        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_names = list(pretrained_dict.keys())

        # 1. filter out unnecessary keys
        for i, param in enumerate(model_dict):
            if model_dict[param].shape != pretrained_dict[pretrained_names[i]].shape:
                print('%d error', i)
                exit()

            print(i, ' ', model_dict[param].numel(), ' ', param, ' : ', model_dict[param].shape, '-----',
                  pretrained_names[i], pretrained_dict[pretrained_names[i]].shape)

            model_dict[param] = pretrained_dict[pretrained_names[i]]

        #model.load_state_dict(model_dict)
        backbone.load_state_dict(model_dict)

    else:

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        """
        model_dict = model.state_dict()

        pretrained_dict = torch.load(model_path, map_location=device)

        # 1. filter out unnecessary keys
        cnt = 0
        for k, v in pretrained_dict.items():
            if np.shape(model_dict[k]) == np.shape(v):
                pretrained_dict = {k: v}
                cnt += 1

            if np.shape(model_dict[k]) != np.shape(v):
                print('error count : ', cnt)
                exit()

        print('count : ', cnt)
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        """