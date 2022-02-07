import torch

path = 'result/last.pt'

ckpt = torch.load(path)  # load checkpoint

optimizer = ckpt['optimizer']
history = ckpt['loss_history']
epoch = ckpt['epoch']
model = ckpt['model']

print('epoch : %d' % epoch)

del ckpt


