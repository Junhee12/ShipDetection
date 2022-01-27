import numpy as np
import matplotlib.pyplot as plt

folder = 'train_4'

train_path = 'logs/' + folder + '/epoch_loss_2021_12_24_16_28_55.txt'
valid_path = 'logs/' + folder + '/epoch_val_loss_2021_12_24_16_28_55.txt'

f_train = open(train_path, 'r')
train_loss = f_train.readlines()
train_loss = np.array(train_loss).astype(float)

f_valid = open(valid_path, 'r')
valid_loss = f_valid.readlines()
valid_loss = np.array(valid_loss).astype(float)

# find smallest valid loss in the list
valid_min_index = valid_loss.argmin()
valid_min_value = valid_loss[valid_min_index]
print('Valid min : %f(%d) / %f' % (valid_min_value, valid_min_index, train_loss[valid_min_index]))
pprint
epochs = range(len(train_loss))

plt.figure()
plt.axis([0, len(train_loss), 0, 30])
#plt.yticks([])

plt.plot(epochs, train_loss, 'red', linewidth=1, label='train loss', linestyle = '--')
plt.plot(epochs, valid_loss, 'green', linewidth=1, label='valid loss', linestyle = '--')
#plt.plot(x, y, 'red', linewidth=2, label='train loss')


plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.show()

f_train.close()

