import torch as t
import torchvision as tv
from data import get_train_dataset, get_validation_dataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import resnet
import torch.utils.data

# set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py
#

train_data=get_train_dataset()
valid_data=get_validation_dataset()
t_data = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

v_data = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True, num_workers=2)

# set up your model
my_model = resnet.Model()

# set up loss (you can find preimplemented loss functions in t.nn) use the pos_weight parameter to ease convergence
loss = t.nn.BCEWithLogitsLoss()
# set up optimizer (see t.optim);
optim = t.optim.Adam(my_model.parameters(), lr=0.001)
# initialize the early stopping callback implemented in stopping.py and create a object of type Trainer
early_stopping_callback = EarlyStoppingCallback()

# go, go, go... call fit on trainer
trainer = Trainer(my_model, loss, optim, t_data, v_data, cuda=True, early_stopping_cb=early_stopping_callback)
res = trainer.fit(epochs=500)
# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')