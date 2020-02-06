import torch as t
import numpy as np
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from evaluation import create_evaluation

class Trainer:
    def __init__(self,               
                 model,                # Model to be trained.
                 crit,                 # Loss function
                 optim = None,         # Optimiser
                 train_dl = None,      # Training data set
                 val_test_dl = None,   # Validation (or test) data set
                 cuda = True,          # Whether to use the GPU
                 early_stopping_cb = None): # The stopping criterion. 
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_cb = early_stopping_cb
        
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, '/proj/ciptmp/xi01cyki/checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('/proj/ciptmp/xi01cyki/checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x=0, y=0):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        batch_loss = 0; b = 0
        #TODO
        self._model.train()
        #dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
        for batch, data in enumerate(self._train_dl): # in dataloader

            self._optim.zero_grad()
            x = data[0]
            y_true = data[1]
            if self._cuda:
                x = x.cuda()
                y_true = y_true.cuda()
            y_pred = self._model(x)
            l = self._crit(y_pred, y_true)
            l.backward()
            batch_loss += l
            self._optim.step()
            b+=1
        return batch_loss/b
    
    def val_test_step(self, x=0, y=0):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        l = 0; b=0
        y_pred_all = []
        y_true_all = []
        self._model.eval() # this sets the mode of the of the environment and it will not attache or compute gradients
        for batch, data in enumerate(self._val_test_dl):
            x = data[0]
            y_true = data[1]
            if self._cuda:
                x = x.cuda()
                y_true = y_true.cuda()
            with t.no_grad():
                y_pred = self._model(x)
                y_pred = t.sigmoid(y_pred)
                l += self._crit(y_pred, y_true)
            y_pred_all.extend(y_pred.cpu().numpy())
            y_true_all.extend(y_true.cpu().numpy())
            b+=1
        return l/b, np.array(y_pred_all), np.array(y_true_all)
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set

        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        return self.train_step()


    
    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        loss, y_pred, y_true = self.val_test_step()

        specificity, sensitivity, mean_f1_score = create_evaluation(y_true, y_pred, classification_threshold = 0.5)
        return loss, [specificity, sensitivity, mean_f1_score]
    
    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        early_stopping_reached = False
        epoch_counter = 0

        final_train_loss = []
        final_val_loss = []

        print("Started Training")
        while (early_stopping_reached != True) and (epoch_counter != epochs):
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            train_loss = self.train_epoch()
            final_train_loss.append(train_loss)
            validation_loss, evaluations = self.val_test()
            final_val_loss.append(validation_loss)

            # use the save_checkpoint function to save the model for each epoch
            self.save_checkpoint(epoch_counter)

            print("Epoch: ", epoch_counter, " train_loss:", train_loss.detach().cpu().numpy(), " validation_loss:",
                  validation_loss.cpu().numpy(), " validation_specificity[Class-0, Class-1]:", evaluations[0],
                  " validation_sensitivity[Class-0, Class-1]:", evaluations[1], " validation_mean_f1_score:", evaluations[2])
            # check whether early stopping should be performed using the early stopping callback and stop if so
            # When you implement stopping.py than should_stop function will return True if criteria is reached.
            early_stopping_reached = self._early_stopping_cb.step(validation_loss.detach().cpu().numpy())
            # stop by epoch number
            epoch_counter += 1


        if early_stopping_reached:
            print("Training Stopped: Early Stopping Patience Reached")
        elif epoch_counter == epochs:
            print("Training Stopped: Max epochs")
        else:
            print("Training Stopped")
        # return the loss lists for both training and validation
        self.save_onnx(fn = '/proj/ciptmp/xi01cyki/model/save_model.onnx')
        return final_train_loss, final_val_loss

                    
        
        
        