# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
torch.multiprocessing.set_start_method('spawn', force=True)
torch.manual_seed(0)

## These are the directories where I saved MSE loss plot, 12-margin error plot, model states and predicted images.
# LOG_DIR = 'all'
# LOG_DIR = 'part1_1'
# LOG_DIR = 'part1_2'
# LOG_DIR = 'part1_3'
# LOG_DIR = 'part2_1'
# LOG_DIR = 'part2_2'
# LOG_DIR = 'part2_3'
LOG_DIR = 'best'

VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

## These are the text files that I wrote the model performance results.
# model_specs = open("all.txt", "w")
# model_specs2 = open("part1_1.txt", "w")
# model_specs3 = open("part1_2.txt", "w")
# model_specs4 = open("part1_3.txt", "w")
# model_specs5 = open("part2_1.txt", "w")
# model_specs6 = open("part2_2.txt", "w")
# model_specs7 = open("part2_3.txt", "w")
model_specs8 = open("best.txt", "w")
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 16
max_num_epoch = 100
## These are the hyperparameters that I experimented. Thus, I trained 45 models to examine the results.
# layer_numbers = [1, 2, 4]
# kernel_numbers = [2, 4, 8]
# learning_rates = [0.0001, 0.001, 0.01, 0.1, 10]
# models = []

def get_loaders(batch_size,device):
    data_root = 'ceng483-hw3-dataset' 
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

## This is the CNN architecture.
class Net(nn.Module):
    def __init__(self, conv_layer_number, kernel_number, lr, batch_norm=False, tanh_func=False):
        super(Net, self).__init__()
        self.convolutional_layers = conv_layer_number
        self.kernel_number = kernel_number
        self.learning_rate = lr
        self.batch = batch_norm
        self.tanh = tanh_func
        if conv_layer_number == 1:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
            if batch_norm:
                self.batch_normalization = nn.BatchNorm2d(3)
        elif conv_layer_number == 2:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernel_number, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=kernel_number, out_channels=3, kernel_size=3, padding=1)
            if batch_norm:
                self.batch_normalization1 = nn.BatchNorm2d(kernel_number)
                self.batch_normalization2 = nn.BatchNorm2d(3)
        elif conv_layer_number == 4:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernel_number, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=kernel_number, out_channels=kernel_number, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(in_channels=kernel_number, out_channels=kernel_number, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(in_channels=kernel_number, out_channels=3, kernel_size=3, padding=1)
            if batch_norm:
                self.batch_normalization1 = nn.BatchNorm2d(kernel_number)
                self.batch_normalization2 = nn.BatchNorm2d(kernel_number) 
                self.batch_normalization3 = nn.BatchNorm2d(kernel_number)
                self.batch_normalization4 = nn.BatchNorm2d(3)   

    def forward(self, grayscale_image):
        if self.convolutional_layers == 1:
            x = self.conv1(grayscale_image)
            if self.batch:
                x = self.batch_normalization(x)
        elif self.convolutional_layers == 2:
            x = self.conv1(grayscale_image)
            if self.batch:
                x = self.batch_normalization1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.batch_normalization2(x)
            else:
                x = F.relu(x)
                x = self.conv2(x)
        elif self.convolutional_layers == 4:
            x = self.conv1(grayscale_image)
            if self.batch:
                x = self.batch_normalization1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.batch_normalization2(x)
                x = F.relu(x)
                x = self.conv3(x)
                x = self.batch_normalization3(x)
                x = F.relu(x)
                x = self.conv4(x)
                x = self.batch_normalization4(x)
            else:
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = self.conv3(x)
                x = F.relu(x)
                x = self.conv4(x)
        if self.tanh:
            x = F.tanh(x)
        return x
    
def train(model):
    if model.batch:
        batch = 1
    else:
        batch = 0
    if model.tanh:
        tanh = 1
    else:
        tanh = 0

    global best_model_name # best model name string for loading it after to create .npy file

    ## Model specifications to write to text files
    # model_specs.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")
    # model_specs2.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")
    # model_specs3.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")
    # model_specs4.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")
    # model_specs5.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")
    # model_specs6.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")
    # model_specs7.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")
    model_specs8.write("Model specs for model: convolutional layer numbers = " + str(model.convolutional_layers) + ", kernel numbers = " + str(model.kernel_number) + ", learning_rate = " + str(model.learning_rate) + ", batch normalization = " + str(model.batch) + ", tanh function = " + str(model.tanh) + "\n")

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=model.learning_rate, weight_decay=0.01) I trained with different regularization strengths using L2 regularization in some cases. Details are on the report.
    train_loader, val_loader = get_loaders(batch_size,device)

    # best_validation_loss = float('inf') Instead of using MSE loss, 12-margin error is used for tuning epoch number. Hence, this is commented.
    # patience = 5
    # counter = 0

    ## These are for the plots.
    training_losses = []
    validation_losses = []
    validation_errors = []

    ## 12-margin error over validation set is used for tuning epoch number. Early break out algorithm uses 5 for error patience. If counter reaches to 5, training stops.
    best_validation_error = float('inf') 
    error_patience = 5
    error_counter = 0

    for epoch in range(max_num_epoch):  
        training_loss = 0.0
        for iteri, data in enumerate(train_loader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()    
            training_loss += loss.item()

        training_loss /= len(train_loader)

        ## Write training MSE loss at some epoch to text files.
        # model_specs.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")
        # model_specs2.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")
        # model_specs3.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")
        # model_specs4.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")
        # model_specs5.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")
        # model_specs6.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")
        # model_specs7.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")
        model_specs8.write("For epoch: " + str(epoch+1) + " training loss = " + str(training_loss) + "\n")

        training_losses.append(training_loss)

        ## Save the model at some epoch.
        # torch.save(model.state_dict(), os.path.join(LOG_DIR,'model_layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.pt' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch + 1)))
        torch.save(model.state_dict(), os.path.join(LOG_DIR,'model_epoch_%d.pt' % (epoch + 1)))

        with torch.no_grad():
            validation_loss = 0.0
            twelve_margin_error = 0.0
            for iteri, data in enumerate(val_loader, 0):
                inputs, targets = data
                preds = model(inputs)
                loss = criterion(preds, targets)
                validation_loss += loss.item()
                twelve_margin_error += torch.mean((torch.abs(preds - targets) >= 24/255).float()) # 12-margin error. Since the original error margin is 12 in [0, 255] scale, it is 24/255 (0.0941) in [-1, 1] scale.

            validation_loss /= len(val_loader)

            ## Write validation MSE loss at some epoch to text files.
            # model_specs.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")
            # model_specs2.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")
            # model_specs3.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")
            # model_specs4.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")
            # model_specs5.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")
            # model_specs6.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")
            # model_specs7.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")
            model_specs8.write("For epoch: " + str(epoch+1) + " validation loss = " + str(validation_loss) + "\n")

            validation_losses.append(validation_loss)
            twelve_margin_error /= len(val_loader)
            twelve_margin_error = twelve_margin_error.cpu().numpy()
            
            ## Write validation 12-margin error at some epoch to text files.
            # model_specs.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")
            # model_specs2.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")
            # model_specs3.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")
            # model_specs4.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")
            # model_specs5.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")
            # model_specs6.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")
            # model_specs7.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")
            model_specs8.write("For epoch: " + str(epoch+1) + " 12-margin error = " + str(twelve_margin_error) + "\n")

            validation_errors.append(twelve_margin_error)

        ## These are for saving predicted images on the validation set. First one saves 10 images. Second one saves 5 images.
        # hw3utils.visualize_best_batch(inputs,preds,targets,os.path.join(LOG_DIR,'model_part1_layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.png' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch+1)),os.path.join(LOG_DIR,'model_part2_layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.png' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch+1)))
        # hw3utils.visualize_validation_batch(inputs,preds,targets,os.path.join(LOG_DIR,'layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.png' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch+1)))

        ## Early break out algorithm. If the last 12-margin error is less than best 12-margin error, then best 12-margin error becomes the last and counter resets to 0.
        ## Else, counter is incremented by 1. Then, if the counter reaches 5, training stops (early stopped).
        if twelve_margin_error < best_validation_error:
            best_validation_error = twelve_margin_error
            error_counter = 0
        else:
            error_counter += 1
        if error_counter == error_patience:
            break

    ## If counter reaches 5, pop last 5 elements from the lists training MSE losses, validation MSE losses and validation 12-margin errors since the automatically chosen
    ## number of epochs becomes the last epoch - 5. Hence, draw plots accordingly.
    if error_counter == error_patience:
        for i in range(error_patience):
            training_losses.pop()
            validation_losses.pop()
            validation_errors.pop()
        best_model_name = 'model_epoch_%d.pt' % (epoch - 4) # Best model's automatically chosen number of epochs is last epoch - 5. Since epoch numbers are saved like epoch + 1 (to start with 1), for loading model state, epoch - 4 is used.

    else:
        best_model_name = 'model_epoch_100.pt' # If no early break, the model's epoch number is 100.

    ## Drawing plots. First plot is training set MSE loss over epochs. Second plot is validation set MSE loss over epochs. Third plot is validation set 12-margin error over epochs. Fourth plot is MSE loss over training set and validation set together.
    # fig, ax = plt.subplots()
    # ax.set(xlabel='Epoch', ylabel='MSE Loss', title='Training Mean-Squared Error Loss')
    # ax.plot(np.arange(1, len(training_losses) + 1), training_losses)
    # fig.savefig(os.path.join(LOG_DIR,'training_losses_layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.png' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch + 1)))
    # plt.clf()
    # fig2, ax2 = plt.subplots()
    # ax2.set(xlabel='Epoch', ylabel='MSE Loss', title='Validation Mean-Squared Error Loss')
    # ax2.plot(np.arange(1, len(validation_losses) + 1), validation_losses)
    # fig2.savefig(os.path.join(LOG_DIR,'validation_losses_layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.png' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch + 1)))
    # plt.clf()
    # fig3, ax3 = plt.subplots()
    # ax3.set(xlabel='Epoch', ylabel='12-Margin Error', title='Validation 12-Margin Error')
    # ax3.plot(np.arange(1, len(validation_errors) + 1), validation_errors)
    # fig3.savefig(os.path.join(LOG_DIR,'validation_errors_layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.png' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch + 1)))
    # plt.clf()  
    # fig4, ax4 = plt.subplots()
    # ax4.plot(np.arange(1, len(training_losses) + 1), training_losses, label='training MSE loss')
    # ax4.plot(np.arange(1, len(validation_losses) + 1), validation_losses, label='validation MSE loss')
    # plt.legend()
    # fig4.savefig(os.path.join(LOG_DIR,'training_validation_losses_layer_%d_kernel_%d_lr_%f_batchnorm_%d_tanh_%d_epoch_%d.png' % (model.convolutional_layers, model.kernel_number, model.learning_rate, batch, tanh, epoch + 1)))  
    # plt.clf()       
       
    # model_specs.write("\n")
    # model_specs2.write("\n")
    # model_specs3.write("\n")
    # model_specs4.write("\n")
    # model_specs5.write("\n")
    # model_specs6.write("\n")
    # model_specs7.write("\n")
    model_specs8.write("\n")        

# for layer in layer_numbers:
#     for kernel in kernel_numbers:
#         for lr in learning_rates:
#             models.append(Net(layer, kernel, lr).to(device=device))

# for model in models:
#     train(model)

### First part:
## Discuss effect of number of conv layers: conv layers = [1,2,4], number of kernels = 4, lr = 0.1
# train(Net(1, 4, 0.1).to(device=device))
# train(Net(2, 4, 0.1).to(device=device))
# train(Net(4, 4, 0.1).to(device=device))

## Discuss effect of number of kernels: conv layers = 2, number of kernels = [2, 4, 8], lr = 0.1
# train(Net(2, 2, 0.1).to(device=device))
# train(Net(2, 4, 0.1).to(device=device))
# train(Net(2, 8, 0.1).to(device=device))

## Discuss effect of learning rate: conv layers = 2, number of kernels = 8, lr = [10, 0.0001, 0.1]
# train(Net(2, 8, 10).to(device=device))
# train(Net(2, 8, 0.0001).to(device=device))
# train(Net(2, 8, 0.1).to(device=device))

### Second part:
## Try add batch-norm layer
# train(Net(2, 8, 0.1, batch_norm=True).to(device=device))

## Try add tanh activation
# train(Net(2, 8, 0.1, tanh_func=True).to(device=device))    

## Try setting number of channels to 16
# train(Net(2, 16, 0.1).to(device=device))

### Third part:
train(Net(2, 16, 0.1).to(device=device))

# model_specs.close()
# model_specs2.close()
# model_specs3.close()
# model_specs4.close()
# model_specs5.close()
# model_specs6.close()
# model_specs7.close()
model_specs8.close()

torch.manual_seed(1)

## This part is for creating 'estimations_test.npy' file. 
# if not os.path.exists("tests"): # create directory for saving predicted test images
#     os.makedirs("tests")    
print(best_model_name)  
best_model = Net(2, 16, 0.1).to(device=device)
best_model.load_state_dict(torch.load(os.path.join(LOG_DIR, best_model_name))) # load the best model parameters
# best_model.load_state_dict(torch.load(os.path.join(LOG_DIR,'model_epoch_35.pt')))
best_model.eval()
test_set = hw3utils.HW3ImageFolder(root='test', device=device, test=True) # change root if necessary
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=0) # choose 100 images from the test set
data_iter = iter(test_loader)
inputs, _ = next(data_iter)
outputs = best_model(inputs)
# print(outputs.size())
# hw3utils.visualize_test_batch(inputs, outputs, 'yes') # save all 100 predicted test images
outputs = (outputs + 1) * 127.5 # scale back to [0, 255]
predictions = torch.permute(outputs, (0, 2, 3, 1)) # reshape from (100, 3, 80, 80) to (100, 80, 80, 3)
predictions = predictions.detach().cpu().numpy()
print(predictions.shape)
np.save('estimations_test.npy', predictions)