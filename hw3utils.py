# These are utility functions / classes that you probably dont need to alter.

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def tensorshow(tensor,cmap=None):
    img = transforms.functional.to_pil_image(tensor/2+0.5)
    if cmap is not None:
        plt.imshow(img,cmap=cmap)
    else:
        plt.imshow(img)

class HW3ImageFolder(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root, device, test=False):
        super(HW3ImageFolder, self).__init__(root, transform=None)
        self.device = device
        self.test = test

    def prepimg(self,img):
        return (transforms.functional.to_tensor(img)-0.5)*2 # normalize tensorized image from [0,1] to [-1,+1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.
        
        ############################################################################################################
        In order to obtain the path of the image with index index you may use following piece of code. As dataset goes
        over different indices you will collect image paths.
        ############################################################################################################
        """
        if self.test:
            myfile = open('test_images.txt', 'a')
            path = self.imgs[index][0]
            myfile.write(path)
            myfile.write('\n')
            myfile.close()

        color_image,_ = super(HW3ImageFolder, self).__getitem__(index) # Image object (PIL)
        grayscale_image = torchvision.transforms.functional.to_grayscale(color_image)
        return self.prepimg(grayscale_image).to(self.device), self.prepimg(color_image).to(self.device)

def visualize_batch(inputs,preds,targets,save_path=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()
    plt.clf()
    bs = inputs.shape[0]
    for j in range(bs):
        plt.subplot(3,bs,j+1)
        assert(inputs[j].shape[0]==1)
        tensorshow(inputs[j],cmap='gray')
        plt.subplot(3,bs,bs+j+1)
        tensorshow(preds[j])
        plt.subplot(3,bs,2*bs+j+1)
        tensorshow(targets[j])
    if save_path != '':
        plt.savefig(save_path)
    else:
        plt.show(block=True)

def visualize_validation_batch(inputs,preds,targets,save_path=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()
    plt.clf()
    bs = 5
    for j in range(bs):
        plt.subplot(3,bs,j+1)
        assert(inputs[j].shape[0]==1)
        plt.axis('off')
        tensorshow(inputs[j],cmap='gray')
        plt.subplot(3,bs,bs+j+1)
        plt.axis('off')
        tensorshow(preds[j])
        plt.subplot(3,bs,2*bs+j+1)
        plt.axis('off')
        tensorshow(targets[j])
    if save_path != '':
        plt.savefig(save_path)
    else:
        plt.show(block=True)    

def visualize_best_batch(inputs,preds,targets,save_path1='',save_path2=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()
    plt.clf()
    bs = 5
    for j in range(bs):
        plt.subplot(3,bs,j+1)
        assert(inputs[j].shape[0]==1)
        plt.axis('off')
        tensorshow(inputs[j],cmap='gray')
        plt.subplot(3,bs,bs+j+1)
        plt.axis('off')
        tensorshow(preds[j])
        plt.subplot(3,bs,2*bs+j+1)
        plt.axis('off')
        tensorshow(targets[j])
    if save_path1 != '':
        plt.savefig(save_path1)
    else:
        plt.show(block=True)  
    plt.clf()
    for k in range(bs):
        plt.subplot(3,bs,k+1)
        assert(inputs[k+5].shape[0]==1)
        plt.axis('off')
        tensorshow(inputs[k+5],cmap='gray')
        plt.subplot(3,bs,bs+k+1)
        plt.axis('off')
        tensorshow(preds[k+5])
        plt.subplot(3,bs,2*bs+k+1)
        plt.axis('off')
        tensorshow(targets[k+5])     
    if save_path2 != '':
        plt.savefig(save_path2)
    else:
        plt.show(block=True)       

def visualize_test_batch(inputs,preds,save_path=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    plt.clf()
    bs = 5
    for i in range(20):
        for j in range(bs):
            plt.subplot(2,bs,j+1)
            assert(inputs[j + 5*i].shape[0]==1)
            plt.axis('off')
            tensorshow(inputs[j + 5*i],cmap='gray')
            plt.subplot(2,bs,bs+j+1)
            plt.axis('off')
            tensorshow(preds[j + 5*i])
        if save_path != '':
            plt.savefig('tests/pred_%d.png' % (i+1))
        else:
            plt.show(block=True)  
        plt.clf()

