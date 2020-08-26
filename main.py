from train import train
from inference import *


"""train and test a classifier"""

if __name__ == '__main__':
    bs = 10 #batch size
    nb_epochs = 1000
    lr = 1e-5 #learning rate
    nb_classes = 2

    images_path = f'{pth.training_sets}/centriole_orientation/sets' #path of training images, should contains 2
    # sub-folder train and valid
    labels_path = f'{pth.training_sets}/centriole_orientation/label2.txt' #txt file with labels

    save_name = 'centriole_orientation_best.pkl'
    save_location = f'{pth.models}/{save_name}'

    txt_name = 'results'
    results_txt = f'{pth.training_sets}/centriole_orientation/{txt_name}.txt'

    train(images_path, labels_path, save_location, bs=bs, nb_epochs=nb_epochs, lr=lr, nb_classes=nb_classes)
    inference(save_name, f'{images_path}/valid', labels_path, results_txt)

