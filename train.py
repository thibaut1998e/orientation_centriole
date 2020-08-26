
import paths_definitions as pth

from fastai.vision import *

import matplotlib.pyplot as plt

from metrics import metrics

from models import *


def get_data(images_path, labels_txt, bs=4):

    with open(labels_txt) as f:
        lines = f.readlines()

    lines = [line.split('\t') for line in lines]
    label_dict = {lines[i][0] : int(lines[i][1][0]) for i in range(len(lines)) if len(lines[i])==2}

    def label_func(x):
        x = str(x)
        name = x.split('/')[-1]
        name = name.split('.')[0]
        #print(f'name : {x}, label {label_dict[name]}')
        return label_dict[name]

    data = ImageList.from_folder(images_path, convert_mode='L').split_by_folder().label_from_func(label_func)
    data = data.transform(get_transforms())
    data = data.databunch(bs=bs).normalize()
    return data


def make_learner(images_path, labels_txt, bs=4, nb_classes=2):
    data = get_data(images_path, labels_txt, bs)
    model = WNResNet(expansion=1, layers=[3,4,6,3], c_out=nb_classes)
    learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=metrics)
    print(learn.summary())
    return learn


def train(images_path, labels_path, save_location, nb_classes=2, bs=10, nb_epochs=1000, lr=1e-5):
    learn = make_learner(images_path, labels_path, bs=bs, nb_classes=nb_classes)
    learn = learn.to_fp16(loss_scale=1)
    learn.fit(epochs=nb_epochs, lr=lr)
    learn.export(save_location)
    learn.recorder.plot_losses()
    loss_fig_location = f'{pth.myHome}/cross_entropy_loss.jpg'
    plt.savefig(loss_fig_location)
    print(f'graph of losses saved at location {loss_fig_location}')
    plt.close()
    learn.recorder.plot_metrics()
    metric_fig = f'{pth.myHome}/accuracy.jpg'
    plt.savefig(metric_fig)
    print(f'metrics figure saved at location {metric_fig}')
    plt.close()









