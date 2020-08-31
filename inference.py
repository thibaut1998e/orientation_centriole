from fastai.vision import *
import paths_definitions as pth
import os
from skimage import io
from models import WNResNet


def inference(model_name, test_data, labels_txt, txt_save_location):
    """"
    test the classification model and store the results in a txt file

    model_name, str model_name with extension, should be located in models folder.
    test_data : str, path of the data that we want to make the prediction on
    label_txt : true labels, written in the result file to compare results quickly
    txt_save_location : path of txt file to save"""
    defaults.device = torch.device('cpu')
    learn = load_learner(pth.models, model_name).to_fp32()
    dict_results = {}
    for name in os.listdir(test_data):
        path = f'{test_data}/{name}'
        im_array = np.array(io.imread(path))
        out = prediction(learn, im_array)
        dict_results[name.split('.')[0]] = out

    #print(dict_results)
    with open(labels_txt) as f:
        lines = f.readlines()
        lines = [line.split('\t') for line in lines]

    with open(txt_save_location, 'w') as f:
        f.write('image name\t\t\t\tprediction\ttrue label\n')
        for line in lines:
            if line[0] in dict_results.keys():
                probas = dict_results[line[0]]
                pred_class = np.argmax(probas)
                to_write = f'{line[0]}\t{pred_class}\t{line[1][0]}\n'
                f.write(to_write)
    print(f'results saved at location : {txt_save_location}')


def prediction(learn, in_img):
    """uses the learner learn to make a prediction on in_img, the output is a n-dimensional vector with the probabilities
    for each classes (n classes)"""
    in_img = (in_img - np.mean(in_img))/np.std(in_img)
    in_img = tensor(in_img)
    out = learn.model(in_img[None][None])[0]
    return out