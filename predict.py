import torch
import os
import matplotlib.pyplot as plt
import util
from dataset import CreateDatasetLoader
from model_newstyle import Unet_resize_conv
from collections import OrderedDict

class OPT(object):
    def __init__(self):
        self.batchSize=1
        self.phase='test'
        self.serial_batches= False
        self.nThreads= 4
        self.syn_norm= False
        self.no_flip = True

def load_network(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join('my_model', save_filename)
    network.load_state_dict(torch.load(save_path))


def display_results(value):
    dictlen = len(value)
    plt.figure(figsize=(15,5))
    plt.suptitle("disply_current_results")
    for i,key in enumerate(value):
        plt.subplot(1,dictlen,i+1)
        plt.title(key)
        plt.axis('off')
        plt.imshow(value[key])
    plt.show()

def predict(net,data):
    real_A = data['A'].cuda()
    real_A_gray = data['A_gray'].cuda()
    fake_B, latent_real_A = net.forward(real_A, real_A_gray)
    latent_real_A = util.tensor2im(latent_real_A.data)
    real_A = util.tensor2im(real_A.data)
    fake_B = util.tensor2im(fake_B.data)
    A_gray = util.atten2im(real_A_gray.data)
    return OrderedDict([('real_A', real_A), ('fake_B', fake_B),('A_gray',A_gray),('latent_real_A', latent_real_A)])

opt = OPT()
data_loader = CreateDatasetLoader(opt)
dataset = data_loader.load_data()
netG = Unet_resize_conv(opt)
load_network(netG,'G_A','200')

for i,data in enumerate(dataset):
    res = predict(netG,data)
    display_results(res)
