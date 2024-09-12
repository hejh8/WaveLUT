import logging
from collections import OrderedDict

import torch

from torch.nn.parallel import DataParallel, DistributedDataParallel
from .base_model import BaseModel

logger = logging.getLogger('base')
import torch
import models.WaveLUT.WaveLUT_model as enhance_model


def define_G(conf):
    conf_net = conf['network_G']
    which_model = conf_net['which_model_G']

    if which_model == 'WaveLUT_LLVE':
        if conf['datasets'].get('test', None):
            netG = enhance_model.WaveLUT_LLVE(input_resolution=conf_net['input_resolution'],
                                              train_resolution=conf_net['input_resolution'],
                                              n_ranks=conf_net['n_ranks'],
                                              n_vertices_4d=conf_net['n_vertices_4d'],
                                              n_base_feats=conf_net['n_base_feats'],)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


class WaveLUT_network(BaseModel):
    def __init__(self, conf):
        super(WaveLUT_network, self).__init__(conf)

        if conf['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.lut_dim = conf['network_G']['n_vertices_4d']

        # define network and load pretrained models
        self.netG = define_G(conf).to(self.device)
        if conf['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # load pretrained model
        self.load_path_G = conf['path']['pretrain_model_G']
        if conf['path']['strict_load'] == None:
            self.strict_load = True
        else:
            self.strict_load = False
        self.load()


    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)

        # n,c,t,h,w=self.var_L.shape
        # self.L_dwt = self.dwt(self.var_L)
        # self.LL, self.LH = self.L_dwt[:n, ...],self.L_dwt[n:, ...]
        
        if need_GT:
            self.real_H = data['GTs'].to(self.device)


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.outputs = self.netG(self.var_L, if_train=False)
        # self.LH = self.he(self.LH)
        # self.outputs = self.iwt(torch.cat((self.outputs,self.LH),dim=0))
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQs'] = self.var_L.detach()[0].float().cpu()
        
        out_dict['rlts'] = self.outputs.detach()[0].float().cpu()
        # out_dict['rlts'] = self.outputs.detach()[0].float().cpu()
        if need_GT:
            out_dict['GTs'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def load(self):
        load_path_G = self.load_path_G
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.strict_load)
