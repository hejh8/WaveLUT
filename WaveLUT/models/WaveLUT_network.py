import logging
from collections import OrderedDict
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
# from .base_model import BaseModel
logger = logging.getLogger('base')
import torch
import models.WaveLUT.WaveLUT as enhance_model
import os
import torch.nn as nn



class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    # def optimize_parameters(self):
    #     pass

    def get_current_visuals(self):
        pass

    # def get_current_losses(self):
    #     pass

    def print_network(self):
        pass

    # def save(self, label):
    #     pass

    def load(self):
        pass

    # def _set_lr(self, lr_groups_l):
    #     """Set learning rate for warmup
    #     lr_groups_l: list for lr_groups. each for a optimizer"""
    #     for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
    #         for param_group, lr in zip(optimizer.param_groups, lr_groups):
    #             param_group['lr'] = lr

    # def _get_init_lr(self):
    #     """Get the initial lr, which is set by the scheduler"""
    #     init_lr_groups_l = []
    #     for optimizer in self.optimizers:
    #         init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
    #     return init_lr_groups_l

    # def update_learning_rate(self, cur_iter, warmup_iter=-1):
    #     for scheduler in self.schedulers:
    #         scheduler.step()
    #     # set up warm-up learning rate
    #     if cur_iter < warmup_iter:
    #         # get initial lr for each group
    #         init_lr_g_l = self._get_init_lr()
    #         # modify warming-up learning rates
    #         warm_up_lr_l = []
    #         for init_lr_g in init_lr_g_l:
    #             warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
    #         # set learning rate
    #         self._set_lr(warm_up_lr_l)

    # def get_current_learning_rate(self):
    #     return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    # def save_network(self, network, network_label, iter_label):
    #     save_filename = '{}_{}.pth'.format(iter_label, network_label)
    #     save_path = os.path.join(self.opt['path']['models'], save_filename)
    #     if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
    #         network = network.module
    #     state_dict = network.state_dict()
    #     for key, param in state_dict.items():
    #         state_dict[key] = param.cpu()
    #     torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)
        del load_net
        del load_net_clean

    # def save_training_state(self, epoch, iter_step):
    #     """Save training state during training, which will be used for resuming"""
    #     state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
    #     for s in self.schedulers:
    #         state['schedulers'].append(s.state_dict())
    #     for o in self.optimizers:
    #         state['optimizers'].append(o.state_dict())
    #     save_filename = '{}.state'.format(iter_step)
    #     save_path = os.path.join(self.opt['path']['training_state'], save_filename)
    #     torch.save(state, save_path)

    # def resume_training(self, resume_state):
    #     """Resume the optimizers and schedulers for training"""
    #     resume_optimizers = resume_state['optimizers']
    #     resume_schedulers = resume_state['schedulers']
    #     assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
    #     assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
    #     for i, o in enumerate(resume_optimizers):
    #         self.optimizers[i].load_state_dict(o)
    #     for i, s in enumerate(resume_schedulers):
    #         self.schedulers[i].load_state_dict(s)


# def define_G(conf):
#     conf_net = conf['network_G']
#     which_model = conf_net['which_model_G']

#     if which_model == 'WaveLUT_LLVE':
#         if conf['datasets'].get('test', None):
#             netG = enhance_model.WaveLUT_LLVE(input_resolution=conf_net['input_resolution'],
#                                               train_resolution=conf_net['input_resolution'],
#                                               n_ranks=conf_net['n_ranks'],
#                                               n_vertices_4d=conf_net['n_vertices_4d'],
#                                               n_base_feats=conf_net['n_base_feats'],)
#     else:
#         raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

#     return netG


class WaveLUT_network(BaseModel):
    def __init__(self, conf):
        super(WaveLUT_network, self).__init__(conf)

        if conf['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.lut_dim = conf['network_G']['n_vertices_4d']
        conf_net = conf['network_G']

        self.netG = enhance_model.WaveLUT_LLVE(input_resolution=conf_net['input_resolution'],
                                              train_resolution=conf_net['input_resolution'],
                                              n_ranks=conf_net['n_ranks'],
                                              n_vertices_4d=conf_net['n_vertices_4d'],
                                              n_base_feats=conf_net['n_base_feats'],)
        self.netG.to(self.device)
        
        # define network and load pretrained models
        # self.netG = define_G(conf).to(self.device)
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
