import argparse
import os

class Options():

    def __init__(self):
        self._parser = argparse.ArgumentParser(description='GANnotation training script')
        self.initialize()
        self.args = self.parse_args()
        self.write_args()

    def initialize(self):
        #### - Main options (filename, folder, print frequency, resume)
        self._parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint (default: none)')
        self._parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
        self._parser.add_argument('--file', default='model_',help='Store model')
        self._parser.add_argument('--folder', default='.', help='Folder to save intermediate models')

        #### - Training details (bSize, gradient clippng, learning rate, gan type)
        self._parser.add_argument('--bSize', default=64, type=int,help='Batch Size')
        self._parser.add_argument('--gc', default=1, help='Use gradient clipping')
        self._parser.add_argument('--lr_dis', default=0.0001, type=float, help='learning rate dis')
        self._parser.add_argument('--lr_gen', default=0.0001, type=float, help='learning rate gen')
        self._parser.add_argument('--gantype', default='wgan-gp', type=str, help='GAN type')
        self._parser.add_argument('--num_workers', default=12, type=int, help='Number of workers')
        
        #### - Optimizer
        self._parser.add_argument('--optim',default='Adam',help='Optimizer')
        self._parser.add_argument('--params', nargs='+', help='List of optim params', default=['betas','(0.5,0.9)'])
        self._parser.add_argument('--weight_decay', default=5*1e-4, type=float, help='weight decay')
        self._parser.add_argument('--step_size', default=1, type=int, help='Step size for scheduler')
        self._parser.add_argument('--gamma', default=0.1, type=float, help='Gamma for scheduler')

        #### - Weights
        self._parser.add_argument('--l_adv', default=0.01, type=float, help='Adv loss')
        self._parser.add_argument('--l_rec', default=1, type=float, help='Reconstruction loss')
        self._parser.add_argument('--l_self', default=1, type=float, help='Cycle consistency loss')
        self._parser.add_argument('--l_triple', default=1, type=float, help='Triple consistency loss')
        self._parser.add_argument('--l_percep', default=0.1, type=float, help='Perceptual loss')
        self._parser.add_argument('--l_tv', default=1e-8, type=float, help='Total variation loss')

        #### - Data
        self._parser.add_argument('--tight', default=16, help='Tight')
        self._parser.add_argument('--path', nargs='+', help='Path to the data')
        self._parser.add_argument('--db', nargs='+', help='Path to the data')
        self._parser.add_argument('--use_edge', action='store_true', help='Use edge-maps')

        #### - Miscellanea
        self._parser.add_argument('--visdom', action='store_true', help='Use visdom')
        self._parser.add_argument('--port', default=9001, help='visdom port')
        self._parser.add_argument('--cuda', default='auto', type=str, help='cuda')


    def parse_args(self):
        self.args = self._parser.parse_args()
        if self.args.folder == '.':
            experimentname = sorted([l for l in os.listdir(os.getcwd()) if os.path.isdir(l) and l.find('Exp') > -1])
            self.args.folder = 'Exp_{:d}'.format(len(experimentname))
        return self.args

    def write_args(self):
        if not os.path.exists('./' + self.args.folder):
            os.mkdir('./' + self.args.folder)
        if not os.path.exists(os.path.join(self.args.folder, 'code')):
            os.mkdir(os.path.join(self.args.folder,'code'))
        with open(self.args.folder + '/args_' + self.args.file[0:-8] + '.txt','w') as f:
            print(self.args,file=f)
        

