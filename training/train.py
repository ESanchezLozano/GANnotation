from __future__ import print_function, division
import glob, os, sys, pickle, torch, cv2, time, numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.utils import save_image
from shutil import copy2
# mystuff
from GANnotation import GANnotation as model
from databases import SuperDB
from utils import AverageMeter
from Train_options import Options

def main():
    # parse args
    global args
    args = Options().args
    
    # copy all files from experiment
    cwd = os.getcwd()
    for ff in glob.glob("*.py"):
        copy2(os.path.join(cwd,ff), os.path.join(args.folder,'code'))

    # initialise seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    np.random.seed(1000)

    # choose cuda
    if args.cuda == 'auto':
        import GPUtil as GPU
        GPUs = GPU.getGPUs()
        idx = [GPUs[j].memoryUsed for j in range(len(GPUs))]
        print(idx)
        assert min(idx) < 11.0, 'All {} GPUs are in use'.format(len(GPUs))
        idx = idx.index(min(idx))
        print('Assigning CUDA_VISIBLE_DEVICES={}'.format(idx))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # get lambdas for the losses in the model and define model (GANnotation)
    myvars = [u for u in list(args.__dict__.keys()) if 'l_' in u]
    losses = dict(zip([m.replace('l_','') for m in myvars],[float(args.__getattribute__(val)) for val in myvars]))
    print(losses)
    with open(args.folder + '/args_' + args.file[0:-8] + '.txt','a') as f: 
        print( losses , file=f)
    GANnotation = model(losses=losses, gradclip=int(args.gc), gantype=args.gantype, edge=args.use_edge)
    if args.resume:
        GANnotation._resume(path_to_gen=args.resume + str(args.start_epoch)+'.gen.pth', path_to_dis=args.resume + str(args.start_epoch) + 'dis.pth')
    
    # define the plotting keys
    plotkeys = ['input','target','generated']
    genkeys = list(GANnotation.loss['G'].keys())
    diskeys = list(GANnotation.loss['D'].keys())
    
    # define plotters
    global plotter
    if not args.visdom:
        print('No Visdom')
        plotter = None
    else:
        from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomSaver, VisdomTextLogger
        experimentsName = str(args.visdom)
        plotter = dict.fromkeys(['images','G','D'])
        plotter['images'] = dict( [ (key, VisdomLogger("images", port=int(args.port), env=experimentsName, opts={'title' : key})) for key in plotkeys ])
        plotter['G'] = dict( [ (key, VisdomPlotLogger("line", port=int(args.port), env=experimentsName,opts={'title': f'G : {key}', 'xlabel' : 'Iteration', 'ylabel' : 'Loss'})) for key in genkeys]  )
        plotter['D'] = dict( [ (key, VisdomPlotLogger("line", port=int(args.port), env=experimentsName,opts={'title': f'D : {key}', 'xlabel' : 'Iteration', 'ylabel' : 'Loss'})) for key in diskeys]  )
        
    # prepare average meters
    global meters, l_iteration
    meterskey = ['batch_time', 'data_time'] 
    meters = dict([(key,AverageMeter()) for key in meterskey])
    meters['G'] = dict([(key,AverageMeter()) for key in genkeys])
    meters['D'] = dict([(key,AverageMeter()) for key in diskeys])
    l_iteration = float(0.0)
    
    # plot number of parameters
    params = sum([p.numel() for p in filter(lambda p: p.requires_grad, GANnotation.GEN.parameters())])
    print('GEN # trainable parameters: {}'.format(params))
    params = sum([p.numel() for p in filter(lambda p: p.requires_grad, GANnotation.DIS.parameters())])
    print('DIS # trainable parameters: {}'.format(params))

    # define data
    assert len(args.db) == len(args.path), 'Number of root paths not equal to number of dbs'
    videoloader = DataLoader(ConcatDataset([SuperDB(path=args.path[i], db=db, size=128, hm_size=128, flip=True, angle=15, tight=int(args.tight), edge=args.use_edge) for i,db in enumerate(args.db)]),
        batch_size=args.bSize, shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
    print('Number of workers is {:d}, and bSize is {:d}'.format(int(args.num_workers),args.bSize))
       
    # define optimizers
    args.params[1::2] = [eval(args.params[2*j+1]) for j in range(int(len(args.params)/2))]
    optim = getattr(args, 'optim', 'Adam')
    params = dict(zip(args.params[::2], args.params[1::2]))
    optimizerGEN = getattr(torch.optim,optim)(GANnotation.GEN.parameters(), lr=args.lr_gen, **params, weight_decay=args.weight_decay)
    optimizerDIS = getattr(torch.optim,optim)(GANnotation.DIS.parameters(), lr=args.lr_dis, **params, weight_decay=args.weight_decay)
    schedulerGEN = torch.optim.lr_scheduler.StepLR(optimizerGEN, step_size=args.step_size, gamma=args.gamma)
    schedulerDIS = torch.optim.lr_scheduler.StepLR(optimizerDIS, step_size=args.step_size, gamma=args.gamma)
    myoptimizers = {'GEN': optimizerGEN, 'DIS': optimizerDIS}

    # path to save models and images
    path_to_model = os.path.join(args.folder,args.file)

    # train
    for epoch in range(0,80):
        train_epoch(videoloader, GANnotation, myoptimizers, epoch, args.bSize)
        GANnotation._save(path_to_model,epoch)
        schedulerDIS.step()
        schedulerGEN.step()
        

def train_epoch(dataloader, model, myoptimizers, epoch, bSize):
    
    global l_iteration
    log_epoch = {}
    end = time.time()
    disiters = 5 if args.gantype == 'wgan-gp' else 2
    for i, data in enumerate(dataloader):
        data = {k: v.to('cuda') for k,v in data.items()}
        if i > 10000:
            break
        
        if (i+1) % disiters == 0:
            model._forward_G(data)
            myoptimizers['GEN'].step()
            for k in meters['G'].keys():
                meters['G'][k].update( model.loss['G'][k].item() if model.loss['G'] is not None else 0, bSize)

        else:
            model._forward_D(data)
            myoptimizers['DIS'].step()
            for k in meters['D'].keys():
                meters['D'][k].update( model.loss['D'][k].item() if model.loss['D'] is not None else 0, bSize)
        
        l_iteration = l_iteration + 1

        
        if i % 100 == 0:                
            # - plot some images
            with torch.no_grad():
                AB = model(data['A_Im'], data['B_Hm'])
            
            if plotter is not None:
                plotter['images']['input'].log(0.5*(1+data['A_Im'].data))
                plotter['images']['target'].log(0.5*(1+data['B_Im'].data))            
                plotter['images']['generated'].log(0.5*(1+AB.cpu().data))
                
                for k in ['G','D']:
                    for (j,_) in filter(lambda x: x[1] is not None, model.loss[k].items()):
                        plotter[k][j].log(l_iteration, model.loss[k][j].item())

                
            saveA = 0.5*(torch.nn.functional.interpolate(data['A_Im'],scale_factor=0.25)+1)
            saveB = 0.5*(torch.nn.functional.interpolate(data['B_Im'],scale_factor=0.25)+1)
            saveAB = 0.5*(torch.nn.functional.interpolate(AB,scale_factor=0.25)+1)
            save = torch.cat((saveA, saveB, saveAB), 2)
            save_image(save, args.folder + '/{}_{}.png'.format(epoch,i))
                    
        log_epoch[i] = model.loss       
        meters['batch_time'].update(time.time()-end)
        end = time.time()
        if i % args.print_freq == 0:
            mystr = 'Epoch [{}][{}/{}] '.format(epoch, i, len(dataloader))
            mystr += 'Time {:.2f} ({:.2f}) '.format(meters['data_time'].val , meters['data_time'].avg )
            mystr += ' '.join(['G: {:s} {:.3f} ({:.3f}) '.format(k, meters['G'][k].val , meters['G'][k].avg ) for k in meters['G'].keys()])
            mystr += ' '.join(['D: {:s} {:.3f} ({:.3f}) '.format(k, meters['D'][k].val , meters['D'][k].avg ) for k in meters['D'].keys()])
            print( mystr )
            with open(args.folder + '/args_' + args.file[0:-8] + '.txt','a') as f: 
                print( mystr , file=f)

    with open(args.folder + '/args_' + args.file[0:-8] + '_' + str(epoch) + '.pkl','wb') as f:
        pickle.dump(log_epoch,f)  

if __name__ == '__main__':
    main()

