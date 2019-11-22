import inspect, torch, pickle, cv2, os, numpy as np, scipy.io as sio, random, glob
from torch.utils.data import Dataset
from utils import process_image, NumpyHeatmap, EdgeMap
import torchvision.transforms as tr

class SuperDB(Dataset):

    def __init__(self, path=None, size=128, hm_size=128, flip=False, angle=0, tight=16, nimages=3, db='300VW', transform=None, edge=False):
        # - automatically add attributes to the class with the values given by the class
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        del args[0] 
        for k in args:
            setattr(self,k,values[k])
        preparedb(self,db)
        self.db = db
        if transform is None:
            transform = tr.Compose([tr.ToTensor(),
                tr.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        self.transform = transform
        if not edge:
            self.heatmap = NumpyHeatmap(out_res=hm_size, num_parts=68)
        else:
            self.heatmap = EdgeMap(out_res=hm_size, num_parts=3)
        self.ratio = int(size/hm_size)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        image, points = self.collect(idx)
        nimg = len(image)
        sample = {}
        flip = np.random.rand(1)[0] > 0.5 if self.flip else False # flip both or none
        for j in range(self.nimages):
            out = process_image(image=image[j%nimg],points=points[j%nimg],angle=(j+1)*self.angle, flip=flip, size=self.size, tight=self.tight)
            if j == 0:
                sample['B_Im'] = self.transform(out['image'])
                sample['B_Hm'] = 2*(torch.from_numpy(self.heatmap(out['points']/self.ratio))-0.5)
            if j == 1:
                sample['A_Im'] = self.transform(out['image'])
                sample['A_Hm'] = 2*(torch.from_numpy(self.heatmap(out['points']/self.ratio))-0.5)
            if j == 2:
                sample['C_Im'] = self.transform(out['image'])
                sample['C_Hm'] = 2*(torch.from_numpy(self.heatmap(out['points']/self.ratio))-0.5)
        return sample


# Define a function that returns the initialisation and the collect function
def preparedb(self, db):

    if db == 'Skeleton': # this is an example of what to prepare in a db
        def init(self):
            # - here there's the stuff needed to collect points and images and labels or whatever
            # - they are then set to db as 
            keys = ['frames','images']
            for k in keys:
                setattr(self, k, eval(k)) # if the variables are named after the keys
            setattr(self,'len', lenval )  # set value of len
        def collect(idx):
            # - function to collect a sample to be processed in getitem
            return [image], [points]
        init(self) # - do the initialisation
        setattr(self,'collect', collect) # - add collect function to the class
 
    if db == '300VW':
        def init(self):
            catC = ['410', '411', '516', '517', '526', '528', '529', '530', '531', '533', '557', '558', '559', '562']
            mylist = next(os.walk(self.path))[1]
            data = dict.fromkeys(sorted(set(mylist) - set(catC)))
            for i,name in enumerate(data.keys()):
                pathtocheck = self.path + name + '/frames/'
                files = list(map(lambda x: x.split('/')[-1], sorted(glob.glob(f'{pathtocheck}/*.jpg'))))
                pathtocheck = self.path + name + '/pts.pt'
                pts = torch.load(pathtocheck)
                data[name] = dict(zip(files,pts))
            setattr(self, 'data', data)
            setattr(self, 'map', [(x,y) for x in data.keys() for y in data[x].keys()])
            setattr(self, 'len', len(self.map))
            setattr(self, 'getim', lambda vid, frame: cv2.cvtColor(cv2.imread(os.path.join(self.path, vid, 'frames', frame)), cv2.COLOR_BGR2RGB))
        def collect(idx):
            vid, frame = self.map[idx]
            frames = list(self.data[vid].keys())
            subframes = frames[max(0, frames.index(frame)-200):min(len(frames)-1, frames.index(frame)+200)]
            subframes = [s for s in subframes if np.abs(int(s.split('.')[0]) - int(frame.split('.')[0])) > 50]
            target = random.sample(subframes, 1)[0]
            third = random.sample(subframes,1)[0]
            image = [self.getim(vid,frame), self.getim(vid,target), self.getim(vid,third)]
            points = [self.data[vid][frame].numpy(), self.data[vid][target].numpy(), self.data[vid][third].numpy()]
            return image, points
        init(self)
        setattr(self,'collect', collect)




        