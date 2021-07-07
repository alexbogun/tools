if True:                        # Imports
    import os, sys
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F
    import sklearn as skl
    import scipy
    from scipy import stats
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torchvision
    import torchvision.transforms


if True:                        # Classes & Constants
    _TIME_DURATION_UNITS = (('w', 60*60*24*7), ('d', 60*60*24),  ('h', 60*60), ('m', 60),  ('s', 1))

    class Logger(object):
        def __init__(self, file):
            self.terminal = sys.stdout
            self.file = open(file, "a")
        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)  
        def flush(self):
            self.terminal.flush()
            self.file.flush()
        def __del__(self):
            self.file.close()


if True:                        # Misc functions
    def limit_threads(n):
        n= str(n)
        os.environ["OMP_NUM_THREADS"] = n # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = n # export OPENBLAS_NUM_THREADS=4
        os.environ["MKL_NUM_THREADS"] = n # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = n # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = n # export NUMEXPR_NUM_THREADS=6

    def transpose_list(x): # transpose list
        return list(map(list, zip(*x)))


    def cossim(a,b, axis=None):
        d = (np.linalg.norm(a, axis=axis)*np.linalg.norm(b, axis=axis))
        if axis==1:
            n = np.einsum('ij,ij->i', a, b)
            d[d==0] = n[d==0]
            return  n / d
        else:
            n = np.dot(a, b)
            if d==0:
                d = n
            return  n / d


    def r2(y_true, y_pred): 
        if (type(y_true) == pd.DataFrame) or (type(y_true) == pd.Series):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
        if type(y_true) == np.ndarray:
            res = 1 - ((y_true - y_pred)**2).mean() / ((y_true - y_true.mean())**2).mean()
        else:
            res = 1 -(((y_true - y_pred)**2).mean() / ((y_true - y_true.mean())**2).mean()).cpu().numpy()
        return res

    def cor(a, b, unbiased=False): 
        dof = int(unbiased)
        if (type(a) == pd.DataFrame) or (type(a) == pd.Series):
            a, b = np.array(a), np.array(b)
        if type(a) == np.ndarray:
            res = (((a - a.mean())*(b - b.mean())).mean()) / (a.std(ddof=dof)*b.std(ddof=dof))
        else:
            res = ((((a - a.mean())*(b - b.mean())).mean()) / (a.std(unbiased=unbiased)*b.std(unbiased=unbiased))).item()
        return res

    def within(y_true, y_pred, perc): 
        with np.errstate(divide='ignore'):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            res = np.sum(np.abs((y_pred / y_true) - 1) <= (perc/100))/len(y_true)
        return res

    def add_noise(tnsr, mgnt):
        noise = torch.randn(tnsr.size(), device=tnsr.device) * mgnt
        return tnsr + noise

    def find_tags(s): #function to find tags in string e.g. find_tag("aaa<tag>aaa<tag2>") = [['<tag>', 'tag'], ['<tag2>', 'tag2']]
        indices = []
        for i, c in enumerate(s):
            if c == '<':
                start = i
            elif c == '>':
                finish = i
                indices.append([s[(start):(finish+1)], s[(start+1):(finish)]])
        return indices
    
    def millify(n, dec=2):
        n = float(n)
        millidx = max(0,min(len(['','k','m','bn','tn'])-1, int(np.floor(0 if n == 0 else np.log10(abs(n))/3))))
        return ('{:.6g}{}').format( np.round(n / 10**(3 * millidx),dec), ['','k','m','bn','tn'][millidx])

    def eval(y_true, y_pred, objective='multiclass'): 
        m = type(sys.implementation)()   #SimpleNamespace
        s = type(sys.implementation)()   #SimpleNamespace
        m.Size      = len(y_true)
        s.Size      = millify(m.Size) 
        if objective == 'multiclass':
            m.ACC   = acc(y_true, y_pred)
            s.ACC   = '{:.2%}'.format(m.ACC)  
        elif objective == 'regression':
            m.MAE   = skl.metrics.mean_absolute_error(y_true , y_pred)
            s.MAE   = millify(m.MAE) 
            m.AE    = m.Size*m.MAE
            s.AE    = millify(m.AE) 
            m.MSE   = skl.metrics.mean_squared_error(y_true , y_pred)
            s.MSE   = millify(m.MSE) 
            m.RMSE  = m.MSE**0.5
            s.RMSE  = millify(m.RMSE) 
            m.w5    = within(y_true , y_pred, 5)
            s.w5   = '{:.2%}'.format(m.w5) 
            m.w10   = within(y_true , y_pred, 10) 
            s.w10   = '{:.2%}'.format(m.w10)
            m.R2    = r2(y_true , y_pred)
            s.R2   = '{:.2%}'.format(m.R2)
        return m, s


if True:                        # Numpy / Pandas / Torch wrappers
    def mean(x, dim=None):           
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = np.array(x)
        if type(x) == np.ndarray:
            return np.mean(x,dim)
        else:
            return torch.mean(x,dim)

    def median(x, dim=None):           
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = np.array(x)
        if type(x) == np.ndarray:
            return np.median(x,dim)
        else:
            return torch.median(x,dim)

    def mode(x):           
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = np.array(x)
        if type(x) == np.ndarray:
            return stats.mode(x).mode
        else:
            return torch.mode(x).values

    def argmax(x, dim=None):            
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = np.array(x)
        if type(x) == np.ndarray:
            return np.argmax(x,dim)
        else:
            return torch.argmax(x,dim)

    def zeros(shape, device == 'cpu'):            
        if (device == 'cpu'):
            return np.zeros(shape)
        else:
            return torch.zeros(shape).to(device)

    def acc(y_true, y_pred):            # function to measure accuracy
        if (type(y_true) == pd.DataFrame) or (type(y_true) == pd.Series):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
        if type(y_true) == np.ndarray:
            return np.sum(y_pred == y_true)/len(y_true)
        else:
            return torch.sum(y_pred == y_true)/len(y_true)

    def normalize(x, mean=0):               # normalize standard deviation (and mean) of a tensor
        if (type(x) == pd.DataFrame) or (type(x) == pd.Series):
            x = np.array(x)
        if type(x) ==torch.Tensor:
            std = torch.std(x)
            mean = torch.mean(x)
        else:
            std = np.std(x)
            mean = np.mean(x)
        if std == 0:
            return x
        else:
            return x / std

    def shape(x):                             # help func to view shapes recursively
        if type(x)==list:
            return '[' + str(len(x)) + ', ' +  shp(x[0]) +']'
        elif hasattr(x, 'shape'):
            return str(x.shape)
        else: 
            return ''


if True:                        # Image functions
    def inverse_transform(X, compose):
        if (type(X) == np.array) or (type(X) == np.ndarray): #should work for numpy arrays as well
            X =torch.tensor(X)
        else:
            X = X.clone()
        device = X.device
        mean = 0
        std = 1
        for t in compose.transforms:
            if isinstance(t, torchvision.transforms.Normalize):
                mean = torch.tensor(t.mean)[:,None,None].to(device)
                std = torch.tensor(t.std)[:,None,None].to(device)
                if len(X.shape)==4:
                    std = std[None,:]
                    mean = mean[None,:]
        return X*std + mean

    def apply_transform(X, compose):
        if (type(X) == np.array) or (type(X) == np.ndarray): #should work for numpy arrays as well
            X =torch.tensor(X)
        else:
            X = X.clone()
        device = X.device
        mean = 0
        std = 1
        for t in compose.transforms:
            if isinstance(t, torchvision.transforms.Normalize):
                mean = torch.tensor(t.mean)[:,None,None].to(device)
                std = torch.tensor(t.std)[:,None,None].to(device)
                if len(X.shape)==4:
                    std = std[None,:]
                    mean = mean[None,:]
        return (X - mean)/std

    def fgsm_attack(model, X, y, epsilon=0.1, criterion=None, clamp=True, transform=None):    # FGSM attack definition
        if criterion == None:
            if hasattr(model, 'criterion'):
                criterion = model.criterion
            else: criterion = nn.CrossEntropyLoss() #fallback loss function
        model.zero_grad()
        model.eval()
        X.requires_grad = True
        out = model(X)
        pred = torch.argmax(out, dim=1)
        loss = criterion(out,y)
        if X.grad != None:
            X.grad = X.grad *0  # zero gradients
        loss.backward()
        grad = X.grad.data
        sign_grad = grad.sign() 
        out = X.clone()
        out =  out + epsilon*sign_grad
        if transform:
            out = inverse_transform(out, transform) 
        if clamp:
            out = torch.clamp(out, 0, 1) 
        if transform:
            out = apply_transform(out, transform)   
        X.requires_grad = False
        return out, y

    def create_adversarial_dataset(dataloader, model, func, epsilon=0.1, criterion=None, clamp=True, transform=None):
        if criterion == None:
            if hasattr(model, 'criterion'):
                criterion = model.criterion
            else: criterion = nn.CrossEntropyLoss() #fallback loss function
        X_adv = []
        y_adv = []
        for x, y in dataloader:
            x_out, y_out = func(model = model, X=x, y=y, epsilon=epsilon, criterion=criterion, clamp=clamp, transform=transform)
            X_adv.append(x_out)
            y_adv.append(y_out)
        X_adv = torch.cat(X_adv)
        y_adv = torch.cat(y_adv)
        advset = torch.utils.data.TensorDataset(X_adv, y_adv)
        torch.cuda.empty_cache() # free cuda memmory after training
        return advset

    def show_image(img, ax = None, cmap = plt.cm.Greys, close=True, hide_axis=True, show=True, cmap_centered=False, transform=''):
        if ax == None:
            ax = plt.gca()
        if type(transform) == torchvision.transforms.transforms.Compose:
            img = inverse_transform(img, transform)
        if type(img) == torch.Tensor:           # if tensor, convert to numpy
            img = img.detach().cpu().numpy()
        img = img.copy()
        if len(img.shape)==4:                   # if 4-d, drop 1 dimension
            img=img[0]
        if len(img.shape) == 2:                 # if 2-d, add 1 dimension
            img = img[:,:,None]
        if len(img.shape) == 1:                 # if 1-d, reshape as square image (3-d)
            d = int(len(img)**0.5)
            img = np.reshape(img, [d,d,1])
        if img.shape[0]<4:                      # if color in the first place, swap axis
            img = np.transpose(img, axes=[1,2,0])
        if transform == 'pos':                  # transformations
            img = (img + np.abs(img)) / 2
        elif transform == 'neg':
            img = (img - np.abs(img)) / 2
        if img.shape[2] == 3: #color            # color image display
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        else:                                   # black / white
            if cmap_centered:   
                img = img / np.max(np.abs(img)) 
                ax.imshow(img, cmap=cmap, norm=mpl.colors.TwoSlopeNorm(0, vmin=-1, vmax=1))
            else:
                ax.imshow(img, cmap=cmap) 
        if hide_axis:
            ax.axis('off')
        if show:
            plt.show()
        if close:
            plt.close(ax.get_figure())

    def show_scatter(x, y, xlabel='', ylabel='', title='', figsize=(10,4)):
        s = ''
        _fig, _ax = plt.subplots(figsize=figsize)

        sns.regplot(x=x, y=y, ax=_ax)
        [r2, p] = r2_p_vals(x, y)
        s = '(R2: '+'{:.1%}'.format(r2) + '; P-Val: '+'{:.1%})'.format(p)

        #_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1)) # set y-axis to %
        plt.grid()

        _ax.set_xlabel(xlabel, fontweight='bold')
        _ax.set_ylabel(ylabel, fontweight='bold')
        if title != "":
            title += ' '
        _ax.set_title(title + s, fontweight='bold')

        plt.show()
        plt.close() 