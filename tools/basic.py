import numpy as np
import matplotlib.pyplot as plt



def gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type = 0, epsilon=0.02):
    


    xpos = np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), nbex // 2)
    xneg = np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), nbex // 2)
    data = np.vstack((xpos, xneg))
    y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
            # un peu de bruit
    data[:, 0] += np.random.normal(0, epsilon, nbex)
    data[:, 1] += np.random.normal(0, epsilon, nbex)


    idx = np.random.permutation((range(y.size)))
    data = data[idx, :]
    y = y[idx]
        
        
    if data_type == 1:
        # melange de 4 gaussiennes
        xpos = np.vstack((np.random.multivariate_normal([centerx, centerx], np.diag([sigma, sigma]), nbex // 4),
                          np.random.multivariate_normal([-centerx, -centerx], np.diag([sigma, sigma]), nbex // 4)))
        xneg = np.vstack((np.random.multivariate_normal([-centerx, centerx], np.diag([sigma, sigma]), nbex // 4),
                          np.random.multivariate_normal([centerx, -centerx], np.diag([sigma, sigma]), nbex // 4)))
        data = np.vstack((xpos, xneg))
        y = np.hstack((np.ones(nbex // 2), -np.ones(nbex // 2)))
                # un peu de bruit
        data[:, 0] += np.random.normal(0, epsilon, nbex)
        data[:, 1] += np.random.normal(0, epsilon, nbex)
        # on mélange les données
        idx = np.random.permutation((range(y.size)))
        data = data[idx, :]
        y = y[idx]

    if data_type == 2:
        # echiquier
        data = np.reshape(np.random.uniform(-4, 4, 2 * nbex), (nbex, 2))
        y = np.ceil(data[:, 0]) + np.ceil(data[:, 1])
        y = 2 * (y % 2) - 1
        
        
        # un peu de bruit
        data[:, 0] += np.random.normal(0, epsilon, nbex)
        data[:, 1] += np.random.normal(0, epsilon, nbex)
        # on mélange les données
        idx = np.random.permutation((range(y.size)))
        data = data[idx, :]
        y = y[idx]
    
    return data, y




def plot_data(data, labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    cols, marks = ["red", "green", "blue", "orange", "black", "cyan"], [".", "+", "*", "o", "x", "^"]
    
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1])
        return
    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels == l, 0], data[labels == l, 1], c=cols[i])

def plot_frontiere(data, f, step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), colors=('gray', 'blue'), levels=[-1, 0, 1])
    
    
    


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:, 0]), np.min(data[:, 0]), np.max(data[:, 1]), np.min(data[:, 1])
        xmax, xmin, ymax, ymin = xmax+0.1, xmin-0.1, ymax+0.1, ymin-0.1
    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step),
                       np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
    grid = np.c_[x.ravel(), y.ravel()]
    return grid, x, y






def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_image(data,n=16):
    plt.imshow(data.reshape((n,n)),interpolation="nearest",cmap="gray")


def draw_pred(X_test,y_test,net,nb_pred=6,n=16):


    random_ind = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(15,5*np.ceil(nb_pred / 3)))
    j = 1
    for i in random_ind:
        plt.subplot(np.ceil(nb_pred / 3),3,j)
        plt.title("pred : {0} true : {1}".format(net.predict(np.asarray([X_test[i]])), y_test[i]))
        show_image(X_test[i],n)
        j+=1



def draw_construction(X_test,net,nb_pred=None,n=28,n_comp=4):
    
    if nb_pred == None:
        ids = [1,3,5,7,2,0,18,15,17,4]
        nb_pred=10
    else:
        ids = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(5*nb_pred,15))
    j = 1
    for i in ids:
        plt.subplot(3,nb_pred,j)
        plt.title("Real")
        show_image(X_test[i],n)
        
        plt.subplot(3,nb_pred,j+nb_pred)
        plt.title("Compressed")
        show_image(net.forward(np.asarray([X_test[i]]))[4],n_comp)
        
        plt.subplot(3,nb_pred,j+2*nb_pred)
        plt.title("Reconstructed")
        show_image(net.predict(np.asarray([X_test[i]])),n)
        j+=1

def add_noise(data,type="gaussian",p=0.1):
    
    if type == "gaussian":
        return data + p * np.random.normal(loc=0.0, scale=0.5, size=data.shape) 
    if type == "salt_pepper":
        out = data + np.random.choice([0, 1], size=data.shape, p=[1-p, p])
        return np.where(out > 1,1,out)
    else:
        print("wrong type")

def draw_noise(X_test,X_bruit,net,nb_pred=None,n=28,n_comp=4):
    
    if nb_pred == None:
        ids = [1,3,5,7,2,0,18,15,17,4]
        nb_pred=10
    else:
        ids = np.random.choice(np.arange(X_test.shape[0]), nb_pred, replace=False)
    plt.figure(figsize=(5*nb_pred,15))
    j = 1
    for i in ids:
        plt.subplot(3,nb_pred,j)
        plt.title("Real")
        show_image(X_test[i],n)
        
        plt.subplot(3,nb_pred,j+nb_pred)
        plt.title("with Noise")
        show_image(X_bruit[i],n)
        
        plt.subplot(3,nb_pred,j+2*nb_pred)
        plt.title("Reconstructed")
        show_image(net.predict(np.asarray([X_bruit[i]])),n)
        j+=1

