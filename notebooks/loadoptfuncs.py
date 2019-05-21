import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import seaborn as sns
from IPython.display import display, HTML
from ipywidgets import *
import warnings
warnings.filterwarnings('ignore')
# from pulp import *

def prodmix_graph(zoom):
    # create the plot object
    fig, ax = plt.subplots(figsize=(8, 8))
    s = np.linspace(0, 3000)

    plt.plot(s, 10000/6 - 5*s/6, lw=3, label='$5x_1 + 6x_2 \leq 10000$')
    plt.fill_between(s, 0, 10000/6 - 5*s/6, alpha=0.1)

    plt.plot(s, 1500 - s/2, lw=3, label='$x_1 + 2x_2 \leq 3000$')
    plt.fill_between(s, 0, 1500 - s/2, alpha=0.1)

    plt.plot(600 * np.ones_like(s), s, lw=3, label='$x_1 \leq 600$')
    plt.fill_betweenx(s, 0, 600, alpha=0.1)

    plt.plot(s, 1200 * np.ones_like(s), lw=3, label='$x_2 \leq 1200$')
    plt.fill_betweenx(0, s, 1200, alpha=0.1)

    # add non-negativity constraints
    plt.plot(s, np.zeros_like(s), lw=3, label='$x_1$ non-negative')
    plt.plot(np.zeros_like(s), s, lw=3, label='$x_2$ non-negative')

    # highlight the feasible region
    path = Path([
        (0., 0.),
        (0., 1200.),
        (560, 1200.),
        (600., 7000/6),
        (600., 0.),
        (0., 0.),
    ])
    patch = PathPatch(path, label='feasible region', alpha=0.5)
    ax.add_patch(patch)

    # labels and stuff
    plt.xlabel('$x_1$ (Basic)', fontsize=16)
    plt.ylabel('$x_2$ (XP)', fontsize=16)
    if zoom:
        plt.xlim(400, 800)
        plt.ylim(1000, 1400)
    else:
        plt.xlim(-0.5, 1500)
        plt.ylim(-0.5, 1500)
    plt.legend(fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.show()

def prodmix_obj(zoom, margin1, margin2):
    fig, ax = plt.subplots(figsize=(9, 8))
    s = np.linspace(0, 1500)

    plt.plot(s, 10000/6 - 5*s/6, lw=3, label='$5x_1 + 6x_2 \leq 10000$')
    plt.plot(s, 1500 - s/2, lw=3, label='$x_1 + 2x_2 \leq 3000$')
    plt.plot(600 * np.ones_like(s), s, lw=3, label='$x_1 \leq 600$')
    plt.plot(s, 1200 * np.ones_like(s), lw=3, label='$x_2 \leq 1200$')
    plt.plot(s, np.zeros_like(s), lw=3, label='$x_1$ non-negative')
    plt.plot(np.zeros_like(s), s, lw=3, label='$x_2$ non-negative')

    # plot the possible (x1, x2) pairs
    pairs = [(x1, x2) for x1 in np.arange(start=0, stop=600, step=25)
                    for x2 in np.arange(start=0, stop=1200, step=30)
                    if (5*x1 + 6*x2) <= 10000
                    and (x1 + 2*x2)  <= 3000
                    and x1<=600 and x2<=1200]

    # split these into our variables
    x1, x2 = np.hsplit(np.array(pairs), 2)

    # caculate the objective function at each pair
    z = margin1*x1 + margin2*x2  # the objective function

    # plot the results
    plt.scatter(x1, x2, c=z, cmap='jet',
                label='Profit={} $x_1$ + {} $x_2$'.format(margin1, margin2), zorder=3)

    # labels and stuff
    cb = plt.colorbar()
    cb.set_label('profit', fontsize=14)
    plt.xlabel('$x_1$ (Basic)', fontsize=16)
    plt.ylabel('$x_2$ (XP)', fontsize=16)
    if zoom:
        plt.xlim(400, 800)
        plt.ylim(1000, 1400)
    else:
        plt.xlim(-0.5, 1500)
        plt.ylim(-0.5, 1500)
    plt.legend(fontsize=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.8, 1))
    plt.show()    
    
def showconvex(values):
    plt.subplots(2, 2, figsize=(17,10))
    function = lambda x: (x-3)**2
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,1)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Convex: Line joining any two poits is above the curve')
    
    function = lambda x: np.log(x) - (x-2)**2
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,2)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Concave: Line joining any two poits is below the curve')
    
    function = lambda x: np.log(x) - 2*x*(x-4)**2
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,3)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Neither convex or concave')
    
    function = lambda x: np.cos(x*2)*x
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,4)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Neither convex or concave')    
    
    plt.legend()
    plt.show()    
    
def deriv(x):
    x_deriv = (x-2)*(x-3)*(x-4)*(x-5)+(x-1)*(x-3)*(x-4)*(x-5)+(x-1)*(x-2)*(x-4)*(x-5)+(x-1)*(x-2)*(x-3)*(x-5)\
                +(x-1)*(x-2)*(x-3)*(x-4)
    return x_deriv

def step(x_new, x_prev, precision, l_r):
    function = lambda x: (x-1)*(x-2)*(x-3)*(x-4)*(x-5)
    x = np.linspace(1,5,500)
    x_list, y_list = [x_new], [function(x_new)]
    while abs(x_new - x_prev) > precision:
        x_prev = x_new
        d_x = - deriv(x_prev)
        x_new = x_prev + (l_r * d_x)
        x_list.append(x_new)
        y_list.append(function(x_new))

    print("Local minimum occurs at: "+ str(x_new))
    print("Number of steps: " + str(len(x_list)))
    
    plt.subplots(1, 2, figsize=(17,7))
    plt.subplot(1,2,1)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.title("Gradient descent")

    plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,function(x), c="r")
    plt.xlim([x_list[0]-.2,x_list[-1]+.2])
    plt.title("Zoomed in Gradient descent to Key Area")
    plt.show()
    
def montyhall(sample_size):
    prizes = [np.append(np.random.permutation(prizes),[1,1])\
              for prizes in np.tile(['goat', 'goat', 'car'], (sample_size,1))]
    prizes = [np.append(r,np.where(r=='car')[0]+1) for r in prizes]
    prizes = [np.append(r,np.random.choice(list(set(np.where(r=='goat')[0]+1)-{1}))) for r in prizes]
    prizes = [np.append(r,list({'2','3'}-{r[-1]})[0]) for r in prizes]    
    df = pd.DataFrame(prizes, columns=['door1','door2','door3','select', 'keep', 'prize', 'open','switch'])
    df['win'] = 'NA'
    df.win[df.prize==df.keep] = 'keep'
    df.win[df.prize==df.switch] = 'switch'
    fig, axes = plt.subplots(1, 1, figsize = (12,6))
    ax = sns.countplot(x='win', data=df, order=df['win'].value_counts().sort_values().index, ax=axes)
    total = len(df.win)
    nbars = len(ax.patches)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/2 -.05
        y = p.get_y() + p.get_height() + total/100
        ax.annotate(percentage, (x, y))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    display(df.head(10))