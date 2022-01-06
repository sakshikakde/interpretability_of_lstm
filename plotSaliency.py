import gi
gi.require_version('Gtk', '2.0')
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plotHeatMapExampleWise(input,
                           title, 
                           saveLocation,
                           greyScale=False,
                           flip=False,
                           x_axis=None,
                           y_axis=None,
                           show=True):
            
    if(flip):
        input=np.transpose(input)
    fig, ax = plt.subplots()

    if(greyScale):
        cmap='gray'
    else:
        cmap='seismic'
    # plt.axis('off')
    cax = ax.imshow(input, interpolation='nearest', cmap=cmap, norm=MidpointNormalize(midpoint=0))
      
    if(x_axis !=None):
        fig.text(0.5, 0.01, x_axis, ha='center' , fontsize=14)
    
    if(y_axis !=None):
        fig.text(0.05, 0.5, y_axis, va='center', rotation='vertical', fontsize=14)

    fig.tight_layout()
    # ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("ResNeT Features")
    plt.savefig(saveLocation+ str(title) + '.png' )

    if(show):
        plt.show()