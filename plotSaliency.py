import argparse
import sys
import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors

Loc_Graph = '../Graphs/'
Results='../Results/'


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plotHeatMapExampleWise(input, title, saveLocation,greyScale=False,flip=False,x_axis=None,y_axis=None,show=True):
            
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


def main(args):
    
    save_folder = "/home/sakshi/courses/CMSC828W/cnn-lstm/saliency/"
    ModelTypes = "lstm_cnn_fc"
    block_number = 0
    file_name = ModelTypes+str(block_number) + ".npy"
    loaded_sal = np.load(save_folder + file_name)
    loaded_sal = (255 * (loaded_sal / np.max(loaded_sal))).astype(np.uint8)
    # frame_count = loaded_sal.shape[0]
    
    print("sal_shape = ", loaded_sal.shape)
    plotHeatMapExampleWise(loaded_sal.T, ModelTypes+str(block_number), save_folder)

    # for s in range(frame_count):
    #     input = loaded_sal[s, :, :, :]
    #     input = input.reshape(150, 150, 3)
    #     # input = input / np.max(input) * 256
    #     plotHeatMapExampleWise(input, str(s), save_folder)
        
  
def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--DataName', type=str ,default="TopBox")
    parser.add_argument('--sequence_length', type=int, default=100)
    parser.add_argument('--input_size', type=int,default=100)
    parser.add_argument('--importance', type=str, default=0)
    parser.add_argument('--data-dir', help='Data  directory', action='store', type=str ,default="../Data/")
    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
