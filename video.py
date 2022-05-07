import os
import numpy as np     # import numerical python package of python for maths
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Video():
    def __init__(self, folderDirectory = 'C:/Users/Jany/thesis_code/structure1_v1'):
        
        # Directories
        self.folderDirectory = folderDirectory
        self.output_dir = folderDirectory + '/output'
    
    def play(self):
        '''Play an animation of the left- and right-going field, as welll as the charge carrier density.'''

        vid_dat_name_LRp  = 'PHIsimout_vid_LRp.txt'
        vid_dat_name_RLp  = 'PHIsimout_vid_RLp.txt'
        vid_dat_name_car  = 'PHIsimout_vid_carriers.txt'

        os.chdir(self.output_dir)

        video_dat_LRp = np.loadtxt(vid_dat_name_LRp,dtype=np.float_,unpack=True,ndmin=2)

        dimens        = np.shape(video_dat_LRp)
        nr_frames     = dimens[1]
        nr_sl_data    = dimens[0]
        print('Nr of frames:', nr_frames)
        print('Nr of data:', nr_sl_data)
        # load the other data files with RLp en carriers as well
        video_dat_RLp = np.loadtxt(vid_dat_name_RLp,dtype=np.float_,unpack=True,ndmin=2)
        video_dat_car = np.loadtxt(vid_dat_name_car,dtype=np.float_,unpack=True,ndmin=2)

        max_LRp = video_dat_LRp.max()
        max_RLp = video_dat_RLp.max()
        max_car = video_dat_car.max()

        maxy = max_LRp
        if max_RLp > max_LRp:
            maxy = max_RLp   

        # copy data to a 3d array and scale all data to 1
        npdata = np.random.rand(nr_sl_data, 3, nr_frames)
        npdata[:,0,:] = video_dat_LRp[:,:] / maxy
        npdata[:,1,:] = video_dat_RLp[:,:] / maxy
        npdata[:,2,:] = video_dat_car[:,:] / max_car

        max_y = 1.0
        # max_y = video_dat_car.max()
        print('Maximum value in video data: ', max_y)

        #colours of the lines
        plotlays, plotcols = [0,1,2], ["red","blue","green"]

        # First set up the figure, the axis, and the plot element we want to animate
        fig   = plt.figure()
        ax    = plt.axes(xlim=(0, nr_sl_data), ylim=(0, max_y))
        y     = np.linspace(1, nr_sl_data, nr_sl_data)

        lines = []
        for index,lay in enumerate(plotlays):
            lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
            lines.append(lobj)

        # initialization function: plot the background of each frame
        def init():    
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            x = np.linspace(1, nr_sl_data, nr_sl_data)
            
            for lnum,line in enumerate(lines):        
                line.set_data(x, npdata[:,lnum,i])

            return lines

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=nr_frames, interval=100, blit=True)

        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

        #anim.save('SOA_only.mp4')

        plt.show()