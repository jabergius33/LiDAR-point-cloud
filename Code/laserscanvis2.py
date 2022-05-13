import numpy as np
import vispy
from vispy.scene import visuals, SceneCanvas
from matplotlib import pyplot as plt
from processing import project
import os
import yaml
from itertools import tee

###
import vispy.io as io
from PIL import Image
from vispy.gloo.util import _screenshot

imgcanv123 = None

class LaserScanVis:
        "Class that creates and handles a visualizer for a pointcloud"

        def __init__(self, *paths, labels, offset=0):
                

                # Get file paths for all datasets
                self.scan_paths = [] # list of lists with file names
                for path in paths:
                        names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                                os.path.expanduser(path)) for f in fn]
                        names.sort()
                        self.scan_paths += [names]
                self.scan_paths = [*zip(*self.scan_paths)] # transpose
                
                assert len(self.scan_paths), 'One or multiple paths have no data'
                
                # check so file names match
                for names in self.scan_paths:
                        name1, name2 = tee(iter(names))
                        next(name2)
                        for n1, n2 in zip(name1, name2):
                                assert n1[-6:] == n2[-6:], f"{n1} doesn't match {n2}"
                                
                #LABELS                
                self.label_names = []
                for label in labels:
                        names2 = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                                os.path.expanduser(label)) for f in fn]
                        names2.sort()
                        self.label_names += [names2]                                
                self.label_names = [*zip(*self.label_names)] # transpose
                
                assert (len(self.label_names) == len(self.scan_paths))
                
 
                
                #RUN
                config = 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\semantic-kitti-all.yaml'
                self.color_dict = yaml.safe_load(open(config, 'r'))
                
                self.offset = offset
                self.reset()
                self.update_scan()

        def reset(self):
                def make_canvas(size:tuple=(800,600)):
                        canvas = SceneCanvas(keys='interactive', show=True, size=size)
                        canvas.events.key_press.connect(self.key_press)
                        canvas.events.draw.connect(self.draw)
                        return canvas

                def make_3d_canvas(number_of_visuals:int=1):
                        # build canvas
                        canvas = make_canvas()
                        grid = canvas.central_widget.add_grid() # grid
                
                        # populate canvas with visuals
                        canvas_visuals = []
                        for i in range(number_of_visuals):
                                view = vispy.scene.widgets.ViewBox(
                                        border_color='white', parent=canvas.scene)
                                grid.add_widget(view, 0, i)
                                vis = visuals.Markers()
                                view.camera = 'turntable'
                                view.add(vis)
                                visuals.XYZAxis(parent=view.scene)
                                canvas_visuals += [vis]

                        return canvas, canvas_visuals
                
                def make_2d_canvas(number_of_visuals:int=1):
                        # build canvas
                        w, h = 19, 48 # offset cause windows borders are included in size
                        canvas_W = w + 1024
                        canvas_H = h + 64 * number_of_visuals
                        canvas = make_canvas(size=(canvas_W, canvas_H))
                        grid = canvas.central_widget.add_grid() # grid
                        
                        # populate canvas with visuals
                        canvas_visuals = []
                        for i in range(number_of_visuals):
                                view = vispy.scene.widgets.ViewBox(
                                        border_color='white', parent=canvas.scene)
                                grid.add_widget(view, i, 0)
                                vis = visuals.Image(cmap='viridis')
                                view.add(vis)
                                canvas_visuals += [vis]
                        return canvas, canvas_visuals

                # canvas for visualizing point cloud data
                kwargs = {'number_of_visuals':len(self.scan_paths[0])}
                self.canvas3d, self.vis3d = make_3d_canvas(**kwargs)

                # canvas for 2d projection
                self.canvas2d, self.vis2d = make_2d_canvas(**kwargs)
                

        def get_mpl_colormap(self, cmap_name):
                # Initialize the matplotlib color map
                cmap = plt.get_cmap(cmap_name)
                sm = plt.cm.ScalarMappable(cmap=cmap)

                # Obtain linear color range
                color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
                return color_range.reshape(256, 3).astype(np.float32) / 255.0

        def update_scan(self):
                # change titles
                title = f'scan {self.offset} of {len(self.scan_paths)-1}'
                self.canvas3d.title = title
                self.canvas2d.title = title
                
                power = 1/16 # non-linearity to hide extreme values
                
                def read_one_velodyne(path):
                        'returns x, y, z and remission for one pointcloud'
                        *scan_path, file_name = path.split('\\')
                        scan_path = '/'.join(scan_path)
                        with open(scan_path+'/'+file_name, mode='rb') as file:
                                scan = np.fromfile(file, dtype=np.float32)
                                return np.reshape(scan, (-1,4))
                            
                def read_one_label(label): #new
                        'returns labels and remission for one pointcloud'
                        *label_path, file_name = label.split('\\')
                        label_path = '/'.join(label_path)
                        with open(label_path+'/'+file_name, mode='rb') as file:
                                lbl = np.fromfile(file, dtype=np.int32)
                                return np.reshape(lbl, (-1))        
                        
                # read all data in one batch
                pointclouds = [read_one_velodyne(path) for path in self.scan_paths[self.offset]]
                Label_tmp  = [read_one_label(labl) for labl in self.label_names[self.offset]] #labels
                print("Label_tmp",len(Label_tmp))
                print("pointclouds",len(pointclouds))
                #color mapping 
                max_sem_key = 0
                for key, data in self.color_dict["color_map"].items():
                    if key + 1 > max_sem_key:
                        max_sem_key = key + 1
                color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
                for key, value in self.color_dict["color_map"].items():

                    color_lut[key] = np.array(value, np.float32) / 255.0
                
                
                for i, points in enumerate(pointclouds):
                    
                        COLORS= color_lut[Label_tmp[i]]
                        
                        # plot 3d point cloud scan
                        remission = points[:,3].copy()
                        remission = remission**power
                        normalize = lambda x: (x-x.min()) / (x.max()-x.min())
                        viridis_range = (normalize(remission)*255).astype(np.uint8)
                        viridis_map = self.get_mpl_colormap("viridis")
                        viridis_colors = viridis_map[viridis_range]
                        
                        self.vis3d[i].set_data(points[:,:3],
                                face_color=COLORS[..., ::-1],
                                edge_color=COLORS[..., ::-1],
                                size=1)

                        # plot 2d range image
                        data = project(points)
                        channel = 1 # [remission, range, x, y, z, indices]
                        data = data[:,:, channel]
                        data[data < 0] = data[data > 0].min() # remove magic -1 values
                        data = data**power
                        data = normalize(data)
                        
                        self.vis2d[i].set_data(data)
                        
                if int(title[5:-7]) >= 32:
                    io.write_png(f"gifs/{title[5:-7]}.png",self.canvas3d.render()) #save picute
                #io.write_png(f"gifs/{title[5:-7]}.png",self.canvas3d.render()) #save picute
                        
                self.vis3d[0].update()


        def key_press(self, event):
                "Handler for key events"
                self.canvas3d.events.key_press.block()
                self.canvas2d.events.key_press.block()
                
                key_press2int = {'N':1, 'B':-1}
                if event.key in key_press2int.keys():
                        n = key_press2int.get(event.key, 0)
                        self.offset += n
                        self.offset %= len(self.scan_paths)
                        self.update_scan()
                        #assert False, type(self)
                elif event.key == 'Q' or event.key == 'Escape':
                        self.destroy()

        def draw(self, event):
                "Handler for draw events"
                if self.canvas3d.events.key_press.blocked():
                        self.canvas3d.events.key_press.unblock()
                if self.canvas2d.events.key_press.blocked():
                        self.canvas2d.events.key_press.unblock()

        def destroy(self):
                'destroys the visualization'
                self.canvas.close()
                self.img_canvas.close()
                vispy.app.quit()

        def run(self):
                vispy.app.run()
                
if __name__ == '__main__':


        
        '''SalsaNext
        paths = [
                 #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Salsa_nosnow\\velodyne', 
                 #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Salsa_nosnow\\velodyne-RED', 
                 ]    
        labels = [
                  #'C:\\Users\\johan\\MasterThesisCode\\input_dataset\\30\\labels',  
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\labels', #binary colors 
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\salsa\\predictions',             
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Salsa_nosnow\\predictions',    
                  #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Salsa_nosnow\\predictions-RED', 
                 ]'''
        
        
        
        '''Cylind  '''
        paths = [
                 #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Cylinder_nosnow\\velodyne', 
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Cylinder_nosnow\\velodyne-RED', 
                 ]    
        labels = [
                 # 'C:\\Users\\johan\\MasterThesisCode\\input_dataset\\30\\labels',  
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\labels', #binary colors 
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Cylinder\\predictions',             
                  #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Cylinder_nosnow\\predictions',     
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Cylinder_nosnow\\predictions-RED', 
                 ]
    
        '''DSOR  
        paths = [
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\DSOR_nosnow\\velodyne', 
                 ]    
        labels = [
                  'C:\\Users\\johan\\MasterThesisCode\\input_dataset\\30\\labels',  
                  #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\labels', #binary colors 
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\DSOR\\predictions',             
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\DSOR_nosnow\\predictions', 
                 ]'''

        '''Polarseg 
        paths = [
                 #'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\velodyne',  
                 'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Polarseg_nosnow\\velodyne', 
                 ]    
        labels = [
                  #'C:\\Users\\johan\\MasterThesisCode\\input_dataset\\30\\labels',  
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\labels', #binary colors 
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Polarseg\\predictions',             
                  'C:\\Users\\johan\\MasterThesisCode\\New_visualization\\30\\Polarseg_nosnow\\predictions', 
                 ]  '''
        
        
       
        LaserScanVis(*paths, labels=labels).run()
