import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import glob
from copy import deepcopy
from collections import Counter
########
import matplotlib.cm as cmx
import igraph as ig
from numpy import linalg as LA
import cProfile, pstats, io
from time import time
import imageio
#########

'''dx and dy elements for 4 and 8 way connectivity'''
DX = {8:[1, 1, 0,   -1],
    4:[1,  0] }
DY = {8:[0, 1, 1,    1] ,
    4:[0, 1] }

##############################################################
def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner
##################################################

def fetch_rect_coordinates(rect_coordinates):
    '''A function which is used to get bounding box coordinates from the GUI output'''
    a, b, c, d = rect_coordinates[:]
    lb_w=a
    ub_w=a+c

    lb_h=b
    ub_h=b+d

    ans={'w_min':lb_w, 'w_max':ub_w, 'h_min':lb_h, 'h_max':ub_h}
    return ans

def print_matrix_elem_freq(mat):
    unique, counts = np.unique(mat, return_counts=True)
    print(np.asarray((unique, counts)).T)
    return

class grabcut_class:

    # @profile
    def __init__(self, img_input, rect_coordinates, 
                            ground_truth_img_path,
                            img_with_rect, 
                            NUM_GMMS_PER_LABEL=5,
                             gamma_val=50, 
                             ITERATIONS=1,
                             connectivity=8):

        ######################################
        '''Difference between Trimap and matte 
        # Trimap has been given by User and hence, is certainly correct
        # Matte may be incorrect and only reflects the current state of the image based on the GrabCut algorithm
        '''
        #######################################
        # storing original image for display purposes while viewing results
        self.original_image=np.copy(img_input)

        self.curr_img=img_input.astype('int64')

        self.img_h, self.img_w= img_input.shape[:2]
        print(f"Size of the image is [h:{self.img_h} ,w:{self.img_w}]")

        # Tweakbale parameters
        self.num_models_per_label=5
        self.GAMMA_VAL=gamma_val
        self.NUM_ITERATIONS=ITERATIONS
        self.connectivity=connectivity

        ###################################
        # store GROUND TRUTH bmp image to compare with
        if ground_truth_img_path!=None:
            self.ground_truth_img=imageio.imread(ground_truth_img_path)
            self.ground_truth_img=np.where(self.ground_truth_img==255, 1, 0)
        else:
            self.ground_truth_img=None
        ####################################
        # stores the reference to the image in the GUI pipeline, has user annotations present
        self.img_with_rect=img_with_rect

        # stores algorithm output across different iterations
        self.matte_backups=[]
        self.user_backups=[]
        #########################
        '''Bounding box coordinates'''
        self.rect_dict=fetch_rect_coordinates(rect_coordinates)
        print("Bounding box coordinates: ", self.rect_dict)
        ##################################
        '''Trimap:Pixels inside the rectangle are marked as TrimapUnknown(2). Pixels outside are marked as TrimapBackground (0). '''
        self.curr_trimap=np.zeros(self.curr_img.shape[:2],dtype=int)
        self.curr_trimap[self.rect_dict['h_min']:self.rect_dict['h_max'],  self.rect_dict['w_min']:self.rect_dict['w_max']]=2

        #############################
        ##################################
        '''Matte is initialized to MatteBackground (0) in the TrimapBackground set and to MatteForeground (1) in the TrimapUnknown set.'''
        self.curr_matte=np.zeros(self.curr_img.shape[:2],dtype=int)
        self.curr_matte[self.rect_dict['h_min']:self.rect_dict['h_max'],  self.rect_dict['w_min']:self.rect_dict['w_max']]=1

        #############################
        '''Component ID'''
        # set to -1 for all initially
        self.component_ids=np.full(self.curr_img.shape[:2], -1, dtype=int)

        ###################################
        # Storing GMMs for each class
        self.NUM_GMMS_PER_LABEL=NUM_GMMS_PER_LABEL
        self.fg_gmms=None
        self.bg_gmms=None
        ####################################
        self.score_acc=[]
        self.score_jaccard=[]
        self.score_dice=[]
        ##################
        self.time_track=0
        self.itr_done=0


    def initialize_gmms(self):
        # Now, initializing GMMs
        # print("Calculating GMMs")
        h,w=self.curr_img.shape[:2]

        # fetch points for foreground        
        fg_pts, fg_locs=self.fetch_input_for_gmms(1)
        fg_gmms = mixture.GaussianMixture(n_components=self.NUM_GMMS_PER_LABEL, covariance_type='full',random_state=12)
        fg_gmms.fit(fg_pts)
        # fg_comp_ids=fg_gmms.predict(fg_pts)
        # for idx, (x,y) in enumerate(fg_locs):
        #     self.component_ids[x,y]=fg_comp_ids[idx]

        # fetch points for both foreground        
        bg_pts, bg_locs=self.fetch_input_for_gmms(0)
        bg_gmms = mixture.GaussianMixture(n_components=self.NUM_GMMS_PER_LABEL, covariance_type='full',random_state=12)
        bg_gmms.fit(bg_pts)
        # bg_comp_ids=bg_gmms.predict(bg_pts)
        # for idx, (x,y) in enumerate(bg_locs):
        #     self.component_ids[x,y]=bg_comp_ids[idx]
        self.fg_gmms=fg_gmms
        self.bg_gmms=bg_gmms
        return

    def fetch_input_for_gmms(self, wanted_val):
        # matte is either foreground or background
        assert(wanted_val in [0,1])
        x, y = np.where(self.curr_matte==wanted_val)
        pts=[tuple(self.curr_img[i,j]) for (i,j) in zip(x,y)]
        locs=[(i,j) for (i,j) in zip(x,y)]
        return pts,locs

    def fetch_node_id(self, i, j):
        ret_val=i*self.img_w+j
        return ret_val

    def fetch_unary_potentials(self):
        # print("Updating unary potentials")
        edges=[]
        wts=[]
        h,w=self.curr_img.shape[:2]
       
        rgb_vals=[self.curr_img[i,j].astype('float64') for i in range(h) for j in range(w)]
        fg_preds=self.fg_gmms.score_samples(rgb_vals)
        bg_preds=self.bg_gmms.score_samples(rgb_vals)
        cnt_now=-1
        INF_WT=1e18

        # lower the score, more likely is the class
        # we want the edge with more likely class to NOT be cut => we want the edge with less likely class to be cut => attach it to the other class

        for i in range(h):
            for j in range(w):
                cnt_now+=1
                n1=self.fetch_node_id(i,j)

                ##################
                # Unkown
                if self.curr_trimap[i,j]==2:
                    edges.append((n1,self.fg_src_id))
                    wts.append(max(0,-bg_preds[cnt_now]))

                    ########################
                    edges.append((n1,self.bg_src_id))
                    wts.append(max(0,-fg_preds[cnt_now]))
                    ################
                else:
                    if self.curr_trimap[i,j]==1:
                        # foreground surely
                        edges.append((n1,self.fg_src_id))
                        wts.append(INF_WT)

                        ########################
                        edges.append((n1,self.bg_src_id))
                        wts.append(max(0,-fg_preds[cnt_now]))
                        ################

                    else:
                        # background surely
                        edges.append((n1,self.fg_src_id))
                        wts.append(max(0,-bg_preds[cnt_now]))

                        ########################
                        edges.append((n1,self.bg_src_id))
                        wts.append(INF_WT)
                        ################
                
        return edges, wts

    # @profile       
    def init_graph(self):
        '''Creates the graph for the pipeline'''
        lb=time()
        h,w=self.curr_img.shape[:2]

        # first create the graph object
        self.graph_obj = ig.Graph(directed=False)
        TOT_NUM_NODES= h * w + 2
        self.fg_src_id=h*w
        self.bg_src_id=self.fg_src_id+1

        self.graph_obj.add_vertices(TOT_NUM_NODES)

        '''Adding all nodes'''

        self.graph_obj.vs[self.fg_src_id]['id']=(-1,-1)
        self.graph_obj.vs[self.bg_src_id]['id']=(-2,-2)
        for idx in range(len(self.graph_obj.vs)-2):
            w_c=idx%w
            h_c=idx//w
            # assert(idx==self.fetch_node_id(h_c,w_c))
            self.graph_obj.vs[self.fg_src_id]['id']=(h_c, w_c)

        '''Adding edges between pixels'''
        self.inter_pixel_edges=[]
        self.inter_pixel_edge_weights=[]
        summation_val=0
        cnt=0
        for i in range(h):
            for j in range(w):
                img_tmp_1=self.curr_img[i,j]
                n1_id=self.fetch_node_id(i,j)
                for (dx,dy) in zip(DX[self.connectivity],DY[self.connectivity]):
                    x_c,y_c=i+dx, j+dy
                    if x_c<h and y_c<w and x_c>=0 and y_c>=0:
                        img_tmp_2=self.curr_img[x_c,y_c]
                        diff_v=cv2.absdiff(img_tmp_1, img_tmp_2).flatten()
                        diff_now=diff_v.dot(diff_v).astype('int64')
                        n2_id=self.fetch_node_id(x_c, y_c)
                        self.inter_pixel_edges.append((n1_id, n2_id))
                        ###########
                        summation_val+=diff_now
                        cnt+=1
                        #########

                        curr_wt=((dx,dy),diff_now)
                        self.inter_pixel_edge_weights.append(curr_wt)
        summation_val/=cnt
        summation_val*=2
 
        self.inter_pixel_edge_weights=[   self.GAMMA_VAL*(math.exp(-(b)/summation_val))/(a[0]**2+a[1]**2) for (a,b) in self.inter_pixel_edge_weights]

        if self.connectivity==8:
            expected_num_wts=4*self.img_h*self.img_w-3*(self.img_h+self.img_w)+2
        else:
            expected_num_wts = 2 * self.img_h * self.img_w -  self.img_h - self.img_w
        assert(expected_num_wts==len(self.inter_pixel_edge_weights))
        ub=time()
        self.time_track+=ub-lb


    def run_an_iteration(self):
        '''Runs an entire iteration of the pipeline,
        1) GMMs are updated
        2) Edge weights to src and sink are recalculated
        3) Mincut is applied
        4) After mincut results, pixel labels (x) are updated
        5) Scores for measuring performance are calculated
        '''
        print(f"Starting iteration: {self.itr_done}")
        lb=time()
        self.initialize_gmms()


        # Assumes that classes are upto date based on previous maxflow runs
        # Adding edges between source, sink and pixels
        # delete any previously occurring edges
        self.graph_obj.delete_edges()

        # fetch new unary potentials
        self.unary_edges, self.unary_edge_weights=self.fetch_unary_potentials()
        
        # Adding all these edges to the graphs
        self.graph_obj.add_edges(self.inter_pixel_edges+ self.unary_edges)    

        # calculate flow
        self.calculate_maxflow_and_update_classes()

        # store a copy of the matte (x) for future analysis
        self.matte_backups.append(np.copy(self.curr_matte))

        # Update scores for this iteration
        self.update_scores()

        if isinstance(self.ground_truth_img, np.ndarray):
            # Print scores of this iteration
            print("Accuracy is ", self.score_acc[-1],end=' | ')
            print("Jaccard is ", self.score_jaccard[-1])
            # print("Dice score is ", self.score_dice[-1])
            # print("--------------------")
            pass
        print("--------------------------")
        ub=time()
        self.time_track+=ub-lb
        self.itr_done+=1


    
    def calculate_maxflow_and_update_classes(self):
        # print("Maxflow-mincut problem being solved")
        start_time=time()

        # perform maxflow
        maxflow= self.graph_obj.st_mincut(self.fg_src_id,
                                            self.bg_src_id,
                                            self.inter_pixel_edge_weights+self.unary_edge_weights )
        print("Energy val: ", maxflow.value)
        # investigate the partitons and update the suspected labels
        fg_set=set()
        bg_set=set()
        if self.fg_src_id in maxflow.partition[0]:
            fg_set=set(maxflow.partition[0])
            bg_set=set(maxflow.partition[1])
        else:
            assert(self.fg_src_id in maxflow.partition[1])
            fg_set=set(maxflow.partition[1])
            bg_set=set(maxflow.partition[0])
        fg_set.remove(self.fg_src_id)
        bg_set.remove(self.bg_src_id)
        for curr_vertex in fg_set:
            i , j = curr_vertex//self.img_w, curr_vertex%self.img_w

            # this vertex should not be a sureshot background vertex
            # assert(self.curr_trimap[i,j]!=0)
            self.curr_matte[i,j]=1

        for curr_vertex in bg_set:
            i , j = curr_vertex//self.img_w, curr_vertex%self.img_w

            # this vertex should not be a sureshot foreground vertex
            # assert(self.curr_trimap[i,j]!=1)
            self.curr_matte[i,j]=0
        
        end_time=time()
        # print("Seconds taken to run maxflow : ", end_time-start_time)
        return      

    def refine_segmentation(self, new_mask):
        # find places where value is NOW 3 and set them as perfect background
        self.user_backups.append(np.copy(new_mask))
        x, y = np.where(new_mask==3)
        for i,j in zip(x,y):
            self.curr_trimap[i,j]=0
            self.curr_matte[i,j]=0
        # find places where value is NOW 4 and set them as perfect foreground
        x, y = np.where(new_mask==4)
        for i,j in zip(x,y):
            self.curr_trimap[i,j]=1
            self.curr_matte[i,j]=1
        return
    
    def update_scores(self):
        if not isinstance(self.ground_truth_img, np.ndarray):
            return
        self.score_acc.append(self.fetch_accuracy())
        self.score_jaccard.append(self.fetch_jaccard())
        self.score_dice.append(self.fetch_dice_score())
        return
    
    def fetch_accuracy(self):
        stat=self.curr_matte==self.ground_truth_img
        return np.sum(stat)/(self.img_h*self.img_w)

    def fetch_jaccard(self):
        return np.sum(self.curr_matte&self.ground_truth_img)/np.sum(self.curr_matte|self.ground_truth_img)

    def fetch_dice_score(self):
        return 2*np.sum(self.curr_matte&self.ground_truth_img)/(np.sum(self.curr_matte)+np.sum(self.ground_truth_img))


def fetch_suspected_img(original_image, binary_mask):
    ans=np.copy(original_image)
    h,w=original_image.shape[:2]
    for i in range(h):
        for j in range(w):
            if binary_mask[i,j]==0:
                ans[i,j]=[0,0,0]
    return ans

def fetch_gt_path(org_path):
    ans=None
    try:
        img_name=org_path.split("/")[-1]
        img_name=img_name.split(".")[0]
        ans=f"../ground_truth/{img_name}.bmp"
    except:
        pass
    return ans

from matplotlib.colors import from_levels_and_colors
use_cmap, use_norm = from_levels_and_colors([0,1,2,3,4,5],['black','green','blue','red','yellow'])

def plot_all_details(gc_obj, show_accuracy_plots=True):
    fig, ax=plt.subplots()
    ax.imshow(np.flip(gc_obj.img_with_rect.astype('uint8'),axis=-1))
    ax.set_title("Original image with bounding box", fontsize=15)
    plt.show()
    #############################
    # Plotting all iterations
    imgs_per_row=3
    imgs_per_row=min(imgs_per_row, gc_obj.NUM_ITERATIONS)
    before_imgs=gc_obj.NUM_ITERATIONS
    n_rows=int(math.ceil(before_imgs/imgs_per_row))
    n_cols=imgs_per_row

    ##########
    fig, ax=plt.subplots(nrows=n_rows , ncols=n_cols)
    fig.set_size_inches(18.5, 10.5)
    if n_rows==1:
        if n_cols==1:
            ax=[[ax]]
        else:
            ax=[ax]
    # print(ax)
    for idx in range(gc_obj.NUM_ITERATIONS):
        curr_img=gc_obj.matte_backups[idx]
        i, j = idx//imgs_per_row, idx%imgs_per_row
        ax[i][j].imshow(np.flip(fetch_suspected_img(gc_obj.original_image, curr_img.astype('uint8')), axis=-1))
        ax[i][j].set_title(f"After itr {idx} :jaccard: {int(gc_obj.score_jaccard[idx]*100)}%",fontsize=15)
    
    for idx in range(gc_obj.NUM_ITERATIONS, n_rows*n_cols):
        i, j = idx//imgs_per_row, idx%imgs_per_row
        ax[i,j].axis('off')

    plt.show()

    #####################################
    #############################
    # Plotting all iterations
    NUM_INTERVENTIONS=len(gc_obj.matte_backups)-gc_obj.NUM_ITERATIONS
    if NUM_INTERVENTIONS!=0:
        n_rows=NUM_INTERVENTIONS
        n_cols=2
        # print("rows is ", n_rows)
        # print("cols is ", n_cols)

        ##########
        fig, ax=plt.subplots(nrows=n_rows , ncols=n_cols)
        fig.set_size_inches(14, 14)
        if n_rows==1:
            ax=[ax]
        # print(ax)
        for idx in range(NUM_INTERVENTIONS):
            itr_num=idx+gc_obj.NUM_ITERATIONS
            curr_img=gc_obj.matte_backups[itr_num]
            ax[idx][1].imshow(np.flip(fetch_suspected_img(gc_obj.original_image, curr_img.astype('uint8')), axis=-1))
            ax[idx][1].set_title(f"After itr {itr_num} :jaccard: {int(gc_obj.score_jaccard[itr_num]*100)}%",fontsize=15)
            ax[idx][0].imshow(gc_obj.user_backups[idx],cmap=use_cmap, norm=use_norm,interpolation=None)
            ax[idx][0].set_title(f"Refinement by user")

        plt.show()
    if show_accuracy_plots:
        
        print("Accuracy: ", [round(x,4) for x in gc_obj.score_acc])
        print("Jaccard: ", [round(x,4) for x in gc_obj.score_jaccard])
        print("Dice score is: ", [round(x,4) for x in gc_obj.score_dice])
        if len(gc_obj.score_acc)>=2:
            x_itr=[i for i in range(len(gc_obj.score_acc))]
            plt.plot(x_itr, gc_obj.score_acc,label='accuracy')
            plt.plot(x_itr, gc_obj.score_jaccard,label='jaccard')
            plt.plot(x_itr, gc_obj.score_dice,label='Dice score')
            plt.xlabel("Iterations")
            plt.ylabel("Scores")
            plt.title("Scores are:",fontsize=15)
            plt.legend()
            plt.show()
    
    #####################################
    n_cols=2
    n_rows=1
    fig, ax=plt.subplots(nrows=n_rows , ncols=n_cols)
    fig.set_size_inches(14, 14)
    if n_rows==1:
        ax=[ax]
    ax[0][0].imshow(gc_obj.ground_truth_img,cmap='gray')
    ax[0][0].set_title(f"Ground truth is",fontsize=15)

    ax[0][1].imshow(gc_obj.matte_backups[-1],cmap='gray')
    ax[0][1].set_title(f"Final result after {len(gc_obj.matte_backups)} itrs is",fontsize=15)
    plt.show()

