from PIL import Image
from cython.parallel import prange
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from queue import Queue
from sklearn.decomposition import PCA
import math
# cimport numpy as cnp
from libc.math cimport floor, sqrt, exp 
cimport numpy as cnp
import cython
from cpython cimport array
from libc.stdint cimport int64_t
from libc.stdio cimport printf
import array
import time
from typing import List
from scipy.stats import multivariate_normal
cnp.import_array()

# cpdef float sum3d(double[:, :] arr):
#     cdef size_t i, j, I, J
#     cdef float totalL = 0, totalA = 0, totalB = 0
#     I = arr.shape[0]
#     J = arr.shape[1]
    
#     for i in prange(I, nogil = True):dif3dq
        
            
#         totalA += arr[i, 0]
#     return totalA
def sum3d(double[:,:] arr, int[:] indxs, int count):
    cdef size_t i
    cdef float totalL = 0, totalA = 0, totalB = 0
    
    
    
    for i in prange(count, nogil = True):
        totalL += arr[indxs[i]][ 0]
        totalA += arr[indxs[i]][ 1]
        totalB += arr[indxs[i]][ 2]
        
    return totalL/count, totalA/count, totalB/count
class Colour_Tree(object):
    def __init__(self, indx, parent):
        self.indx = indx
        self.parent = parent
        self.lchld = None
        self.rchld = None
        pass
cdef class SuperPixel:
    cdef array.array pixelsarr
    cdef int[:] pixels_view
    pixels: dict
    colour: float | List[float]
    x: int
    y: int
    cntrx: float
    cntry: float
    pxls: int
    palette_indx: int
    
    def __init__(self, ratio) -> None:
        self.pixels = dict()
        self.colour = 0
        self.x = 0
        self.y = 0
        self.cntrx = 0
        self.cntry = 0
        self.pixelsarr = array.array('i', [-1 for _ in range(8*ratio*ratio)])
        self.pixels_view = self.pixelsarr
        self.pxls = 0
        self.palette_indx = 0
        # self.pixelsarr2 = []
         
        pass
    def recenter(self, dt_arr, mask, idx):
        self.cntry = np.mean(np.floor(np.where(mask == idx)[0]/ 400))
        self.cntrx = np.mean(np.floor(np.where(mask == idx)[0]% 400))
        self.pixels_view = np.where(mask == idx)[0].astype(int)
        
        
        
        self.pxls = self.pixels_view.shape[0]
        
       
    def set_colour(self, col):
        self.colour = col
    def get_pixels_view(self):
        return self.pixels_view 
    def get_colour(self):
        return self.colour
    def get_count(self):
        return self.pxls
    def get_pixels(self):
        return self.pixels
    def set_palette_index(self, indx):
        self.palette_index = indx
    def get_palette_index(self):
        return self.palette_index
    def get_centerx(self):
        return self.cntrx
    def get_centery(self):
        return self.cntry
class Pixeliser(object):
    sz = (8, 8)
    quantisation = 8
    glob_img = None
    quantised = None
    method = "None"
    doing = False
    running = True
    avg_pixels = False
    msg_queue = Queue()
    norm = multivariate_normal(mean = [0,0,0], cov=[[1,0,0],[0,1,0],[0,0,1]])
    eps_c: float = 0.0025 
    eps_d: float = 0.005
    palette_history = []
    sprpxl_history = []
    dt_arr = None
    mp = None
    sprpxls = None
    N = 64
    minimal_change = 0.0005
    def nearest():
        return Pixeliser.quantised.resize(Pixeliser.sz, Image.NEAREST)
    def cubic():
        return Pixeliser.quantised.resize(Pixeliser.sz, Image.CUBIC)
    def make_img(sprpxls, palette,sprpxl_indxs):
        ret = []
        # print(palette)
        # post processing:
        palette[:,1] *= 1.1
        palette[:,2] *= 1.1
        for i, pxl in enumerate(sprpxls):
            if Pixeliser.avg_pixels:
                ret.append(pxl.get_colour())
            else:
                # print(palette[sprpxl_indxs[i]])
                # print(sprpxl_indxs[i])
                
                ret.append(palette[sprpxl_indxs[i]])
        ret = np.array(ret)
        # print(ret.shape)
        ret = ret.reshape(Pixeliser.sz[0], Pixeliser.sz[1],3)
        
        Pixeliser.msg_queue.put(Image.fromarray((lab2rgb(ret)*255).astype(np.uint8)))
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def paper():
        # Pre-workout
        pca = PCA()
        cdef int N = Pixeliser.sz[0]*Pixeliser.sz[1]
        lab_img = rgb2lab(Pixeliser.glob_img)
        dt_arr = lab_img.reshape(-1, 3)
        sprpxls = []
        Pixeliser.palette_history = []
        Pixeliser.sprpxl_history = []
        ratio_y = math.floor(400 / Pixeliser.sz[1])
        ratio_x = math.floor(400 / Pixeliser.sz[0])
        nbrhoods = []
        change_min = Pixeliser.minimal_change
        mask = np.ones(400*400, dtype = np.intc)*-1
        cdef int[:] cmask = mask
        cdef double[:,:] pcp = np.zeros((2*Pixeliser.quantisation,N))
        cdef double[:] pc = np.ones(2*Pixeliser.quantisation)/2
        # cdef int[:] par = np.zeros(2*Pixeliser.quantisation, dtype = np.intc)
        cdef array.array cntrs_x = array.array('f', [-1 for _ in range(N)])
        cdef array.array cntrs_y = array.array('f', [-1 for _ in range(N)])
        cdef float[:] view_cntsx = cntrs_x
        cdef float[:] view_cntsy = cntrs_y

        cdef array.array cntrs2_x = array.array('f', [-1 for _ in range(N)])
        cdef array.array cntrs2_y = array.array('f', [-1 for _ in range(N)])
        cdef float[:] view_cntsx2 = cntrs2_x
        cdef float[:] view_cntsy2 = cntrs2_y
        # cdef int[:] palette_indexes = np.zeros(N, dtype = np.intc)
        cdef array.array colours = array.array('f', [-1 for _ in range(N*3)])
        cdef float[:] view_colours = colours
        cdef int i2
        
        for i in range(N):
            tmp = SuperPixel(ratio_x)
            tmp.x = i % Pixeliser.sz[0]
            tmp.y = math.floor(i / Pixeliser.sz[1])
            for y in range(ratio_y):
                for x in range(ratio_x):
                    # if tmp.x*Pixeliser.sz[0] + x >= 400 or tmp.y*Pixeliser.sz[1] + y >= 400:
                    #     continue
                    indx = (tmp.y*ratio_y + y) * 400 + tmp.x*ratio_x + x
                    
                    
                    tmp.pixels[indx] = 1
                    
                    cmask[indx] = i

            tmp.recenter(dt_arr, mask, i)
            nbrhoods.append((tmp.cntrx, tmp.cntry))
            cntrs_x[i] = tmp.cntrx
            cntrs_y[i] = tmp.cntry
            view_cntsx2[i] = tmp.cntrx
            view_cntsx2[i] = tmp.cntry
            sprpxls.append(tmp)
        print("Initialised")
        
        # step 1:
        principle_choice = -1
        pca.fit(dt_arr)
        TC = pca.explained_variance_[principle_choice] * 2
        cdef float t = 1.1 * TC
        tmp_mn = np.mean(dt_arr, axis = 0)
        EPS_change = Pixeliser.eps_c
        print("With epsilon change of ", EPS_change)
        print(pca.components_.shape)
        print(pca.explained_variance_)
        root_tree = Colour_Tree(-1, None)
        q_tree = []
        tmp_chld = Colour_Tree(0, root_tree)
        root_tree.lchld = tmp_chld
        tmp_chld = Colour_Tree(1, root_tree)
        root_tree.rchld = tmp_chld
        palette = np.array([tmp_mn-EPS_change*pca.components_[principle_choice],tmp_mn+EPS_change*pca.components_[principle_choice]])
        cdef int loc_sz = Pixeliser.sz[0]
        cdef int n2 = 400*400
        cdef float m = 40 * math.sqrt(Pixeliser.sz[0]*Pixeliser.sz[1] / (400*400))
        cdef double[:,:] pxls_view = dt_arr    
        new_colours: cython.float[:] = np.zeros(loc_sz*loc_sz*3 ,dtype= np.single)
        tmp_color = None
        # step 2: 
        len_p: cython.int = 1
        for i in range(loc_sz*loc_sz):
                l,a,b = sum3d(dt_arr,sprpxls[i].get_pixels_view(), sprpxls[i].get_count())
                sprpxls[i].set_colour([l,a,b])
                view_colours[i*3] = l
                view_colours[i*3 + 1] = a
                view_colours[i*3 + 2] = b
        while t > 1 and len_p < 2000:
            # print("START", palette)
            print(len_p)
            
            
            
            
            i2 = 0
            
            # perform every other run because this is the bottleneck :/
            for i2 in prange(n2*(len_p%2) , nogil=True):
                if cmask[i2] == -1:
                    continue
                
                y_c:  cython.float = floor(i2 // 400)
                x_c:  cython.float = i2 % 400
                x_c = x_c
                y_c = y_c
                s_c: int64_t = cmask[i2]
                sy_c: int64_t = <int>floor(s_c // loc_sz)
                sx_c: int64_t = s_c % loc_sz
                min_c: cython.float = 4000000.0
                min_indx: int64_t = 4000
                tmp_dist_x: cython.float = 0.0
                tmp_dist_y: cython.float = 0.0
                res: cython.float = 0.0
                acc_loc: cython.float = 0.0 
                if sy_c > 0:
                    tmp_dist_x = view_cntsx[s_c - loc_sz] - x_c
                    tmp_dist_y = view_cntsy[s_c - loc_sz] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 - loc_sz*3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 1 - loc_sz*3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 2 - loc_sz*3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    # res += dif3d(dt_arr, view_colours, i2, s_c, m)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c - loc_sz
                
                if sx_c > 0:
                    tmp_dist_x = view_cntsx[s_c - 1] - x_c
                    tmp_dist_y = view_cntsy[s_c - 1] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 - 3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 - 2]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 - 1]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c - 1
                if sx_c > 0 and sy_c > 0:
                    tmp_dist_x = view_cntsx[s_c - loc_sz - 1] - x_c
                    tmp_dist_y = view_cntsy[s_c - loc_sz - 1] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 - loc_sz*3 - 3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 1 - loc_sz*3 - 3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 2 - loc_sz*3 - 3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c - loc_sz - 1
                if True:
                    tmp_dist_x = view_cntsx[s_c] - x_c
                    tmp_dist_y = view_cntsy[s_c] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 1]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 2]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c
                if sx_c < loc_sz - 1:
                    tmp_dist_x = view_cntsx[s_c + 1] - x_c
                    tmp_dist_y = view_cntsy[s_c + 1] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 + 3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 4]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 5]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c + 1
                if sy_c < loc_sz - 1:
                    tmp_dist_x = view_cntsx[s_c + loc_sz] - x_c
                    tmp_dist_y = view_cntsy[s_c + loc_sz] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 + loc_sz*3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 1 + loc_sz*3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 2 + loc_sz*3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c + loc_sz   

                if sx_c < loc_sz - 1 and sy_c < loc_sz - 1:
                    tmp_dist_x = view_cntsx[s_c + loc_sz + 1] - x_c
                    tmp_dist_y = view_cntsy[s_c + loc_sz + 1] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 + loc_sz*3 + 3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 1 + loc_sz*3 + 4]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 2 + loc_sz*3 + 5]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c + loc_sz + 1
                
                if sx_c < loc_sz - 1 and sy_c > 0:
                    tmp_dist_x = view_cntsx[s_c - loc_sz + 1] - x_c
                    tmp_dist_y = view_cntsy[s_c - loc_sz + 1] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 - loc_sz*3 + 3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 1 - loc_sz*3 + 3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 2 - loc_sz*3 + 3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c - loc_sz + 1
                
                if sx_c > 0 and sy_c < loc_sz - 1:
                    tmp_dist_x = view_cntsx[s_c + loc_sz - 1] - x_c
                    tmp_dist_y = view_cntsy[s_c + loc_sz - 1] - y_c
                    res =  m *  sqrt(tmp_dist_x*tmp_dist_x + tmp_dist_y*tmp_dist_y)
                    tmp_dist_x = pxls_view[i2][0] - view_colours[s_c*3 + loc_sz*3 - 3]
                    acc_loc = tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][1] - view_colours[s_c*3 + 1 + loc_sz*3 - 3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    tmp_dist_x = pxls_view[i2][2] - view_colours[s_c*3 + 2 + loc_sz*3 - 3]
                    acc_loc += tmp_dist_x*tmp_dist_x
                    res += sqrt(acc_loc)
                    if res < min_c:
                        min_c = res
                        min_indx = s_c + loc_sz - 1
                
                res = 0
                acc_loc = 0
                if min_indx != s_c:
                    cmask[i2] = min_indx
                    with gil:
                        # check += 1
                        # print(min_indx)
                        sprpxls[ s_c].get_pixels().pop(i2)
                        sprpxls[min_indx].get_pixels()[i2] = 1

            
            # if check > 1:
            #     print(check)
            
            # recenter
            for i in range(N*(len_p%2)):
                tmp_loc = sprpxls[i]
                tmp_loc.recenter(dt_arr, mask, i)
                view_cntsx[i] = tmp_loc.get_centerx()
                view_cntsy[i] = tmp_loc.get_centery()
            
            # calculate colours
            for i in range(loc_sz*loc_sz):
                l,a,b = sum3d(dt_arr,sprpxls[i].get_pixels_view(), sprpxls[i].get_count())
                sprpxls[i].set_colour([l,a,b])
                view_colours[i*3] = l
                view_colours[i*3 + 1] = a
                view_colours[i*3 + 2] = b
            i2 = 0
            # laplacian smoothing
            for i2 in prange(loc_sz*loc_sz , nogil=True):   
                
                sum_x: cython.float = 0.0
                sum_y: cython.float = 0.0
                x_loc: int64_t = 0
                sy_c: cython.int = <cython.int>floor(i2 / loc_sz)
                sx_c: int64_t = i2 % loc_sz
                count: cython.float = 0
                if sy_c > 0:
                    sum_x += view_cntsx[i2 - loc_sz]
                    sum_y += view_cntsy[i2 - loc_sz]
                    count += 1
                if sx_c > 0:
                    sum_x += view_cntsx[i2 - 1]
                    sum_y += view_cntsy[i2 - 1]
                    count += 1
                # printf("%d %d %lld %d\n", i2+1, sx_c, sy_c, loc_sz)
                if sx_c < loc_sz - 1:
                    
                    sum_x += view_cntsx[i2 + 1]
                    sum_y += view_cntsy[i2 + 1]
                    count += 1
                if sy_c < loc_sz - 1:
                    sum_x += view_cntsx[i2 + loc_sz]
                    sum_y += view_cntsy[i2 + loc_sz]
                    count += 1
                view_cntsx2[i2] = 0.6 * view_cntsx[i2] + 0.4 * (sum_x/count)
                view_cntsy2[i2] = 0.6 * view_cntsy[i2] + 0.4 * (sum_y/count)
                # printf("%f %f %f %f\n", view_cntsx2[i2], view_cntsx[i2], view_cntsy2[i2], view_cntsy[i2])
                sum_x = 0
                sum_y = 0
                count = 0



            
            # print("finished smoothing")
            len_p += 1
            tmp_ctrs = view_cntsx
            view_cntsx = view_cntsx2
            view_cntsx2 = tmp_ctrs
            tmp_ctrs = view_cntsy
            view_cntsy = view_cntsy2
            view_cntsy2 = tmp_ctrs
            # Bilateral filter:
            i2 = 0
            # print(view_colours[56])
            for i2 in prange(0*loc_sz*loc_sz, nogil=True):
                loc_idx: int64_t = 0
                res_L_loc: cython.double = 0
                res_a_loc: cython.double = 0
                res_b_loc: cython.double = 0
                sy_c: cython.int = <cython.int>floor(i2 / loc_sz)
                sx_c: int64_t = i2 % loc_sz
                glob_k_const: cython.double = 0
                for loc_idx in prange(9):
                        x_off: int64_t = loc_idx % 3
                        y_off: int64_t = <cython.int>floor(loc_idx/3)
                        if sx_c + (x_off - 1) >= loc_sz or sx_c + (x_off - 1) < 0:
                            continue
                        if sy_c + (y_off - 1) >= loc_sz or sy_c + (y_off - 1) < 0:
                            continue
                        if loc_idx == 4:
                            continue
                        nxt_indx: int64_t = i2 + x_off - 1 + (y_off - 1)*loc_sz
                        g: cython.float = exp( -((view_colours[i2*3] - view_colours[nxt_indx*3] )*(view_colours[i2*3] - view_colours[nxt_indx*3] ) 
                        +  (view_colours[i2*3 + 1] - view_colours[nxt_indx*3 + 1] )*(view_colours[i2*3 + 1] - view_colours[nxt_indx*3 + 1] ) + 
                        (view_colours[i2*3 + 2] - view_colours[nxt_indx*3 + 2] )*(view_colours[i2*3 + 2] - view_colours[nxt_indx*3 + 2] )) / (2 * 0.5 * 0.5) )
                        g2: cython.float = exp( -((y_off -  1)*(y_off -  1 ) 
                        +  ( x_off -  1)*( x_off -  1) 
                        ) / (2 * 1 * 1) )
                        res_L_loc+=g2*g*view_colours[nxt_indx*3]
                        res_a_loc+=g2*g*view_colours[nxt_indx*3+1]
                        res_b_loc+=g2*g*view_colours[nxt_indx*3+2]
                        glob_k_const+=g2*g
                        # printf("%f - g %f - g2 %d %d\n", g, g2, i2, nxt_indx, sy_c, sx_c, x_off, y_off)
                        # g = 0
                        # g2 = 0
                    
                       
                new_colours[i2*3] = (view_colours[i2*3]+res_L_loc)/(glob_k_const+ 1)
                new_colours[i2*3 + 1] = (view_colours[i2*3+1]+res_a_loc)/(glob_k_const+ 1)
                new_colours[i2*3 + 2] = (view_colours[i2*3+2]+res_b_loc)/(glob_k_const+ 1)
                # printf("%f %f  %f %f %f\n",view_colours[i2*3], new_colours[i2*3], view_colours[i2*3+1], new_colours[i2*3 +1], glob_k_const)
                glob_k_const = 0
                res_a_loc = 0
                res_L_loc = 0
                res_b_loc = 0
            # print("bilateral",palette,palette.shape,new_colours[56],view_colours[56])
            tmp_color = view_colours
            view_colours = new_colours
            new_colours = tmp_color
            # print("bilateral2 log:" ,new_colours[56],view_colours[56])
            
            # Step 2: Palette association:
            
            # calculate probability
            i2 = 0
            
            palette_size: cython.int = palette.shape[0]
            palette_view: cython.double[:,:]= palette
            for i2 in prange(loc_sz*loc_sz , nogil=True):
                c_count: int64_t = 0
                
                for c_count in prange(palette_size):
                    
                    res: cython.float = 0
                    tmp_dist_x: cython.float  = palette_view[c_count][0] - view_colours[i2*3]
                    res +=tmp_dist_x*tmp_dist_x
                    tmp_dist_x = palette_view[c_count][1] - view_colours[i2*3 + 1]
                    res +=tmp_dist_x*tmp_dist_x
                    tmp_dist_x = palette_view[c_count][2] - view_colours[i2*3 + 2]
                    res +=tmp_dist_x*tmp_dist_x
                    res = sqrt(res)
                    pcp[c_count][i2] = pc[c_count] * exp(- res / t)
                    
                    res = 0
            pcp = np.array(pcp)/np.array(pcp).sum(axis=0)

            # calculate pc
            
            if palette_size != 1:
                i2 = 0
                for i2 in prange(palette_size , nogil=True):
                    
                    c_count: int64_t = 0
                    pc[i2] = 0
                    for c_count in prange(loc_sz*loc_sz):
                        pc[i2] += pcp[i2][c_count]
                    pc[i2] = pc[i2]/(loc_sz*loc_sz)
            
            i2 = 0

            # refine palette:
            # print("Pre",palette)
            change: cython.float = 0
            for i2 in prange(palette_size , nogil=True):
                
                c_count: int64_t = 0
                res_L: cython.float = 0
                res_a: cython.float = 0
                res_b: cython.float = 0
                loc_tmp: cython.float = 0
                for c_count in prange(loc_sz*loc_sz):
                    res_L += view_colours[c_count*3]*pcp[i2][c_count]
                    res_a += view_colours[c_count*3+1]*pcp[i2][c_count]
                    res_b += view_colours[c_count*3+2]*pcp[i2][c_count]
                old: cython.float = palette_view[i2][0]

                palette_view[i2][0] = res_L/(loc_sz*loc_sz*pc[i2])
                loc_tmp += (old-palette_view[i2][0]) * (old-palette_view[i2][0])
                old = palette_view[i2][1]
                palette_view[i2][1] = res_a/(loc_sz*loc_sz*pc[i2])
                loc_tmp += (old-palette_view[i2][1]) * (old-palette_view[i2][1])
                old = palette_view[i2][2]
                palette_view[i2][2] = res_b/(loc_sz*loc_sz*pc[i2])
                loc_tmp += (old-palette_view[i2][2]) * (old-palette_view[i2][2])
                change+=sqrt(loc_tmp)
                loc_tmp = 0
            
            # calculate true palette
            q_tree.append(root_tree)
            true_palette = np.ones((2*Pixeliser.quantisation,3))*700
            disp_palette = []
            max_size = 0
            # idx_arr = []
            while not len(q_tree)==0:
                cur = q_tree.pop(0)
                if (cur.lchld.lchld == None or cur.lchld.rchld == None or cur.rchld.lchld == None or cur.rchld.rchld == None) and not (cur.lchld.lchld == None and cur.lchld.rchld == None and cur.rchld.lchld == None and cur.rchld.rchld == None):
                    print("ISSUE")
                    return
                if cur.lchld.lchld == None:
                    if (true_palette[cur.lchld.indx][0] != 700):
                        continue
                    true_palette[cur.lchld.indx][0] = (palette_view[cur.lchld.indx][0] + palette_view[cur.rchld.indx][0]) /2
                    # idx_arr.append(cur.lchld.indx)
                    # idx_arr.append(cur.rchld.indx)
                    true_palette[cur.lchld.indx][1] = (palette_view[cur.lchld.indx][1] + palette_view[cur.rchld.indx][1]) /2
                    true_palette[cur.lchld.indx][2] = (palette_view[cur.lchld.indx][2] + palette_view[cur.rchld.indx][2]) /2
                    disp_palette.append(true_palette[cur.lchld.indx])
                    true_palette[cur.rchld.indx] = true_palette[cur.lchld.indx]
                else:
                    q_tree.append(cur.lchld)
                    q_tree.append(cur.rchld)
            
            
            # idx_arr_2 = np.unique(np.array(idx_arr))
            # if idx_arr_2.shape[0] != len(idx_arr):
            #     print("ISSUE")
            #     print(idx_arr_2)
            #     print(idx_arr)
            #     return

            Pixeliser.palette_history.append(lab2rgb(np.array(disp_palette))*255)
            print(change)
            # expand:
            if change_min > change:
                t = t * 0.7
                if palette.shape[0]<2*Pixeliser.quantisation:
                    q_tree.append(root_tree)
                    
                    maximal_childrens = []
                    while not len(q_tree)==0:
                        cur = q_tree.pop(0)
                        if cur.lchld.lchld == None:
                            loc_dif = 0
                            loc_dif += (palette_view[cur.lchld.indx][0] - palette_view[cur.rchld.indx][0])**2
                            loc_dif += (palette_view[cur.lchld.indx][1] - palette_view[cur.rchld.indx][1])**2
                            loc_dif += (palette_view[cur.lchld.indx][2] - palette_view[cur.rchld.indx][2])**2
                            loc_dif = math.sqrt(loc_dif)
                            if True:
                                # l_maximal = cur.lchld
                                # r_maximal = cur.rchld

                                # # keep probabilities (divided by 2)
                                # tmp_prob = pc[r_maximal.indx]
                                # pc[r_maximal.indx] = pc[l_maximal.indx]/2
                                # pc[l_maximal.indx] = pc[l_maximal.indx]/2
                                # pc[palette_size] = tmp_prob/2
                                # pc[palette_size + 1] = tmp_prob/2

                                # # add new children
                                # tmp_chld = Colour_Tree(l_maximal.indx, l_maximal)
                                # l_maximal.lchld = tmp_chld
                                # tmp_chld = Colour_Tree(r_maximal.indx, l_maximal)
                                # l_maximal.rchld = tmp_chld

                                # tmp_chld = Colour_Tree(palette_size, r_maximal)
                                # r_maximal.lchld = tmp_chld
                                # tmp_chld = Colour_Tree(palette_size + 1, r_maximal)
                                # r_maximal.rchld = tmp_chld
                                # pca.fit([palette[r_maximal.indx],palette[l_maximal.indx]])
                                
                                # tmp_col = palette[r_maximal.indx]
                                # palette[r_maximal.indx] = palette[l_maximal.indx]-EPS_change*pca.components_[principle_choice]
                                # palette[l_maximal.indx] = palette[l_maximal.indx] + EPS_change*pca.components_[principle_choice]
                                # palette = np.append(palette, [tmp_col-EPS_change*pca.components_[principle_choice],tmp_col+EPS_change*pca.components_[principle_choice]], axis=0)
                                maximal_childrens.append([cur, loc_dif])
                                # maximal_diff = loc_dif
                                # maximal_children = cur
                        else:
                            q_tree.append(cur.lchld)
                            q_tree.append(cur.rchld)
                    # split children:
                    maximal_children =  sorted(maximal_childrens, key= lambda e: e[1], reverse= True)
                    maximal_children = [el for el in maximal_children if el[1] > Pixeliser.eps_d]
                    if len(maximal_children) >= 0:
                        # maximal_children = maximal_children[:]
                        print("N", Pixeliser.eps_d, len(maximal_children))
                    
                    for el, diff in maximal_children:
                        if palette.shape[0]>=2*Pixeliser.quantisation:
                            continue
                        l_maximal = el.lchld
                        r_maximal = el.rchld

                        # keep probabilities (divided by 2)
                        tmp_prob = pc[r_maximal.indx]
                        pc[r_maximal.indx] = pc[l_maximal.indx]/2
                        pc[l_maximal.indx] = pc[l_maximal.indx]/2
                        pc[palette_size] = tmp_prob/2
                        pc[palette_size + 1] = tmp_prob/2

                        # add new children
                        tmp_chld = Colour_Tree(l_maximal.indx, l_maximal)
                        l_maximal.lchld = tmp_chld
                        tmp_chld = Colour_Tree(r_maximal.indx, l_maximal)
                        l_maximal.rchld = tmp_chld

                        tmp_chld = Colour_Tree(palette_size, r_maximal)
                        r_maximal.lchld = tmp_chld
                        tmp_chld = Colour_Tree(palette_size + 1, r_maximal)
                        r_maximal.rchld = tmp_chld
                        pca.fit([palette[r_maximal.indx],palette[l_maximal.indx]])
                        print(pca.components_[principle_choice])
                        tmp_col = palette[r_maximal.indx]
                        palette[r_maximal.indx] = palette[l_maximal.indx]-EPS_change*pca.components_[principle_choice]
                        palette[l_maximal.indx] = palette[l_maximal.indx] + EPS_change*pca.components_[principle_choice]
                        palette = np.append(palette, [tmp_col-EPS_change*pca.components_[principle_choice],tmp_col+EPS_change*pca.components_[principle_choice]], axis=0)
                        palette_size+=2

                    
            # print(palette)
            # print("CHANGE: ", change)
            # print(np.argmax(np.array(pcp), axis = 0))
            # if len_p % 100 == 0:
            #     input()


            # colour pixels with palette
            associations = np.argmax(np.array(pcp), axis = 0).astype(int)
            i2 = 0
            associations_v: cython.int[:] = associations
            for i in range(loc_sz*loc_sz*((len_p-1)%2)):
                view_colours[i*3] = true_palette[associations_v[i]][0]
                view_colours[i*3 + 1] = true_palette[associations_v[i]][1]
                view_colours[i*3 + 2] = true_palette[associations_v[i]][2]
                
            # print(len_p)
            if (len_p%2==1):
                spr_borders = np.zeros((400*400,4))
                spr_borders_view: cython.double[:,:]= spr_borders
                for i2 in prange(n2 , nogil=True):
                    if cmask[i2] == -1:
                        continue
                    if i2 > 0 and cmask[i2 - 1] != cmask[i2]:
                        spr_borders_view[i2][0] = 205
                        spr_borders_view[i2][3] = 255

                    if i2 < n2 - 1 and cmask[i2 + 1] != cmask[i2]:
                        spr_borders_view[i2][0] = 205
                        spr_borders_view[i2][3] = 255
                    if i2 > 400 and cmask[i2 - 400] != cmask[i2]:
                        spr_borders_view[i2][0] = 205
                        spr_borders_view[i2][3] = 255
                    if i2 < n2 - 401 and cmask[i2 + 400] != cmask[i2]:
                        spr_borders_view[i2][0] = 205
                        spr_borders_view[i2][3] = 255
                Pixeliser.sprpxl_history.append(spr_borders)
            

                        
                        
                # sprpxl_history

            if t <= 1 or len_p == 2000:
                
                Pixeliser.make_img(sprpxls,true_palette,np.argmax(np.array(pcp), axis = 0))
                
        print(palette.shape)
        print(palette, 2*Pixeliser.quantisation)
        print(len_p)
    
    
    def process(cmd):
        Pixeliser.doing = True
        if cmd == "Nearest":
            Pixeliser.msg_queue.put(Pixeliser.nearest())
        elif cmd == "Cubic":
            Pixeliser.msg_queue.put(Pixeliser.cubic())
        elif cmd == "Paper":
            Pixeliser.paper()
        Pixeliser.doing = False
    def quantise():
        if Pixeliser.method == "None":
            Pixeliser.quantised = Pixeliser.glob_img
        elif Pixeliser.method == "Median cut":
            Pixeliser.quantised = Pixeliser.glob_img.quantize(Pixeliser.quantisation, method = Image.Quantize.MEDIANCUT, dither = Image.Dither.NONE)
        elif Pixeliser.method == "Max coverage":
            Pixeliser.quantised = Pixeliser.glob_img.quantize(Pixeliser.quantisation, method = Image.Quantize.MAXCOVERAGE, dither = Image.Dither.NONE)
        elif Pixeliser.method == "Fast Octree":
            Pixeliser.quantised = Pixeliser.glob_img.quantize(Pixeliser.quantisation, method = Image.Quantize.FASTOCTREE, dither = Image.Dither.NONE)
    def main(q: Queue):
        
        
        while Pixeliser.running:
            cmd = q.get(timeout=None)
            Pixeliser.process(cmd)

