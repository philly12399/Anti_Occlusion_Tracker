import numpy as np
import math
class PDF:
    def __init__(self, mean, cov, voxel_size ,noise=0.05):
        self.mean = mean
        self.cov = cov
        self.cov_inv = np.linalg.pinv(cov)
        self.noise = noise
        self.voxel_size = max(voxel_size,0.1)
        self.coef()

    def coef(self):  
        vs = self.voxel_size
        c1 = 10*(1- self.noise)
        c2 = self.noise/(vs**3)
        d3 = -np.log(c2)
        d1 = -np.log(c1 + c2) - d3
        d2 = -2 * np.log((-np.log(c1 * np.exp(-0.5) + c2) - d3) / d1)
        self.d = [0,d1,d2,d3]
        
    def pdf(self, x):  
        x = x.reshape(-1,3)        
        dx = (x - self.mean)
        factor = 1/np.sqrt(((2*np.pi)**3)*(np.linalg.det(self.cov)))
        power = -0.5*np.sum(np.multiply(np.matmul(dx, self.cov_inv), dx), axis = 1)
        p = np.exp(power)*factor
        return p
    
    def mixed_pdf(self, x):        
        x = x.reshape(-1,3)
        dx = (x - self.mean)
        ## more efficient 
        power = -0.5*self.d[2]*np.sum(np.multiply(np.matmul(dx, self.cov_inv), dx), axis = 1)
        p = -self.d[1]*np.exp(power)    
        # out=[]
        # for i in range(dx.shape[0]):
        #     p = -self.d[1]*np.exp(-0.5*self.d[2]*np.matmul(np.matmul(dx[i].T, self.cov_inv), dx[i])) 
        #     assert (p != math.inf)           
        #     out.append(p)            
        return p
 