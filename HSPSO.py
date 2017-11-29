#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors
np.random.seed(14526847)

#--- MAIN ---------------------------------------------------------------------+

class Particle:
	def __init__(self, ndim, search_range, ptype):
		self.position=np.empty((ndim,))  # particle position
		self.velocity=np.empty((ndim,))  # particle velocity
		self.pos_best=np.empty((ndim,))  # best position individual
		self.first_eval = True
		self.err_best=-1                  # best error individual
		self.err=-1                       # error individual
		if isinstance(search_range, list):
			search_range = np.asarray(search_range).astype(np.float32)
			shift = np.reshape((search_range[:,0] + search_range[:,1]) /2, (ndim,))
			search_range = np.reshape((search_range[:,0] - search_range[:,1]) /2, (ndim,))
			self.position = shift + np.random.uniform(-1., 1., (ndim,)) * search_range
			self.velocity = np.random.uniform(-0.5, 0.5, (ndim,)) * search_range
			self.pos_best = self.position
		else:
			self.position = np.random.uniform(-1., 1., (ndim,)) * search_range
			self.velocity = np.random.uniform(-0.5, 0.5, (ndim,)) * search_range
			self.pos_best = self.position
		self.neighbors = []
		# update_velocity_SI if type == 1 (single information)
		self.update_velocity = self.update_velocity_SI if ptype else self.update_velocity_FI

	# evaluate current fitness
	def evaluate(self,costFunc):
		self.err=costFunc(self.position)

		# check to see if the current position is an individual best
		if self.err<self.err_best or self.first_eval:
			self.first_eval = False
			self.pos_best=self.position
			self.err_best=self.err
					
	# update new particle velocity
	def update_velocity_SI(self):
		chi = 0.729			# constant downscaling
		phi = 4.1			# update weightage
		r1, r2 = np.random.uniform(0.,1.,2)

		def group_best_pos(neighbors):
			best_pos = neighbors[0].pos_best
			best_err = neighbors[0].err_best
			for n in neighbors[1:]:
				if best_err > n.err_best:
					best_err = n.err_best
					best_pos = n.pos_best
			return best_pos

		vel_self_update = phi/2*r1*(self.pos_best - self.position)
		vel_group_update = phi/2*r2*(group_best_pos(self.neighbors) - self.position)
		self.velocity = chi*(self.velocity + vel_self_update + vel_group_update)

	# update new particle velocity
	def update_velocity_FI(self):
		chi = 0.729			# constant downscaling
		phi = 4.1			# update weightage

		vel_group_update = np.zeros(self.velocity.shape)
		for n in self.neighbors:
			r = np.random.uniform(0.,1.)
			vel_group_update += phi/2*r*(n.pos_best - self.position)/len(self.neighbors)

		self.velocity = chi*(self.velocity + vel_group_update)

	# update the particle position based off new velocity updates
	def update_position(self,bounds):
		self.position=self.position+self.velocity

		# bound position if necessary
		self.position = np.clip(self.position, bounds[:,0], bounds[:,1])
		
class PSO():
	def __init__(self,costFunc,ndim,bounds,num_particles,FIfrac,nbd_sz):
		"""
		FIfrac 	:	Fraction of particles to be updated by full information PSO	
		"""
		assert (FIfrac >= 0. and FIfrac<=1.)

		self.ndim = ndim
		self.bounds = np.asarray(bounds)
		self.costFunc = costFunc
		self.nbd_sz = nbd_sz
		self.pos_best_g = None
		self.err_best_g = None
		# establish the swarm
		self.swarm=[]
		for i in range(int(num_particles*FIfrac)):
			self.swarm.append(Particle(ndim, bounds, 0))
		for i in range(num_particles - int(num_particles*FIfrac)):
			self.swarm.append(Particle(ndim, bounds, 1))

	def run(self, maxiter):
		first_eval = True
		pos_best_g = []
		err_best_g = []
		# begin optimization loop
		for i in range(maxiter):
			if i%(maxiter/10)==0:
				print 'Iteration',i
			# cycle through particles in swarm and evaluate fitness
			for p in self.swarm:
				p.evaluate(self.costFunc)

				# determine if current particle is the best (globally)
				if p.err<err_best_g or first_eval:
					first_eval = False
					pos_best_g=list(p.position)
					err_best_g=float(p.err)

			all_pos = np.zeros((len(self.swarm),self.ndim))
			for p_i,p in enumerate(self.swarm):
				all_pos[p_i,:] = np.ravel(p.position)
			knn = NearestNeighbors(n_neighbors=self.nbd_sz, n_jobs=-1)
			knn.fit(all_pos)
			kneighbors = knn.kneighbors(return_distance=False)

			# cycle through swarm and update velocities and position
			for p_i,p in enumerate(self.swarm):
				p.neighbors = []
				for n_i in range(self.nbd_sz):
					p.neighbors.append(self.swarm[kneighbors[p_i,n_i]])
				p.update_velocity()
				p.update_position(self.bounds)
				p.neighbors = []

		# print final results
		print 'FINAL:'
		print pos_best_g
		print err_best_g
		self.pos_best_g = pos_best_g
		self.err_best_g = err_best_g

#--- RUN ----------------------------------------------------------------------+

def func1(x):
	return -20*np.exp(-0.02*np.sqrt(np.sum(x**2)/float(len(x)))) + 20 + np.e - np.exp(np.sum(np.cos(2*np.pi*x))/float(len(x)))

bounds=[[-35,35],[-35,35],[-35,35],[-35,35]]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
optim = PSO(costFunc=func1,ndim=4,bounds=bounds,num_particles=2000,FIfrac=0.5,nbd_sz=10)
optim.run(1000)

#--- END ----------------------------------------------------------------------+