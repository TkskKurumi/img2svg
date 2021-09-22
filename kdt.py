from myGeometry import point as gPoint
import random
from algorithms import quick_rank
import numpy as np
_inf=float('inf')
class point:
	def __init__(self,arr):
		self.arr=arr
		self.hash=None
		self.id=None
	def dist(self,other):
		
		ret=0
		
		for idx,i in enumerate(self.arr):
			ret+=(i-other.arr[idx])**2
		return ret
	
	def __str__(self):
		return str(self.arr)
	def __repr__(self):
		return self.arr.__repr__()
	def __eq__(self,other):
		return hash(self)==hash(other)
	def __hash__(self):
		if(self.hash is not None):
			return self.hash
		if(isinstance(self.arr,gPoint)):
			return hash(self.arr)
		elif(isinstance(self.arr,np.ndarray)):
			
			enmiao=1e2
			h=hash(tuple([int(i*enmiao) for i in self.arr]))
			self.hash=h
			return h
		else:
			return hash(self.arr)
	def aspoint(self):
		return gPoint(self.arr[0],self.arr[1])
	def __mul__(self,other):
		if(isinstance(other,point)):
			ret=0
			for idx,i in enumerate(self.arr):
				ret+=i*other.arr[idx]
			return ret
		elif(isinstance(other,int) or isinstance(other,float)):
			arr=[i*other for idx,i in enumerate(self.arr)]
			return point(arr)
		return NotImplemented
	def __add__(self,other):
		arr=[i+other.arr[idx] for idx,i in enumerate(self.arr)]
		return point(arr)
	def __sub__(self,other):
		arr=[i-other.arr[idx] for idx,i in enumerate(self.arr)]
		return point(arr)
	def __truediv__(self,other):
		if(isinstance(other,int) or isinstance(other,float)):
			arr=[i/other for idx,i in enumerate(self.arr)]
			return point(arr)
		else:
			return NotImplemented
	def distO(self):
		return sum([i*i for i in self.arr])**0.5
	def dist_line(self,A,B):
		AB=B-A
		AP=self-A
		dot=AB*AP
		dAP=AP.distO()
		if(AB.distO()<1e-8):
			return dAP
		#assert dAP>=0,str(dAP)
		#assert AB.distO()>0,str(AB.distO())
		dAH=dot/AB.distO() #AH
		tmp=dAP*dAP-dAH*dAH
		assert tmp>-1e-8
		if(tmp<0):
			return 0
		return abs(tmp**0.5)
	def dist_bisector(self,A,B):
		C=(A+B)/2
		PC=C-self
		AB=B-A
		dot=AB*PC
		return abs(dot/AB.distO())
def variance(ls):
	avg=sum(ls)/len(ls)
	ret=0
	for i in ls:
		ret+=(i-avg)**2
	return ret
class kdt:
	def __init__(self):
		self.node_points=[]
		self.left_child=[]
		self.right_child=[]
		self.axis=[]
		self.value=[]
		self.root=None
	def _new_index(self):
		self.node_points.append(None)
		self.left_child.append(None)
		self.right_child.append(None)
		self.axis.append(None)
		self.value.append(None)
		if(self.root is None):
			self.root=len(self.node_points)-1
		self.size=len(self.node_points)
		return self.size-1
	def initiate_statics(self):
		self._cnt_call_ann_top=0
		self._cnt_call_ann_recursive=0
		self._cnt_calc_dist=0
		self._sum_leaf=0
		self._sum_leaf_depth=0
	def print_performance(self):
		print("call ann",self._cnt_call_ann_recursive/self._cnt_call_ann_top)
		print("calc dist",self._cnt_calc_dist/self._cnt_call_ann_top)
	def build(self,points,stop_num=1,depth=0,stop_depth=20):
		#print(len(points),depth)
		if(depth==0):
			
			self.initiate_statics()
			points=list(set(points))
			for idx,i in enumerate(points):
				if(not isinstance(i,point)):
					points[idx]=point(i)
			for id,i in enumerate(points):
				i.id=id
		u=self._new_index()
		if(len(points)<=stop_num or stop_depth<=depth):
			self.node_points[u]=points
			self._sum_leaf+=1
			self._sum_leaf_depth+=depth
			return u
		
		nd=len(points[0].arr)
		vars=[]
		for i in range(nd):
			ls=[p.arr[i] for p in points]
			var=variance(ls)
			vars.append((var,i))
		_,axis=max(vars)
		ls=[p.arr[axis] for p in points]
		value=quick_rank(ls,(len(ls)-1)//2)
		self.axis[u]=axis
		self.value[u]=value
		lpoints=list()
		rpoints=list()
		for p in points:
			if(p.arr[axis]<=value):
				lpoints.append(p)
			else:
				rpoints.append(p)
		if((not lpoints) or (not rpoints)):
			self.axis[u]=None
			self.value[u]=None
			self.node_points[u]=points
			return u
		self.left_child[u]=self.build(lpoints,stop_num=stop_num,depth=depth+1,stop_depth=stop_depth)
		self.right_child[u]=self.build(rpoints,stop_num=stop_num,depth=depth+1,stop_depth=stop_depth)
		return u
	def ann1(self,p,u=None,cut_dist=_inf):
		
		if(u is None):
			self._cnt_call_ann_top+=1
			ret = self.ann1(p,u=self.root)
			assert ret is not None
			return ret
		self._cnt_call_ann_recursive+=1
		#leaf node
		if(self.node_points[u] is not None):
			ret=None
			retd=None
			for i in self.node_points[u]:
				if(ret is None):
					ret=i
					self._cnt_calc_dist+=1
					retd=p.dist(ret)
				self._cnt_calc_dist+=1
				id=p.dist(i)
				if(id<retd):
					ret=i
					retd=id
			assert (ret is not None),'ret is None,%s'%self.node_points[u]
			return ret
		#print(self.axis[u],self.value[u],self.node_points[u],u)
		axis=self.axis[u]
		value=self.value[u]
		if(p.arr[axis]<=value):
			ret=self.ann1(p,self.left_child[u],cut_dist=cut_dist)
			self._cnt_calc_dist+=1
			retd=ret.dist(p)
			if(min(retd,cut_dist)<abs(p.arr[axis]-value)):
				ret1=self.ann1(p,self.right_child[u],cut_dist=min(retd,cut_dist))
				self._cnt_calc_dist+=1
				ret1d=ret1.dist(p)
				if(ret1d<retd):
					assert (ret1 is not None)
					return ret1
				else:
					assert (ret is not None)
					return ret
			else:
				return ret
		else:
			ret = self.ann1(p,self.right_child[u],cut_dist=cut_dist)
			assert (ret is not None),"%s,%s"%(self.right_child[u],self.node_points[self.right_child[u]])
			self._cnt_calc_dist+=1
			retd=ret.dist(p)
			if(min(retd,cut_dist)<abs(p.arr[axis]-value)):
				ret1=self.ann1(p,self.left_child[u],cut_dist=min(retd,cut_dist))
				self._cnt_calc_dist+=1
				ret1d=ret1.dist(p)
				if(ret1d<retd):
					assert (ret1 is not None)
					return ret1
				else:
					assert (ret is not None)
					return ret
			else:
				return ret
			
class _kdt:
	def __init__(self):
		self.node_points=[]
		self.left_child=[]
		self.right_child=[]
		self.left_point=[]
		self.right_point=[]
		self.root=None
		self.size=0
		self._sum_depth=0
		self._sum_calc=0
		self._ann_top=0
		self._ann=0
		self._dist=0
		self._recall=0
	def _new_index(self):
		self.node_points.append(None)
		self.left_child.append(None)
		self.right_child.append(None)
		self.left_point.append(None)
		self.right_point.append(None)
		if(self.root is None):
			self.root=len(self.node_points)-1
		self.size=len(self.node_points)
		return self.size-1
	@property
	def avg_depth(self):
		return self._sum_depth/self._sum_points
	@property
	def avg_calc(self):
		return self._sum_calc/self._sum_points
	def print_performance(self):
		print("recall",self._recall/self._ann_top)
		print("nodes",self._ann/self._ann_top,len(self.node_points))
		print("calc dist",self._dist/self._ann_top)
	def build(self,points,stop_num=1,select_point='random',depth=0,stop_depth=20,set_id=True):
		if(set_id):
			points=list(set(points))
			for id,i in enumerate(points):
				i.id=id
		if(depth==0):
			
			self._sum_depth=0
			self._sum_points=len(points)
		u=self._new_index()
		if(len(points)<=stop_num or depth>stop_depth):
			self.node_points[u]=points
			self._sum_depth+=len(points)*depth
			self._sum_calc+=(2*depth+len(points))*len(points)	#compare lr son, compare leaf points
			return u
		#2means
		lpoint,rpoint=random.sample(points,2)
		for i in range(10):
			lp1=point([0 for _ in lpoint.arr])
			rp1=point([0 for _ in lpoint.arr])
			ln=0
			rn=0
			for i in points:
				if(i.dist(lpoint)<i.dist(rpoint)):
					ln+=1
					lp1=lp1+i
				else:
					rn+=1
					rp1=rp1+i
			lpoint=lp1/ln
			rpoint=rp1/rn
		lpoints=[]
		rpoints=[]
		for p in points:
			if(p.dist(lpoint)<p.dist(rpoint)):
				lpoints.append(p)
			else:
				rpoints.append(p)
		assert lpoints,"%s,%s,%s,%s"%(lpoint,rpoint,points,lpoints)
		assert rpoints
		self.left_point[u]=lpoint
		self.right_point[u]=rpoint
		self.left_child[u]=self.build(lpoints,select_point=select_point,stop_num=stop_num,depth=depth+1,set_id=False)
		self.right_child[u]=self.build(rpoints,select_point=select_point,stop_num=stop_num,depth=depth+1,set_id=False)
		return u
	def ann1(self,p,u=None,cut_dist=_inf):
		self._ann+=1
		if(u is None):
			self._ann_top+=1
			return self.ann1(p,u=self.root)
		if(self.left_point[u] is None):
			ret=None
			retd=None
			for i in self.node_points[u]:
				if(ret is None):
					ret=i
					self._dist+=1
					retd=p.dist(ret)
				self._dist+=1
				id=p.dist(i)
				if(id<retd):
					ret=i
					retd=id
			assert ret is not None,'ret is None,%s'%self.node_points[u]
			return ret
		self._dist+=2
		if(p.dist(self.left_point[u])<p.dist(self.right_point[u])):
			ret=self.ann1(p,self.left_child[u])
			self._dist+=2
			dret=ret.dist(p)
			if(min(dret,cut_dist)>p.dist_bisector(self.left_point[u],self.right_point[u])):
				ret1=self.ann1(p,self.right_child[u],min(dret,cut_dist))
				self._recall+=1
				self._dist+=1
				if(ret1.dist(p)<dret):
					return ret1
				else:
					return ret
			else:
				return ret
		else:
			ret=self.ann1(p,self.right_child[u])
			self._dist+=2
			dret=ret.dist(p)
			if(min(dret,cut_dist)>p.dist_bisector(self.left_point[u],self.right_point[u])):
				ret1=self.ann1(p,self.left_child[u],min(dret,cut_dist))
				self._recall+=1
				self._dist+=1
				if(ret1.dist(p)<dret):
					return ret1
				else:
					return ret
			else:
				return ret
if(__name__=='__main__'):
	import random
	def rand_nd(n):
		return tuple([random.random() for i in range(n)])
	nd=10
	num=1024
	points=[point(rand_nd(nd)) for i in range(num)]
	def find_nearest_basic():
		p=point(rand_nd(nd))
		ret=None
		retd=0
		for i in points:
			if(ret is None):
				ret=i
				retd=i.dist(p)
			id=i.dist(p)
			if(id<retd):
				retd=id
				ret=i
	K=kdt()
	K.build(points)
	def find_nearest_kdt():
		p=point(rand_nd(nd))
		K.ann1(p)
	
	from timeit import timeit
	print(timeit(stmt=find_nearest_basic,number=100))
	print(timeit(stmt=find_nearest_kdt,number=100))
	K.print_performance()