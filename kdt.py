from myGeometry import *
import random
class point:
	def __init__(self,arr):
		self.arr=arr
	def dist(self,other):
		ret=0
		for idx,i in enumerate(self.arr):
			try:
				ret+=(i-other.arr[idx])*(i-other.arr[idx])
			except Exception as e:
				print(type(other),str(other)[:20])
				raise e
		return ret
	def dist_line(self,A,B):
		pass
class kdt:
	def __init__(self):
		self.node_points=[]
		self.left_child=[]
		self.right_child=[]
		self.left_point=[]
		self.right_point=[]
		self.root=None
		self.size=0
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
	def build(self,points,stop_num=3,select_point='random',depth=0,stop_depth=40):
		u=self._new_index()
		if(len(points)<=stop_num or depth>stop_depth):
			self.node_points[u]=points
			return u
		if(select_point=='random'):
			lpoint,rpoint=random.sample(points,2)
		lpoints=[]
		rpoints=[]
		for p in points:
			if(p.dist(lpoint)<p.dist(rpoint)):
				lpoints.append(p)
			else:
				rpoints.append(p)
		self.left_point[u]=lpoint
		self.right_point[u]=rpoint
		self.left_child[u]=self.build(lpoints,select_point=select_point,stop_num=stop_num,depth=depth+1)
		self.right_child[u]=self.build(rpoints,select_point=select_point,stop_num=stop_num,depth=depth+1)
		return u
	def ann1(self,p,u=None):
		if(u is None):
			return self.ann1(p,u=self.root)
		if(self.left_point[u] is None):
			ret=None
			for i in self.node_points[u]:
				if((ret is None) or(p.dist(ret)<p.dist(i))):
					return i
			return ret
		if(p.dist(self.left_point[u])<p.dist(self.right_point[u])):
			ret=self.ann1(p,self.left_child[u])
			#if(ret.dist(p)
			return ret
		else:
			return self.ann1(p,self.right_child[u])
		