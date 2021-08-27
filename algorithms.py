import random
import numpy as np
class disjointset:
	def __init__(self):
		self.d=dict()
		import sys
		sys.setrecursionlimit(1000000)
	def find(self,u):
		if(self.d.get(u,u)==u):
			return u
		self.d[u]=self.find(self.d[u])
		return self.d[u]
	def join(self,u,v):
		self.d[self.find(u)]=self.find(v)
class DJS_sum:
	def __init__(self):
		self.d=dict()
		self.s=dict()
		import sys
		sys.setrecursionlimit(1000000)
	def find(self,u):
		if(self.d.get(u,u)==u):
			return u
		v=self.d[u]
		self.s[v]=self.s.get(v,1)+self.s.get(u,1)
		self.s[u]=0
		self.d[u]=self.find(self.d[u])
		return self.d[u]
	def finds(self,u):
		return self.s.get(self.find(u),1)
	def join(self,u,v):
		self.d[self.find(u)]=self.find(v)
		self.find(u)
		self.find(v)
DJS=disjointset
def normalize_edge(u,v):
	return tuple(sorted([u,v]))
class graph:	#undirected unweighted graph
	def __init__(self,neibours=None,edges=None):
		self.neibours=neibours or dict()
		self.edges=edges or set()
	def seperate_by_connectivity(self):
		ret=dict()
		djs=DJS()
		for u,v in self.edges:
			djs.join(u,v)
		for u,v in self.edges:
			fu=djs.find(u)
			if(fu not in ret):
				ret[fu]=graph()
			ret[fu].add_edge(u,v)	   #one graph for each connected subgraph
		return [j for i,j in ret.items()]
	def add_edge(self,u,v):
		self.neibours[u]=self.neibours.get(u,set())
		self.neibours[u].add(v)
		self.neibours[v]=self.neibours.get(v,set())
		self.neibours[v].add(u)
		edg=normalize_edge(u,v)
		self.edges.add(edg)
	def farthest_path(self,s=None):
		if(s is None):
			s=list(self.neibours)[0]
			pth=self.farthest_path(s)
			s=pth[-1]
			return self.farthest_path(s)
		else:
			bfsu=[s]
			vis={s}
			fa=dict()
			bfsi=0
			while(bfsi<len(bfsu)):
				u=bfsu[bfsi]
				for v in self.neibours[u]:
					if(v in vis):
						continue
					vis.add(v)
					bfsu.append(v)
					fa[v]=u
				bfsi+=1
			t=bfsu[-1]
			ret=[]
			while(t!=s):
				ret.append(t)
				t=fa[t]
			ret.append(s)
			return ret[::-1]
	def span_tree(self,root=None):   #span tree on connected graph without rules
		djs=disjointset()
		ret=graph()
		fa=dict()
		depth=dict()
		if(root is None):
			root=list(self.neibours)[0]
		vis=set()
		vis.add(root)
		bfsu=[root]
		depth[root]=0
		i=0
		while(i<len(bfsu)):
			u=bfsu[i]
			for v in self.neibours[u]:
				if(v in vis):
					continue
				ret.add_edge(u,v)
				vis.add(v)
				fa[v]=u
				depth[v]=depth[u]+1
				bfsu.append(v)
			i+=1
		return tree(root,ret.neibours,fa,ret.edges,depth)
class tree:
	def __init__(self,root=None,neibours=None,fa=None,edges=None,depth=None):
		self.neibours=neibours or dict()
		self.fa=fa or dict()
		self.edges=edges
		self.root=root
		self.depths=depth or dict()
		self.lca_initiated=False
	def init_lca(self):
		self.anc=dict()
		flag=True
		level=0
		for u in self.neibours:
			self.anc[u]=dict()
			self.anc[u][0]=self.fa.get(u,u)
		while(flag):
			flag=False
			for u in self.neibours:
				self.anc[u][level+1]=self.anc[self.anc[u][level]][level]
				flag=flag or self.anc[u][level+1]!=self.anc[u][level]
			level+=1
			self.max_level=level
		self.lca_initiated=True
		
	def get_depth(self,u):
		if(u in self.depths):
			return self.depths[u]
		if(u not in self.fa):
			return 0
		self.depths[u]=self.get_depth(self.fa[u])+1
		return self.depths[u]
	def dist(self,u,v):
		lca=self.get_lca(u,v)
		return self.get_depth(u)+self.get_depth(v)-self.get_depth(lca)*2
	def get_lca(self,u,v):
		if(self.get_depth(u)<self.get_depth(v)):
			return self.get_lca(v,u)
		if(not self.lca_initiated):
			self.init_lca()
		tmp=self.get_depth(u)-self.get_depth(v)
		level=self.max_level
		while(tmp):
			if((1<<level)<=tmp):
				u=self.anc[u][level]
				tmp-=1<<level
			level-=1
		if(u==v):
			return u
		level=self.max_level
		while(level>=0):
			if(self.anc[u][level]!=self.anc[v][level]):
				u=self.anc[u][level]
				v=self.anc[v][level]
			level-=1
		if(self.fa.get(u,u)==self.fa.get(v,v)):
			return self.fa[u]
		raise Exception("Node not in same tree")
def kmeans_with_weights(k,points,weights,n_iter=10,az=None):
	typ=np.float32
	def asarr(i):
		if(isinstance(i,np.ndarray)):
			return i
		else:
			return np.array(i,typ)
	def dist(a,b):
		return np.linalg.norm(a-b)
	points=[asarr(i) for i in points]
	n=len(points)
	p_shape=points[0].shape
	ret=random.sample(points,k)
	idxs=list(range(k))
	if(az is None):
		az=int((k**0.3)*5)
	for i_iter in range(n_iter):
		sum_point=[np.zeros(p_shape,typ) for _ in range(n)]
		sum_weight=[0 for _ in range(n)]
		for idx,p in enumerate(points):
			best_jdx=0
			best_dist=dist(p,ret[0])
			for jdx in random.sample(idxs,az):
				_dist=dist(p,ret[jdx])
				if(_dist<best_dist):
					best_dist=_dist
					best_jdx=jdx
			sum_point[best_jdx]+=p*weights[idx]
			sum_weight[best_jdx]+=weights[idx]
		for i in range(k):
			if(sum_weight[i]):
				ret[i]=sum_point[i]/sum_weight[i]
	return ret
class wh_iter:
	def __init__(self,w,h):
		self.w=w
		self.h=h
		self.x=0
		self.y=0
	def __next__(self):
		self.y+=self.x//self.w
		self.x=self.x%self.w
		if(self.y==self.h):
			raise StopIteration
		else:
			ret=(self.x,self.y)
			self.x+=1
			return ret
	def __iter__(self):
		return self

if(__name__=='__main__'):
	for x,y in wh_iter(10,10):
		print(x,y)