class disjointset:
	def __init__(self):
		self.d=dict()
	def find(self,u):
		if(self.d.get(u,u)==u):
			return u
		self.d[u]=self.find(self.d[u])
		return self.d[u]
	def join(self,u,v):
		self.d[self.find(u)]=self.find(v)
DJS=disjointset
def normalize_edge(u,v):
    return tuple(sorted([u,v]))
class graph:    #undirected unweighted graph
    def __init__(self,neibours=None,edges=None):
        self.neibours=neibours or dict()
        self.edges=edges or set()
    def add_edge(self,u,v):
        self.neibours[u]=self.neibours.get(u,set())
        self.neibours[u].add(v)
        self.neibours[v]=self.neibours.get(v,set())
        self.neibours[v].add(u)
        edg=normalize_edge(u,v)
        self.edges.add(edg)
    def span_tree(self):   #span tree without rules
        djs=disjointset()
        ret=graph()
        fa=dict()
        root=None
        for u,neib in self.neibours.items():
            if(root is None):
                root=u
            for v in neib:
                if(djs.find(u)==djs.find(v)):
                    continue
                djs.join(u,v)
                ret.add_edge(u,v)
        #vis=dict()
        #vis[root]=True
        def dfs(u):
            nonlocal ret,fa
            for v in ret.neibours[u]:
                if(v==fa.get(u,None)):
                    continue
                
                
                fa[v]=u
                dfs(v)
        dfs(root)
        return tree(root,ret.neibours,fa,ret.edges)
class tree:
    def __init__(self,root=None,neibours=None,fa=None,edges=None):
        self.neibours=neibours or dict()
        self.fa=fa or dict()
        self.edges=edges
        self.root=root
        self.depths=dict()
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
if(__name__=='__main__'):
    g=graph()
    for i in range(0,10):
        for j in range(i+1,10):
            g.add_edge(i,j)
    
    t=g.span_tree()
    print(t.fa)
    print(t.root)
    for i in range(10):
        for j in range(i+1,10):
            print('dist(%d,%d)=%d'%(i,j,t.dist(i,j)))