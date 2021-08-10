from algorithms import *
from myGeometry import polygon_area,point
import numpy as np
from PIL import Image,ImageDraw
def colordis(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))
dx4=[0,0,1,-1]
dy4=[1,-1,0,0]
dxy4=[(dx4[i],dy4[i]) for i in range(4)]

dx8=[1,1,0,-1,-1,-1,0,1]
dy8=[0,1,1,1,0,-1,-1,-1]
dxy8=[(dx8[i],dy8[i]) for i in range(8)]

dxy3=[(1,0),(0,1),(1,1)]
print(dxy4,dxy8)
def npa2tuple_color(arr):
    return tuple([int(i) for i in arr])
def img2ldl(im,thresh=3,debug=False):
    color_groups=disjointset()
    
    w,h=im.size
    group_color=dict()
    group_pixeln=dict()
    for x in range(w):
        for y in range(h):
            c=im.getpixel((x,y))
            c=np.array(c,np.float32)
            group=color_groups.find((x,y))
            if(group not in group_color):
                group_color[group]=np.zeros(c.shape,np.float32)
            group_color[group]+=c
            group_pixeln[group]=group_pixeln.get(group,0)+1
            c=group_color[group]/group_pixeln[group]
            #c=im.getpixel(group)
            #for dx,dy in [(1,0),(1,1),(0,1),(-1,1)]:
            for dx,dy in [(1,0),(0,1)]:
                x1=x+dx
                y1=y+dy
                if(0<=x1 and x1<w and y1<h and colordis(c,im.getpixel((x1,y1)))<thresh):
                    color_groups.join((x,y),(x1,y1))
                    color_groups.find((x,y))
                    color_groups.find((x1,y1))
            
    
    group_edges=dict()      #edge pixels of groups
                            #pixels!! not the edge(u,v) in graph
    group_color=dict()
    group_pixeln=dict()
    
    for x in range(w):
        for y in range(h):
            c=im.getpixel((x,y))
            c=np.array(c,np.float32)
            group=color_groups.find((x,y))
            if(group not in group_color):
                group_color[group]=np.zeros(c.shape,np.float32)
            group_color[group]+=c
            group_pixeln[group]=group_pixeln.get(group,0)+1
            group=color_groups.find((x,y))
            for dx,dy in dxy4:
                x1,y1=x+dx,y+dy
                group1=color_groups.find((x1,y1))
                if(group!=group1):
                    if(group not in group_edges):
                        group_edges[group]=set()
                    group_edges[group].add((x,y))
    if(debug and False):
        debug_enmiao=list(group_edges)[0]
        for i in group_edges:
            enmiao=Image.new(im.mode,im.size,(0,)*3)
            for x,y in group_edges[i]:
                enmiao.putpixel((x,y),(255,255,255))
            enmiao.show()
            break
        enmiao=Image.new(im.mode,im.size,(0,)*3)
        for x in range(w):
            for y in range(h):
                if(color_groups.find((x,y))==i):
                    enmiao.putpixel((x,y),(255,255,255))
        enmiao.show()
    group_graphs=dict()
    for x in range(w):
        for y in range(h):
            
            group=color_groups.find((x,y))
            if((x,y) not in group_edges.get(group,set())):
                continue
            for dx,dy in dxy8:
                x1,y1=x+dx,y+dy
                group1=color_groups.find((x1,y1))
                if(debug and (x,y)==(0,0) and (x1,y1)==(0,1)):
                    print('ln74',group,group1,(x1,y1) in group_edges.get((x1,y1),set()))
                if((x1,y1) not in group_edges.get(group1,set())):
                    continue
                if(group1!=group):
                    continue
                if(group not in group_graphs):
                    group_graphs[group]=graph()
                group_graphs[group].add_edge((x,y),(x1,y1))     #candidate outline of a same color group
    loops=[]
    dots=[]
    lines=[]
    '''if(debug):
        print('ln80',sorted(list(group_edges[debug_enmiao])))
        print(group_graphs[debug_enmiao].neibours)'''
    for i in group_graphs:                                          #maybe many outlines for a group. 
        #print(group_graphs[i].edges)
        group_graphs[i]=group_graphs[i].seperate_by_connectivity()  #for something look like "回" may have outer and inner edge
        
        largest=None
        for idx,G in enumerate(group_graphs[i]):
            #look for a loop in the graph
            #can't think of an algorithm can find biggest(in area) loop fast.
            spt=G.span_tree()
            #print(G.edges)
            longest=None
            for edg in G.edges:
                if(edg in spt.edges):
                    continue
                u,v=edg
                dist=spt.dist(u,v)
                if(longest is None or (dist,edg)>longest):
                    longest=(dist,edg)
            if(longest is None):
                continue
            dist,edg=longest
            u,v=edg
            pathu=[]
            pathv=[]
            while(u!=v):
                if(spt.get_depth(u)>spt.get_depth(v)):
                    pathu.append(u)
                    u=spt.fa[u]
                elif(spt.get_depth(u)<spt.get_depth(v)):
                    pathv.append(v)
                    v=spt.fa[v]
                else:
                    pathu.append(u)
                    u=spt.fa[u]
                    pathv.append(v)
                    v=spt.fa[v]
            loop=pathu
            loop.append(u)
            loop.extend(pathv[::-1])
            area=polygon_area([point(i) for i in loop])
            if((largest is None)or ((area,loop)>largest)):
                largest=(area,loop)
        if(largest is not None):        #loop
            area,loop=largest
            c=group_color[i]/group_pixeln[i]
            loops.append((area,loop,npa2tuple_color(c)))
        elif(False and group_pixeln[i]<=4):       #dot
            c=group_color[i]/group_pixeln[i]
            dots.append((i,npa2tuple_color(c),group_pixeln[i]**0.5))
        else:                           #lines
            longest=None
            c=group_color[i]/group_pixeln[i]
            c=npa2tuple_color(c)
            for idx,G in enumerate(group_graphs[i]):
                pth=G.farthest_path()
                '''if(longest is None or len(pth)>longest):
                    longest=pth'''
                if(len(pth)>1):
                    lines.append((pth,c))
            '''if(longest is not None):
                c=group_color[i]/group_pixeln[i]
                lines.append((longest,npa2tuple_color(c)))'''
    if(debug):
        enmiao=Image.new(im.mode,im.size)
        az=dict()
        for x in range(w):
            for y in range(h):
                g=color_groups.find((x,y))
                #c=npa2tuple_color(group_color[g]/group_pixeln[g])
                ammiao=hash(g)&((1<<12)-1)
                amiaom=lambda x:(255//3)*x
                c=(amiaom(ammiao&3),amiaom((ammiao&12)>>2),amiaom((ammiao&48)>>2))
                enmiao.putpixel((x,y),tuple(c))
                az[g]=az.get(g,0)+1
        #print('ln142',az)
        enmiao.show()
    return sorted(loops,key=lambda x:-x[0]),dots,lines
def smooth_points(points,step=2.4,start=0,end=None):
    if(end is None):
        end=len(points)
    if(end-start<step*step):
        return points
    ret=[]
    while(start<end):
        lower=int(start)
        az=start-lower
        pl=point(points[lower])
        pu=point(points[(lower+1)%len(points)])
        p=pu*az+pl*(1-az)
        ret.append(p.xy)
        start+=step
    return ret
def ldl2svg(loops,dots,lines,smooth=2.4):
    pass
if(__name__=='__main__'):
    for ss in (1e5,1e6):
        im=Image.open(r"C:\Users\xiaofan\AppData\Roaming\Typora\themes\autumnus-assets\5XEclgMCLtPhfSb.jpg")
        w,h=im.size
        rate=(ss/w/h)**0.5
        #rate=1
        w=int(w*rate)
        h=int(h*rate)
        im1=im.resize((w,h))
        import time
        tm=time.time()
        loops,dots,lines=img2ldl(im1,debug=False)
        tm=time.time()-tm
        print("%.2f secs,%d pixs per sec"%(tm,ss/tm))
        print(len(loops),'loops')
        print(len(lines),'lines')
        print(len(dots),'dots')
        #im1.show()
        im2=Image.new("RGB",im1.size)
        dr=ImageDraw.Draw(im2)
        for xy,c,siz in dots:
            x,y=xy
            siz=siz*2
            az=x-siz,x+siz,y-siz,y+siz
            x1,x2,y1,y2=[int(i) for i in az]
            dr.ellipse((x1,y1,x2,y2),fill=c)
        
        for loop in loops:
            area,points,c=loop
            points=smooth_points(points)
            dr.polygon(points,fill=c)
        for line in lines:
            points,c=line
            points=smooth_points(points)
            dr.line(points,fill=c,width=4)
        im2.show()