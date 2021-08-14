from algorithms import *
from myGeometry import polygon_area,point
import numpy as np
import random
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
def img2ldl(im,thresh=8,force_group=True,debug=False):
	
	
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
			for dx,dy in [(1,0),(1,1),(0,1),(-1,1)]:
			#for dx,dy in [(1,0),(0,1)]:
				x1=x+dx
				y1=y+dy
				if(0<=x1 and x1<w and y1<h and colordis(c,im.getpixel((x1,y1)))<thresh):
					color_groups.join((x,y),(x1,y1))
					color_groups.find((x,y))
					color_groups.find((x1,y1))
	if(force_group):
		group_pixeln=dict()
		for x in range(w):
			for y in range(h):		
				group=color_groups.find((x,y))
				group_pixeln[group]=group_pixeln.get(group,0)+1
		for x in range(w):
			for y in range(h):		
				group=color_groups.find((x,y))
				c=im.getpixel((x,y))
				if(group_pixeln[group]==1):
					nearest=None
					for dx,dy in dxy8:
						x1,y1=x+dx,y+dy
						if(0<=x1 and x1<w and 0<=y1 and y1<h):
							c1=im.getpixel((x1,y1))
							dis=colordis(c,c1)
							if((nearest is None)or dis<nearest[0]):
								nearest=(dis,(x1,y1))
								
					color_groups.join((x,y),nearest[1])
						
	
	group_edges=dict()	  #edge pixels of groups
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
				group_graphs[group].add_edge((x,y),(x1,y1))	 #candidate outline of a same color group
	loops=[]
	dots=[]
	lines=[]
	'''if(debug):
		print('ln80',sorted(list(group_edges[debug_enmiao])))
		print(group_graphs[debug_enmiao].neibours)'''
	for i in group_graphs:										  #maybe many outlines for a group. 
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
		if(largest is not None):		#loop
			area,loop=largest
			c=group_color[i]/group_pixeln[i]
			loops.append((area,loop,npa2tuple_color(c)))
		elif(False and group_pixeln[i]<=10):	   #dot
			c=group_color[i]/group_pixeln[i]
			dots.append((i,npa2tuple_color(c),group_pixeln[i]**0.5))
		else:						   #lines
			longest=None
			c=group_color[i]/group_pixeln[i]
			c=npa2tuple_color(c)
			for idx,G in enumerate(group_graphs[i]):
				pth=G.farthest_path()
				'''if(longest is None or len(pth)>longest):
					longest=pth'''
				if(len(pth)>1):
					lines.append((pth,c))
				else:
					#dots.append((pth[0],c,1))
					x,y=pth[0]
					loops.append((1,[(x,y),(x+1,y),(x+1,y+1),(x,y+1)],c))
					
			'''if(longest is not None):
				c=group_color[i]/group_pixeln[i]
				lines.append((longest,npa2tuple_color(c)))'''
	for g in group_color:
		if(g in group_graphs):
			continue
		c=group_color[g]/group_pixeln[g]
		dots.append((g,npa2tuple_color(c),1))
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
	
	ret=[]
	while(start<end):
		lower=int(start)
		az=start-lower
		pl=point(points[lower])
		pu=point(points[(lower+1)%len(points)])
		p=pu*az+pl*(1-az)
		ret.append(p.xy)
		start+=step
	if(len(ret)<10):
		return points
	return ret
def ldl2svg(loops,dots,lines,smooth=1.7,blur_dots=1.2,scale=3,cutdown_dots=10000,line_alpha=0.5,loop_stroke=False):
	out=""
	def prt(*args,end='\n'):
		nonlocal out
		out+=" ".join([str(i) for i in args])
		out+=end
	if(len(dots)>cutdown_dots):
		blur_dots*=(len(dots)/cutdown_dots)**0.5
		blur_dots=min(blur_dots,2.5)
		dots=random.sample(dots,cutdown_dots)
	prt('<?xml version="1.0" standalone="no"?>')
	prt('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"')
	prt('"http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">')
	prt('<svg xmlns="http://www.w3.org/2000/svg" version="1.1">')
	for loop in loops:
		area,points,c=loop
		points=smooth_points(points,smooth)
		prt('<path d="',end='')
		f="M"
		for x,y in points:
			prt(f,end='')
			prt("%.2f"%(x*scale),"%.2f"%(y*scale),end=' ')
			f="L"
		if(loop_stroke):
			prt('Z" fill="RGB%s" stroke="RGB(%d,%d,%d,70%%)" stroke-width="%.1f" />'%(c,*c[:3],2*scale))
		else:
			prt('Z" fill="RGB%s" stroke="none" />'%(c,))
	for i in dots:
		xy,c,rad=i
		x,y=xy
		rad=rad/1.2
		prt('<circle cx="%.1f" cy="%.1f" r="%.2f" fill="RGB(%d,%d,%d,%d%%)"/>'%(x*scale,y*scale,rad*blur_dots*scale,*c[:3],70/blur_dots/blur_dots))
	for line in lines:
		points,c=line
		points=smooth_points(points,smooth)
		prt('<path d="',end='')
		f="M"
		for x,y in points:
			prt(f,end='')
			prt("%.2f"%(x*scale),"%.2f"%(y*scale),end=' ')
			f="L"
		prt('Z" stroke="RGBA(%d,%d,%d,%.1f%%)" fill="none" stroke-width="%.1f" />'%(*c[:3],100*line_alpha,1.5*scale))
	for line in lines:
		points,c=line
		points=smooth_points(points,smooth)
		prt('<path d="',end='')
		f="M"
		for x,y in points:
			prt(f,end='')
			prt("%.2f"%(x*scale),"%.2f"%(y*scale),end=' ')
			f="L"
		prt('Z" stroke="RGBA(%d,%d,%d,%d)" fill="none" stroke-width="%.1f" />'%(*c[:3],50*line_alpha,3*scale))
	
	prt("</svg>",end='')
	return out
if(__name__=='__main__'):
	secs=30
	speed=13000
	for ss in (speed*secs,):
		im=Image.open(r"D:\img2svg\QQ截图20210814002508.png")
		w,h=im.size
		rate=(ss/w/h)**0.5
		
		w=int(w*rate)
		h=int(h*rate)
		im1=im.resize((w,h))
		import time
		tm=time.time()
		
		loops,dots,lines=img2ldl(im1,debug=False)
		s=ldl2svg(loops,dots,lines,scale=min(1200/w,700/h))
		from os import path
		with open(path.join(path.dirname(__file__),'sample.svg'),"w") as f:
			f.write(s)
		tm=time.time()-tm
		print("%.2f secs,%d pixs per sec"%(tm,w*h/tm))
		print(len(loops),'loops')
		print(len(lines),'lines')
		print(len(dots),'dots')
		im1.show()
		continue
		im2=Image.new("RGB",im1.size)
		dr=ImageDraw.Draw(im2)
		for loop in loops:
			area,points,c=loop
			points=smooth_points(points)
			dr.polygon(points,fill=c)
		for xy,c,siz in dots:
			x,y=xy
			siz=siz
			az=x-siz,x+siz,y-siz,y+siz
			x1,x2,y1,y2=[int(i) for i in az]
			dr.ellipse((x1,y1,x2,y2),fill=c)
		for line in lines:
			points,c=line
			points=smooth_points(points)
			dr.line(points,fill=c,width=4)
		im2.show()
	print(ldl2svg([(1,((0,0),(1,0),(1,1),(0,1)),(255,100,120))],[],[]))