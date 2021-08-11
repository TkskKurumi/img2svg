from algorithms import *
from myGeometry import polygon_area,point
import numpy as np
import random
from PIL import Image,ImageDraw
from delaunay_mesh import mesh
def colordis(a,b):
	return np.linalg.norm(np.array(a)-np.array(b))
def npa2tuple_color(arr):
	return tuple([int(i) for i in arr])
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
dx8=[1,1,0,-1,-1,-1,0,1]
dy8=[0,1,1,1,0,-1,-1,-1]
dxy8=[(dx8[i],dy8[i]) for i in range(8)]
dx4=[0,0,1,-1]
dy4=[1,-1,0,0]
dxy4=[(dx4[i],dy4[i]) for i in range(4)]
def img2loops(img,n_points=int(4e4),sample_points=None,sample_ss=1e6,ensure_corner=True,debug=True):
	
	import heapq
	w,h=img.size
	rate=(sample_ss/w/h)**0.5
	
	sample_w,sample_h=int(w*rate),int(h*rate)
	sim=img.resize((sample_w,sample_h),Image.LANCZOS)
	pts=[]
	pts1=[]
	tmp=list(wh_iter(sample_w-2,sample_h-2))
	if(sample_points is None):
		sample_points=int((n_points*sample_ss)**0.5)
	if(debug):tm=__import__('time').time()
	for x,y in random.sample(tmp,sample_points):
		x=x+1
		y=y+1
		enmiao=0
		c=sim.getpixel((x,y))
		for dx,dy in dxy4:
			enmiao+=colordis(sim.getpixel((x+dx,y+dy)),c)
		heapq.heappush(pts,(-enmiao,(x,y)))
		if(len(pts)>int(n_points*0.8)):
			heapq.heappop(pts)
		pts1.append((x,y))
	pts=[_[1] for _ in pts]+random.sample(pts1,n_points-len(pts))
	pts=list(set(pts))
	if(debug):
		print('init',__import__('time').time()-tm)
		tm=__import__('time').time()
	
	'''pts=kmeans_with_weights(n_points,pts,wei,n_iter=2)
	if(debug):
		print('kmeans',__import__('time').time()-tm)
		tm=__import__('time').time()'''
	pts=[point(i[0]*w/sample_w,i[1]*h/sample_h) for i in pts]
	#print(pts)
	
	M=mesh.delaunay(pts)
	if(debug):
		print('mesh',__import__('time').time()-tm)
		tm=__import__('time').time()
		#M.illust().show()
	loops=[]
	tri_points=M.get_tri_integral_point()
	def get(x,y):
		return im.getpixel((int(x),int(y)))[:3]
	debug_f=True
	for abc,_pts in tri_points.items():
		if(not _pts):
			continue
		a,b,c=abc
		color=np.zeros((3,),np.float32)
		
		for x,y in _pts:
			color+=np.array(get(x,y),np.float32)
		color=npa2tuple_color(color/len(_pts))
		A,B,C=M.points[a],M.points[b],M.points[c]
		loops.append((1,[A.xy,B.xy,C.xy],color))
	return loops
if(__name__=='__main__'):
	im=Image.open(r"C:\Users\xiaofan\AppData\Roaming\Typora\themes\autumnus-assets\WPxSwEYVtfm6Ba1.png")
	import time
	tm=time.time()
	loops=img2loops(im)
	tm=time.time()-tm
	print("%.1f seconds"%tm)
	s=ldl2svg(loops,[],[],scale=0.3)
	from os import path
	with open(path.join(path.dirname(__file__),'sample.svg'),"w") as f:
		f.write(s)