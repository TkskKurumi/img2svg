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
def ldl2svg(loops,dots,lines,smooth=1.7,blur_dots=1.2,scale=3,cutdown_dots=10000,line_alpha=0.5,loop_stroke=True,loop_stroke_width=1.2):
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
			prt('Z" fill="RGB%s" stroke="RGB(%d,%d,%d,70%%)" stroke-width="%.1f" />'%(c,*c[:3],loop_stroke_width*scale))
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
def kmeans_with_kdt(k,points,n_iter=3,wei=None):
	import kdt
	def convert(p):
		if(isinstance(p,point)):
			return kdt.point(p.xy)
		else:
			return kdt.point(p)
	rets=random.sample(points,k)
	for iter in range(n_iter):
		print('ln98',len(rets))
		K=kdt.kdt()
		K.build([convert(_) for _ in rets])
		cnt=dict()
		sum=dict()
		for idx,i in enumerate(points):
			nn=K.ann1(convert(i))
			if(wei is None):
				cnt[nn]=cnt.get(nn,0)+1
				sum[nn]=sum.get(nn,point(0,0))+i
			else:
				cnt[nn]=cnt.get(nn,0)+wei[idx]
				sum[nn]=sum.get(nn,point(0,0))+i*wei[idx]
		rets=[]
		for i in cnt:
			rets.append(sum[i]/cnt[i])
		if(len(rets)<k):
			rets.extend(random.sample(points,k-len(rets)))
	return rets
def img2loops1(img,ss=1e5,n_colors=128,sample_color=None,n_points=None,merge_samecolor_tri=False,debug=True,merge_thresh=6,point_cut_method='kmeans',ensure_corner=True):
	w,h=img.size
	rate=(ss/w/h)**0.5
	sample_w,sample_h=int(w*rate),int(h*rate)
	simg=img.resize((sample_w,sample_h),Image.LANCZOS)
	if(sample_color is None):
		sample_color=int((n_colors*ss)**0.5)
	colors=set()
	if(debug):
		import time
		start_time=time.time()
	for i in range(sample_color):
		x=random.randrange(w)
		y=random.randrange(h)
		c=img.getpixel((x,y))
		colors.add(c)
	if(n_points is None):
		n_points=int(ss/10)
		
	colors=list(colors)
	colors=kmeans_with_weights(n_colors,colors,[1 for i in colors],n_iter=3)
	colors=[npa2tuple_color(i) for i in colors]
	def nearest(colors,c):
		ret=None
		retdist=0
		for i in colors:
			d=colordis(c,i)
			if((ret is None)or d<retdist):
				retdist=d
				ret=i
		return ret
	import kdt
	KDT=kdt.kdt()
	KDT.build([kdt.point(i) for i in colors])
	if(debug):
		import time
		print('KDT size',KDT.size)
		t=time.time()-start_time
		start_time=time.time()
		print('color kmeans use %.1f seconds'%t)
	
	for xy in wh_iter(sample_w,sample_h):
		c=simg.getpixel(xy)
		c=KDT.ann1(kdt.point(c)).arr
		simg.putpixel(xy,c)
	if(debug):
		import time
		t=time.time()-start_time
		start_time=time.time()
		print('color cutdown use %.1f seconds'%t)
		#simg.show()
	points=[]
	p_diff=[]
	for xy in wh_iter(sample_w-1,sample_h-1):
		x,y=xy
		c=simg.getpixel(xy)
		c1=simg.getpixel((x+1,y))
		c2=simg.getpixel((x,y+1))
		if(c!=c1 or c!=c2):
			points.append(point(*xy))
			if(point_cut_method in ['sort','kmeans']):
				p_diff.append(colordis(c,c1)+colordis(c,c2))
	if(debug):
		import time
		t=time.time()-start_time
		start_time=time.time()
		print('border detect use %.1f seconds'%t)
	if(n_points is None):
		n_points=int(len(points)**0.5)
	if(n_points<len(points)):
		if(debug):
			print('%d points cutdown to %d points'%(len(points),n_points))
		if(point_cut_method=='random'):
			points=random.sample(points,n_points)
		elif(point_cut_method=='sort'):
			az=[]
			for idx,p in enumerate(points):
				import heapq
				heapq.heappush(az,(p_diff[idx],random.random(),p))
				if(len(az)>int(n_points*0.9)):
					heapq.heappop(az)
			
			points=[_[-1] for _ in az]+random.sample(points,n_points-len(az))
			points=list(set(points))
		elif(point_cut_method=='kmeans'):
			
			points=kmeans_with_kdt(n_points,points,n_iter=3,wei=[i**0.8 for i in p_diff])
			#points=[point(int(p.x),int(p.y)) for p in points]
		if(debug):
			import time
			t=time.time()-start_time
			start_time=time.time()
			print('point cutdown use %.1f seconds'%t)
	if(ensure_corner):
		for x in [0,sample_w-2]:
			for y in [0,sample_h-2]:
				points.append(point(x,y))
	enmiao=1e3
	points=list(set([point(int(p.x*enmiao),int(p.y*enmiao)) for p in points]))
	print('%d points',len(points))
	points=[point(p.x/enmiao,p.y/enmiao) for p in points]
	M=mesh.delaunay(points)
	if(debug):
		import time
		t=time.time()-start_time
		start_time=time.time()
		print('delaunay use %.1f seconds'%t)
	loops=[]
	tri_points=M.get_tri_integral_point()
	def get(x,y):
		return im.getpixel((int(x),int(y)))[:3]
	tmp=lambda p:point(p.x*w/sample_w,p.y*h/sample_h)
	if(merge_samecolor_tri):
		#unfinished
		djs=DJS()
		tri_color=dict()
		_sorted=lambda x:tuple(sorted(list(x)))
		for abc,_pts in tri_points.items():
			if(not _pts):
				continue
			a,b,c=abc
			color=np.zeros((3,),np.float32)
			
			for x,y in _pts:
				color+=np.array(get(x*w/sample_w,y*h/sample_h),np.float32)
			color=npa2tuple_color(color/len(_pts))
			tri_color[abc]=color
		tri_edge=dict()
		for uv in M.edges:
			u,v=uv
			for w in M.edge2mesh[uv]:
				tri=_sorted([u,v,w])
				tri_edge[tri]=tri_edge.get(tri,set())
				tri_edge[tri].add(uv)
				tri_edge[tri].add(_sorted([u,w]))
				tri_edge[tri].add(_sorted([w,v]))
				
			if(len(M.edge2mesh[uv])==2):
				p,q=M.edge2mesh[uv]
				tri_p=_sorted([u,v,p])
				tri_q=_sorted([u,v,q])
				if(colordis(tri_color[tri_p],tri_color[tri_q])<merge_thresh):
					djs.join(tri_p,tri_q)
					tri_edge[tri_p].remove(uv)
					tri_edge[tri_q].remove(uv)
			
				
	else:
		for abc,_pts in tri_points.items():
			a,b,c=abc
			A,B,C=M.points[a],M.points[b],M.points[c]
			if(not _pts):
				CC=(A+B+C)/3
				color=get(CC.x*w/sample_w,CC.y*h/sample_h)
			else:
				
				color=np.zeros((3,),np.float32)
				
				for x,y in _pts:
					color+=np.array(get(x*w/sample_w,y*h/sample_h),np.float32)
				color=npa2tuple_color(color/len(_pts))
			
			
			loops.append((1,[tmp(A).xy,tmp(B).xy,tmp(C).xy],color))
	if(debug):
		import time
		t=time.time()-start_time
		start_time=time.time()
		print('generate loops use %.1f seconds'%t)
	print("%d loops"%len(loops))
	return loops
def img2loops(img,n_points=int(3e4),sample_points=None,sample_ss=1e6,ensure_corner=True,debug=True):
	
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
		enmiao=[]
		c=sim.getpixel((x,y))
		for dx,dy in dxy4:
			enmiao.append(colordis(sim.getpixel((x+dx,y+dy)),c))
		heapq.heappush(pts,(min(enmiao)-max(enmiao),(x,y)))
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
	im=Image.open(r"C:\Users\xiaofan\AppData\Roaming\Typora\themes\autumnus-assets\pbtEyWkQ9aI2SCK.png").convert("RGB")
	import time
	tm=time.time()
	loops=img2loops1(im)
	tm=time.time()-tm
	print("%.1f seconds"%tm)
	ww=1280
	hh=720
	w,h=im.size
	s=ldl2svg(loops,[],[],scale=min(ww/w,hh/h))
	from os import path
	with open(path.join(path.dirname(__file__),'sample.svg'),"w") as f:
		f.write(s)