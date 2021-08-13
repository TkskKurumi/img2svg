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
def ldl2svg(loops,dots,lines,smooth=1.7,blur_dots=1.2,scale=3,cutdown_dots=10000,line_alpha=0.5,loop_stroke=True,loop_stroke_width=1.2,loop_trim=True):
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
		xs=[x for x,y in points]
		ys=[y for x,y in points]
		dx=max(xs)-min(xs)
		dy=max(ys)-min(ys)
		az=max(dx,dy)/((min(dx,dy)+0.1)**0.5)
		if(loop_trim and az>30):
			if(dx<loop_stroke_width and dy<loop_stroke_width):
				continue
		
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
def kmeans_with_kdt(k,points,n_iter=3,wei=None,progress_cb=None):
	import kdt
	def convert(p):
		if(isinstance(p,point)):
			return kdt.point(p.xy)
		else:
			return kdt.point(p)
	n=len(points)
	rets=random.sample(points,k)
	for iter in range(n_iter):
		#print('ln98',len(rets))
		K=kdt.kdt()
		K.build([convert(_) for _ in rets])
		cnt=dict()
		sum=dict()
		for idx,i in enumerate(points):
			if(progress_cb):
				progress_cb((iter*n+idx)/n/n_iter)
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
def img2loops(img,ss=1e5,n_colors=64,sample_color=None,n_points=None,merge_samecolor_tri=False,debug=True,merge_thresh=6,point_cut_method='none',ensure_corner=True,smooth_points=3,print_progress=True):
	w,h=img.size
	rate=(ss/w/h)**0.5
	sample_w,sample_h=int(w*rate),int(h*rate)
	simg=img.resize((sample_w,sample_h),Image.LANCZOS)
	if(sample_color is None):
		sample_color=int((n_colors*ss)**0.5)
	colors=set()
	import time
	last_prog=time.time()
	i_prog=0
	first_prog=dict()
	def progbar(title,prog,width=20):
		nonlocal i_prog,last_prog
		i_prog+=1
		if(i_prog&0b111):
			return
		t=time.time()
		if(title not in first_prog):
			first_prog[title]=t
		if(t-last_prog>0.1):
			enmiao="#"*int(prog*width)
			enmiao+="."*max(width-len(enmiao),0)
			remain=(t-first_prog[title])/(prog+1e-10)*(1-prog)
			print(title,"["+enmiao+"] %.1f seconds remain"%(remain),end='\r')
			last_prog=t
	if(debug):
		import time
		start_time=time.time()
	for i in range(sample_color):
		if(print_progress):
			progbar("sampling color",i/sample_color)
		x=random.randrange(w)
		y=random.randrange(h)
		c=img.getpixel((x,y))
		colors.add(c)
	if(n_points is None):
		n_points=int(ss/13)
		
	colors=list(colors)
	colors=kmeans_with_weights(n_colors,colors,[1 for i in colors],n_iter=4)
	
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
		x,y=xy
		if(print_progress):
			progbar("merge color",(y*sample_w+x)/sample_w/sample_h)
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
	for xy in wh_iter(sample_w-2,sample_h-2):
		x,y=xy
		x=x+1
		y=y+1
		if(print_progress):
			progbar("detect border",(y*(sample_w-1)+x)/(sample_w-1)/(sample_h-1))
		c=simg.getpixel((x,y))
		c1=simg.getpixel((x,y+1))
		c2=simg.getpixel((x+1,y))
		c3=simg.getpixel((x,y-1))
		c4=simg.getpixel((x-1,y))
		
		#cs=[c,c1,c2,c3]
		cs=[c,c1,c2,c3,c4]
		cd=0
		cm=float('inf')
		for i in range(len(cs)):
			for j in range(i+1,len(cs)):
				_cd=colordis(cs[i],cs[j])
				cd+=_cd
				cm=min(cm,_cd)
		cd+=cm*len(cs)*(len(cs)-1)/2
		if(cd):
			points.append(point(*xy))
			if(point_cut_method in ['sort','kmeans']):
				p_diff.append(cd)
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
				if(print_progress):
					progbar("merge points",idx/len(points))
				import heapq
				heapq.heappush(az,(p_diff[idx],random.random(),p))
				if(len(az)>int(n_points*0.9)):
					heapq.heappop(az)
			
			points=[_[-1] for _ in az]+random.sample(points,n_points-len(az))
			points=list(set(points))
		elif(point_cut_method=='kmeans'):
			if(print_progress):
				cb=lambda prog:progbar("merge points",prog)
			else:
				cb=None
			points=kmeans_with_kdt(n_points,points,n_iter=4,wei=[i**8 for i in p_diff],progress_cb=cb)
			#points=kmeans_with_kdt(n_points,points,n_iter=4)
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
	
	enmiao=1e2
	points=list(set([point(int(p.x*enmiao),int(p.y*enmiao)) for p in points]))
	print('%d points'%len(points))
	points=[point(p.x/enmiao,p.y/enmiao) for p in points]
	if(print_progress):
		cb=lambda prog:progbar("delaunay",prog)
	else:
		cb=None
	M=mesh.delaunay(points,prog_cb=cb)
	'''if(smooth_points):
		l=list(M.neibours)
		random.shuffle(l)
		for u in l:
			p=M.points[u]
			for v in M.neibours[u]:
				p+=M.points[v]
			p=p/(len(M.neibours[u])+1)
			M.points[u]=p'''
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

if(__name__=='__main__'):
	from glob import glob
	from os import path
	import random
	pth=path.dirname(__file__)
	ims=list(glob(path.join(pth,'*.jpg')))
	ims+=list(glob(path.join(pth,'*.png')))
	if(ims):
		im=Image.open(random.choice(ims))
	else:
		im=None
	for method in ['kmeans']:
		
		import time
		tm=time.time()
		loops=img2loops(im,n_colors=24,ss=1.9e5,point_cut_method=method)
		tm=time.time()-tm
		
		ww=1600
		hh=900
		w,h=im.size
		s=ldl2svg(loops,[],[],scale=min(ww/w,hh/h))
		from os import path
		with open(path.join(path.dirname(__file__),'sample_loops=%d_method=%s.svg'%(len(loops),method)),"w") as f:
			f.write(s)
		performance=len(loops)/tm
		print("===[method=%s,\ttime=%d seconds,\tperformance=%d loop/sec]==="%(method,tm,performance))