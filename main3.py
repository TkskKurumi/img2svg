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
			if(dx<loop_stroke_width*2 or dy<loop_stroke_width*2):
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
		#print('ln109',len(rets),k)
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
				if(nn in sum):
					sum[nn]=i+sum[nn]
				else:
					sum[nn]=i
			else:
				cnt[nn]=cnt.get(nn,0)+wei[idx]
				if(nn in sum):
					sum[nn]=i*wei[idx]+sum[nn]
				else:
					sum[nn]=i*wei[idx]
		rets=[]
		for i in cnt:
			rets.append(sum[i]/cnt[i])
		if(len(rets)<k):
			rets.extend(random.sample(points,k-len(rets)))
	return rets
def img2loops(img,ss=1e4,n_points=None,n_colors=64,print_progress=True,ensure_corner=True,debug=False):
	import time
	last_prog=time.time()
	i_prog=0
	first_prog=dict()
	if(n_points is None):
		n_points=int(min((ss**0.5)*30,ss/3))
	def prog_cb(title):
		if(print_progress):
			def inner(prog,title=title):
				return progbar(title,prog)
			return inner
		else:
			return None
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
			print(title,"["+enmiao+"] %.1f secs remain"%(remain),end='\r')
			last_prog=t
	w,h=img.size
	rate=(ss/w/h)**0.5
	sample_w,sample_h=int(w*rate),int(h*rate)
	
	simg=img.resize((sample_w,sample_h),Image.LANCZOS)
	
	colors=[]
	
	sample_color=int((n_colors*ss)**0.5)
	xys=list(wh_iter(sample_w,sample_h))
	for idx,xy in enumerate(random.sample(xys,sample_color)):
		if(print_progress):
			progbar('sample colors',idx/sample_color)
		colors.append(np.array(simg.getpixel(xy),np.float32))
	
	colors=kmeans_with_kdt(n_colors,colors,progress_cb=prog_cb('merge color'))
	import kdt
	K=kdt.kdt()
	K.build([kdt.point(c) for c in colors])
	for u in range(K.size):
		if(not K.node_points[u]):continue
		for idx,i in enumerate(K.node_points[u]):
			
			K.node_points[u][idx].arr=tuple([int(j) for j in i.arr])
	for xy in wh_iter(sample_w,sample_h):
		x,y=xy
		if(print_progress):
			progbar("simplify image color",(y*sample_w+x)/sample_w/sample_h)
		c=simg.getpixel(xy)
		c=K.ann1(kdt.point(c)).arr
		#c=npa2tuple_color(c)
		simg.putpixel(xy,c)
	if(debug):
		#simg.show()
		pass
	pixel_group=DJS()
	pixel_wei=dict()
	for xy in wh_iter(sample_w-1,sample_h-1):
		x,y=xy
		if(print_progress):
			progbar("join color block",(y*(sample_w-1)+x)/(sample_w-1)/(sample_h-1))
		c=simg.getpixel(xy)
		c1=simg.getpixel((x+1,y))
		c2=simg.getpixel((x,y+1))
		if(c==c1):
			pixel_group.join(xy,(x+1,y))
			pixel_group.find(xy)
			pixel_group.find((x+1,y))
		if(c==c2):
			pixel_group.join(xy,(x,y+1))
			pixel_group.find(xy)
			pixel_group.find((x,y+1))
		pixel_wei[xy]=colordis(c,c1)+colordis(c,c2)+1
	group_edges=dict()
	for xy in wh_iter(sample_w-1,sample_h-1):
		x,y=xy
		g=pixel_group.find(xy)
		for dx,dy in [(0,1),(1,0)]:
			x1=x+dx
			y1=y+dy
			xy1=(x1,y1)
			g1=pixel_group.find(xy1)
			if(g!=g1):
				if(g not in group_edges):
					group_edges[g]=set()
				group_edges[g].add(xy)
				if(g1 not in group_edges):
					group_edges[g1]=set()
				group_edges[g1].add(xy1)
			
	now_points=sum([len(group_edges[i]) for i in group_edges])
	_prog=0
	rate=n_points/now_points
	if(rate<1):
		print("merge %d points into %d points"%(now_points,n_points))
		if(debug):
			pass
	points=list()
	def upscale(p):
		return point(p.x*w/sample_w,p.y*h/sample_h)
	for i,edges in group_edges.items():
		_points=[point(x,y) for x,y in edges]
		
		le=len(_points)
		k=int(rate*le)
		if(k<le and k):
			#print("ln239")
			wei=[pixel_wei[xy] for xy in edges]
			_points=kmeans_with_kdt(k,_points,wei=wei)
			if(len(_points)>k):
				print("wtf")
				exit()
		else:
			#print("ln242")
			if(random.random()>rate):
				_points=[]
		_prog+=le
		if(print_progress):
			progbar("merge points",_prog/now_points)
		points.extend(_points)
	
	if(ensure_corner==True):
		for x in [0,sample_w-1]:
			for y in [0,sample_h-1]:
				points.append(point(x,y))
	enmiao=1e2
	points=list(set([point(int(p.x*enmiao),int(p.y*enmiao)) for p in points]))
	print('%d points'%len(points))
	points=[point(p.x/enmiao,p.y/enmiao) for p in points]
	M=mesh.delaunay(points,prog_cb=prog_cb('delaunay'))
	mx=max([i.x for i in points])
	
	loops=[]
	tri_points=M.get_tri_integral_point()
	
	def get(x,y):
		return im.getpixel((int(x),int(y)))[:3]
	
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
		
		
		loops.append((1,[upscale(A).xy,upscale(B).xy,upscale(C).xy],color))
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
	import time
	tm=time.time()
	loops=img2loops(im,n_colors=32,ss=3e5,debug=True)
	tm=time.time()-tm
	
	ww=1600
	hh=900
	w,h=im.size
	s=ldl2svg(loops,[],[],scale=min(ww/w,hh/h))
	from os import path
	with open(path.join(path.dirname(__file__),'sample_method=main3_loops=%d.svg'%len(loops)),"w") as f:
		f.write(s)
	performance=len(loops)/tm
	print("===[time=%d seconds,\tperformance=%d loop/sec]==="%(tm,performance))