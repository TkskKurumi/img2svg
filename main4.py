from algorithms import *
from myGeometry import polygon_area,point
import numpy as np
import random,shutil
from PIL import Image,ImageDraw
from delaunay_mesh import mesh
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
def smooth_points_linear_cutdown(points,step=2.4,start=0,end=None):
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
def smooth_points_momentum(points,*args,**kwargs):
	if(len(points)<5):
		return points
	_points=[point(x,y) for x,y in points]
	integ=point(0,0)
	now=_points[0]
	__points=[]
	last_delta=point(0,0)
	for i in _points:
		delta=i-now
		deriv=delta-last_delta
		last_delta=delta
		integ+=(delta*0.1+deriv*0.1)/2
		
		'''if(integ.distO()>5):
			#print('ln36')
			integ=integ*4/integ.distO()'''
		d=integ*delta
		
		if(integ.distO()<1e-6 or delta.distO()<1e-6):
			d=0.5
		else:
			d=d/integ.distO()/delta.distO()
			d=(1-d)/2
		a=1-d
		if(d<0.3):
			integ*=d+0.7
		
		now+=delta*(a+0.5)/4+integ*(d+1)/2+deriv*0.1
		__points.append(now.xy)
	return __points
def smooth_points(points,smooth=1.5):
	points=smooth_points_linear_cutdown(points,smooth)
	points=smooth_points_momentum(points)
	return points

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
def img2ldl(im,ss=1e5,n_colors=None,debug=False,print_progress=True,back_delaunay=None,force_group=16):
	import time
	last_prog=time.time()
	last_title=""
	i_prog=0
	first_prog=dict()
	def prog_cb(title):
		if(print_progress):
			def inner(prog,title=title):
				return progbar(title,prog)
			return inner
		else:
			return None
	def progbar(title,prog,width=None,print_finish=False):
		nonlocal i_prog,last_prog,last_title
		if(print_finish):
			t=time.time()
			print("%s finished in %.1f seconds  "%(last_title,t-first_prog[last_title]))
		i_prog+=1
		if(i_prog&0b111):
			return
		t=time.time()
		if((title!=last_title) and (last_title in first_prog)):
			print("%s finished in %.1f seconds  "%(last_title,t-first_prog[last_title]))
		last_title=title
		if(title not in first_prog):
			first_prog[title]=t
		if(t-last_prog>0.1):
			remain=(t-first_prog[title])/(prog+1e-10)*(1-prog)
			tmp1="["
			tmp2="] %.1f secs remain"%(remain)
			if(width is None):
				col=shutil.get_terminal_size().columns
				width=col-len(tmp1)-len(tmp2)-len(title)-1
			enmiao="#"*int(prog*width)
			enmiao+="."*max(width-len(enmiao),0)
			print(title,tmp1+enmiao+tmp2,end='\r')
			last_prog=t
	
	
	w,h=im.size
	if(ss!='dont_change'):
		rate=(ss/w/h)**0.5
		print("resize rate =",rate)
		sw,sh=int(w*rate),int(h*rate)
		sim=im.resize((sw,sh),Image.LANCZOS)
	else:
		global dont_change_ss
		sim=im.copy()
		sw,sh=sim.size
		
		ss=sw*sh
		dont_change_ss=ss
		perfj,perf=estimate_performance()
		print("estimated runtime",sw*sh/perf)
		rate=1
	if(n_colors is None):
		n_colors=int((ss**0.5)/30)
		print("n_colors =",n_colors)
	else:
		n_colors=int(n_colors)
	xys=list(wh_iter(sw,sh))
	xys1=list(wh_iter(sw-1,sh-1))
	xys2=list(wh_iter(sw-2,sh-2))
	calc_xy_prog=lambda x,y,w,h:(y*w+x)/w/h
	
	colors=[]
	n_sample_color=int((n_colors**0.3)*(ss**0.7))
	for idx,xy in enumerate(random.sample(xys,n_sample_color)):
		if(print_progress):progbar("sample colors",idx/n_sample_color)
		colors.append(np.array(sim.getpixel(xy)))
	colors=kmeans_with_kdt(n_colors,colors,progress_cb=prog_cb("merge colors"))
	
	import kdt
	K=kdt.kdt()
	K.build([kdt.point(c) for c in colors],stop_num=3)
	sim_arr=np.zeros(sim.size,np.uint32)
	#_sim_arr=np.asarray(sim).swapaxes(0,1)
	if(debug):id2c=dict()
	for xy in xys:
		if(print_progress):progbar("simplify image",calc_xy_prog(*xy,sw,sh))
		x,y=xy
		c=sim.getpixel(xy)
		#c=_sim_arr[x,y]
		nn=K.ann1(kdt.point(c))
		sim_arr[x,y]=nn.id
		if(debug):id2c[nn.id]=c
	
	pixel_group=DJS_sum()
	all_edges=set()
	group_color=dict()
	group_pixeln=dict()
	group_pixels=dict()
	for xy in xys1:
		if(print_progress):progbar("pixel grouping",calc_xy_prog(*xy,sw-1,sh-1))
		x,y=xy
		for dx,dy in [(0,1),(1,0)]:
			x1,y1=x+dx,y+dy
			if(sim_arr[x,y]==sim_arr[x1,y1]):
				pixel_group.join((x,y),(x1,y1))
				#pixel_group.find((x,y))
				#pixel_group.find((x1,y1))
	for xy in xys1:
		if(print_progress):progbar("detect edge",calc_xy_prog(*xy,sw-1,sh-1))
		x,y=xy
		
		for dx,dy in [(0,1),(1,0),(1,1),(-1,1)]:
			if(pixel_group.finds(xy)>=force_group):
				break
			x1,y1=x+dx,y+dy
			if(x1<0):
				continue
			if(pixel_group.finds((x1,y1))<force_group):
				pixel_group.join((x,y),(x1,y1))
			#d,x1,y1=closest
			#pixel_group.join((x,y),(x1,y1))
		for dx,dy in [(0,1),(1,0)]:
			x1,y1=x+dx,y+dy
			if(pixel_group.find(xy)!=pixel_group.find((x1,y1))):
				all_edges.add(xy)
				all_edges.add((x1,y1))
	for x in range(sw):
		all_edges.add((x,0))
		all_edges.add((x,sh-1))
	for y in range(sh):
		all_edges.add((0,y))
		all_edges.add((sw-1,y))
	
	npaf32=lambda x:np.array(x,np.float64)
	for xy in xys:
		if(print_progress):progbar("calc group avg color",calc_xy_prog(*xy,sw,sh))
		g=pixel_group.find(xy)
		c=sim.getpixel(xy)
		if(g not in group_color):
			group_color[g]=npaf32(c)
		else:
			group_color[g]+=npaf32(c)
		group_pixeln[g]=group_pixeln.get(g,0)+1
		if(g not in group_pixels):
			group_pixels[g]=list()
		group_pixels[g].append(xy)
	if(debug):
		for xy in xys:
			progbar("debug ln166",calc_xy_prog(*xy,sw,sh))
			x,y=xy
			sim.putpixel(xy,id2c[sim_arr[x,y]])
		sim.show()
	group_graph=dict()
	for idx,xy in enumerate(all_edges):
		if(print_progress):progbar("edge graphing",idx/len(all_edges))
		x,y=xy
		g=pixel_group.find(xy)
		if(g not in group_graph):
			group_graph[g]=graph()
		for dx,dy in dxy4:
			x1,y1=x+dx,y+dy
			if(x1<0 or x1>=sw or y1<0 or y1>=sh):
				continue
			if(g!=pixel_group.find((x1,y1))):
				continue
			group_graph[g].add_edge((x,y),(x1,y1))
	loops=[]
	def upscale(t):
		if(isinstance(t,tuple)):
			x,y=t
			return (x*w/sw,y*h/sh)
		elif(isinstance(t,point)):
			return point(upscale(t.xy))
	def downscale(t):
		if(isinstance(t,tuple)):
			x,y=t
			return int(x*sw/w),int(y*sh/h)
	if(print_progress):
		sm_edges=sum([len(g.edges) for i,g in group_graph.items()])
		no_edges=0
	lines=[]
	if(debug):
		im_pixtype=Image.new("RGB",sim.size)
		areas=0
		print("len group_graph",len(group_graph))
	dots=[]
	for i in group_graph:
		graphs=group_graph[i].seperate_by_connectivity()
		#largest=None
		_loops=list()
		for idx,G in enumerate(graphs):
			spt=G.span_tree()
			longest=None
			for edg in G.edges:
				if(print_progress):
					no_edges+=1
					progbar("generating loops",no_edges/sm_edges)
				if(edg in spt.edges):
					continue
				u,v=edg
				dist=spt.dist(u,v)
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
				if(len(loop)-0.05<=force_group**0.5):
					continue
				area=polygon_area([point(i) for i in loop])
				'''if(area>sw*sw*0.998):
					print("too large an area of loop, continued")
					continue'''
				c=group_color[i]/group_pixeln[i]
				_loops.append((area,[upscale(_) for _ in loop],npa2tuple_color(c)))
		_loops.sort(key=lambda x:-x[0])
		le=len(_loops)
		from math import ceil
		
		_loops=_loops[:ceil(le**0.5)]
		do_lines=True
		if(_loops):		#loop
			#area,loop=largest
			#areas+=area
			do_lines=False
			#loops.extend([__loops for __loops in _loops if len(__loops[1])>force_group**0.5])
			loops.extend(_loops)
			#loops.append((area,[upscale(_) for _ in loop],))
			if(debug):
				for xy in group_pixels[i]:
					im_pixtype.putpixel(xy,(255,0,0))
				for xy in loop:
					im_pixtype.putpixel(xy,(0,0,255))
		
		else:						   #lines
			_lines=[]
			_len=0
			c=group_color[i]/group_pixeln[i]
			c=npa2tuple_color(c)
			for idx,G in enumerate(graphs):
				pth=G.farthest_path()
				'''if(longest is None or len(pth)>longest):
					longest=pth'''
				
				if(len(pth)>1):
					_lines.append(([upscale(_) for _ in pth],c))
			if(_lines):
				lines.extend(_lines)
			else:
				#add dot
				xs,ys=0,0
				for x,y in group_pixels[i]:
					xs+=x
					ys+=y
				xs/=group_pixeln[i]
				ys/=group_pixeln[i]
				rad=0
				for x,y in group_pixels[i]:
					_rad=(x-xs)**2+(y-ys)**2+1
					rad=max(rad,_rad)
					#print(_rad)
				rad**=0.5
				dots.append((upscale((xs,ys)),c,rad/rate))
				pass
	if(debug):
		
		im_pixtype.show()
		dr=ImageDraw.Draw(im_pixtype)
		for loop in loops:
			area,points,c=loop
			points=smooth_points(points)
			dr.polygon(points,fill=c)
		for xy,c,siz in []:
			x,y=xy
			siz=siz
			az=x-siz,x+siz,y-siz,y+siz
			x1,x2,y1,y2=[int(i) for i in az]
			dr.ellipse((x1,y1,x2,y2),fill=c)
		for line in lines:
			points,c=line
			points=smooth_points(points)
			dr.line(points,fill=c,width=4)
		
		for a,l,c in loops:
			for xy in l:
				xy=downscale(xy)
				im_pixtype.putpixel(xy,(0,255,255))
		im_pixtype.show()
		print(areas,ss)
	delaunay_loops=[]
	if(back_delaunay is None):
		back_delaunay=int(len(loops)/23)
	max_area=max([a for a,b,c in loops])
	if(back_delaunay):
		delaunay_pts=[]
		#delaunay_pts.extend(xys)
		for a,l,c in loops:
			delaunay_pts.extend([downscale(i) for i in l])
		for x in range(sw):
			delaunay_pts.append((x,0))
			delaunay_pts.append((x,sh-1))
		for y in range(sh):
			delaunay_pts.append((0,y))
			delaunay_pts.append((sw-1,y))
		delaunay_pts=random.sample(delaunay_pts,back_delaunay-4)
		for x in [0,sw-1]:
			for y in [0,sh-1]:
				delaunay_pts.append((x,y))
		delaunay_pts=[point(x,y) for x,y in set(delaunay_pts)]
		
		
		
		M=mesh.delaunay(delaunay_pts,prog_cb=prog_cb('delaunay'))
		tri_points=M.get_tri_integral_point()
		
		def get(x,y):
			return im.getpixel((int(x),int(y)))[:3]
		for abc,_pts in tri_points.items():
			a,b,c=abc
			A,B,C=M.points[a],M.points[b],M.points[c]
			if(not _pts):
				CC=(A+B+C)/3
				color=get(CC.x*w/sw,CC.y*h/sh)
			else:
				
				color=np.zeros((3,),np.float32)
				
				for x,y in _pts:
					color+=np.array(get(x*w/sw,y*h/sh),np.float32)
				color=npa2tuple_color(color/len(_pts))
			area=polygon_area([A,B,C])
			area=(area**0.3)*(ss**0.7)
			loop=[upscale(A).xy,upscale(B).xy,upscale(C).xy]
			
			delaunay_loops.append((area,loop,color))
	
	if(print_progress):
		progbar('',0,print_finish=True)
	#print(delaunay_loops)
	print("delaunay loops %d"%len(delaunay_loops))
	
	loops.extend(delaunay_loops)
	#loops=delaunay_loops
	return sorted(loops,key=lambda x:-x[0]),dots,lines,rate
def ldl2svg(loops,dots,lines,smooth=4,blur_dots=1.2,scale=3,cutdown_dots=10000,line_alpha=0.3,loop_stroke=True,loop_stroke_width=1.2,loop_trim=False):
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
			if(dx<loop_stroke_width*3 or dy<loop_stroke_width*3):
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
		prt('" stroke="RGBA(%d,%d,%d,%.1f%%)" fill="none" stroke-width="%.1f" />'%(*c[:3],100*line_alpha,loop_stroke_width*scale/2))
	for line in lines:
		points,c=line
		points=smooth_points(points,smooth)
		prt('<path d="',end='')
		f="M"
		for x,y in points:
			prt(f,end='')
			prt("%.2f"%(x*scale),"%.2f"%(y*scale),end=' ')
			f="L"
		prt('" stroke="RGBA(%d,%d,%d,%.1f%%)" fill="none" stroke-width="%.1f" />'%(*c[:3],50*line_alpha,loop_stroke_width*scale))
	
	prt("</svg>",end='')
	return out
def estimate_performance():
	import platform
	from os import path
	import tempfile
	pth=path.join(tempfile.gettempdir(),"img2svg_performance.json")
	if(not path.exists(pth)):
		j={}
		perf={"AMD64":7670,'aarch64':6800}.get(platform.machine(),4000)
	else:
		try:
			import json
			f=open(pth,'r')
			j=json.load(f)
			f.close()
			perf=j['ss']/j['time']
		except Exception:
			print("Cannot load recorded performance")
			j={}
			perf={"AMD64":7670,'aarch64':6800}.get(platform.machine(),4000)
	return j,perf
if(__name__=='__main__'):
	import sys,simple_arg_parser
	args=" ".join(sys.argv[1:])
	args=simple_arg_parser.parse_args(args)
	from glob import glob
	from os import path
	import random
	pth=path.dirname(__file__)
	ims=list(glob(path.join(pth,'*.jpg')))
	ims+=list(glob(path.join(pth,'*.png')))
	if(ims):
		im=Image.open(random.choice(ims)).convert("RGB")
	import time
	tm=time.time()
	
	
	quality=args.get("q",None) or args.get("quality",None) or 'dont_change'
	perfj,perf=estimate_performance()
	if(quality=='dont_change'):
		ss=quality
	else:
		ss=float(quality)*perf
	
	n_colors=args.get("n_color",None)
	#n_colors=int(n_colors)
	print("ss=%s,n_colors=%s"%(ss,n_colors))
	loops,dots,lines,rate=img2ldl(im,n_colors=n_colors,ss=ss,debug=False)
	if(args.get("no_lines",False) or args.get("nl",False)):
		lines=[]
	if(args.get("no_dots",False) or args.get("nd",False)):
		dots=[]
	
	print(len(loops),'loops')
	print(len(dots),'dots')
	print(len(lines),'lines')
	tm=time.time()-tm
	
	ww=1600
	hh=900
	w,h=im.size
	scale=min(ww/w,hh/h)
	s=ldl2svg(loops,dots,lines,scale=scale,loop_stroke_width=3/rate)
	if(quality=='dont_change'):
		ss=dont_change_ss
	else:
		print('ln566',ss,quality)
	from os import path
	with open(path.join(path.dirname(__file__),'sample_loops=%d_method=main4.svg'%len(loops)),"w") as f:
		f.write(s)
	performance=ss/tm
	print("===[time=%d seconds,\tperformance=%d pixels/sec]==="%(tm,performance))
	perfj['time']=perfj.get('time',0)/2+tm
	perfj['ss']=perfj.get('ss',0)/2+ss
	import tempfile
	pth=path.join(tempfile.gettempdir(),"img2svg_performance.json")
	f=open(pth,'w')
	import json
	json.dump(perfj,f)
	f.close()
	
	'''im2=Image.new("RGB",(1600,900))
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
	im2.show()'''
		