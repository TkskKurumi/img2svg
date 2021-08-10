from algorithms import *
def colordis(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))**0.5
def img2svg(im,thresh=30):
    color_groups=disjointset()
    
    w,h=im.size
    for x in range(w-1):
        for y in range(h-1):
            c=im.getpixel((x,y))
            if(colordis(c,im.getpixel((x+1,y)))):
                color_groups.join((x,y),(x+1,y))
            if(colordis(c,im.getpixel((x,y+1)))):
                color_groups.join((x,y),(x,y+1))
            