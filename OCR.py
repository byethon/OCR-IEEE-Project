import os.path
import cv2
import numpy as np
from numpy.core.fromnumeric import reshape
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")

BaseDir = os.path.dirname(os.path.abspath(__file__))
inimg = os.path.join(BaseDir, 'inimg.png')

imgsrc = cv2.imread(inimg)
src_h, src_w, scr_c= imgsrc.shape
if (src_w>1600):
    expect_h = src_h/src_w*1600
    expect_h = int(expect_h)
    img= cv2.resize(imgsrc, (1600,expect_h), interpolation=cv2.INTER_AREA)
elif (src_w<1600):
    expect_h = src_h/src_w*1600
    expect_h = int(expect_h)
    img= cv2.resize(imgsrc, (1600,expect_h), interpolation=cv2.INTER_NEAREST)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

####CONVOLUTION 1####
## Pre-Processing for Paragraph detection ##
para_img = cv2.resize(gray, (400,int(expect_h/4)), interpolation=cv2.INTER_AREA)
para_blur = cv2.blur(para_img, (45,1))
para_thresh_val, para_thresh = cv2.threshold(para_blur, 0, 255, cv2.THRESH_OTSU)
para_blur_x2 = cv2.blur(para_thresh, (1,30))
## END ##

## Paragraph Detection ##
def contourprocess(inputimg):
    thresh_val, thresh = cv2.threshold(inputimg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pre_sort=[] #list to contain all the contour boxes before proper arrangement

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        pre_sort.append((x,y,w,h))

    sort_arr=np.array(pre_sort, dtype=[('loc_x', np.int32), ('loc_y', np.int32), ('char_w', np.int32), ('char_h', np.int32)])
    sort_arr=np.sort(sort_arr, order='loc_x')

    hsv=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

    step=0
    crop_arr=[]
    while step<expect_h:
        f=0
        for val in sort_arr:
            x = sort_arr[f]['loc_x']
            y = sort_arr[f]['loc_y']
            w = sort_arr[f]['char_w']
            h = sort_arr[f]['char_h']
            if ((y+h)>step and (y+h)<=(step+8)):  # scans screen for contours in 8px slices
                if (x,y,w,h) in crop_arr:
                    print("Not optimized")
                else:
                    test=cv2.rectangle(hsv, (x,y), (x+w , y+h), (255,0,0), 2)
                    cv2.imshow('test',test)
                    cv2.waitKey(1000)
                    crop_arr.append((x,y,w,h))
            f=f+1
        step=step+8
    return crop_arr

para_arr = contourprocess(para_blur_x2)
print(para_arr)
