import cv2 
import os
import numpy as np


def video2image(video_path, image_path, fps):
    vc=cv2.VideoCapture(video_path)  
    c=1  
    a=0
    if vc.isOpened():  
        rval,frame=vc.read()  
    else:  
        rval=False  
    while rval:  
        rval,frame=vc.read() 
        print (c)
        if c % fps == 0:
            a=a+1
            cv2.imwrite(image_path +'\\'+str(a)+'.jpg',frame) 
            print (c)

        c=c+1  
        cv2.waitKey(1)  
    vc.release() 

def carcrop(dirs, imagename, length, classids, x1, x2, y1, y2):

    
            
        fn = os.path.splitext(os.path.basename(imagename))[0]
        print (fn)
        
        path = dirs +'/images/' + str(fn) #或者直接image path
#        path = dirs +'/images/8' #或者直接image path
        os.makedirs(path)
        cropimage = cv2.imread(imagename)
        
        cx = np.empty(length)
        cy = np.empty(length)
        filenum = np.empty(length)
        
        i = 0
        ii = 0
        for i in range(0,length):
            cx[i] = (x1 + x2)/2
            cy[i] = (y1 + y2)/2
            if classids[i] in [3,6,8]:
                if y1[i] > 200:
                    cropImg = cropimage[y1[i]:y2[i],x1[i]:x2[i]] 
                    cv2.imwrite(path + "//" + str(ii) + ".jpg",cropImg)
                    filenum[i] = ii
                    ii = ii + 1
            else:
                filenum[i] = -1
            i = i+1
        
        return cx, cy, filenum
            
         


