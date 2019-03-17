# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##mask RCNN

import os
import sys
import numpy as np
import pandas as pd
import cv2 as cv
import copy
#from detectors import Detectors
from tracker import Tracker

# Root directory of the project
ROOT_DIR = os.path.abspath("/Users/Fumi/Desktop/Research Project/MRCNN&kalman")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


'''
    File name         : object_tracking.py
    File Description  : Multi Object Tracker Using Kalman Filter
                        and Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Modified by       : Fumi Wu
    Date created      : 07/14/2017
    Date last modified: 08/18/2018
    Python Version    : 3.6
'''

# Import python libraries

"""Main function for multi object tracking
Usage:
    $ python2.7 objectTracking.py
Pre-requisite:
    - Python2.7
    - Numpy
    - SciPy
    - Opencv 3.0 for Python
Args:
    None
Return:
    None
"""

# Create opencv video capture object
cap = cv.VideoCapture('data/Finalright.mp4')

# Create Object Tracker
tracker = Tracker(100, 21, 400, 0)

# Variables initialization
skip_frame_count = 0
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127)]
detectedtrackclr = (255, 255, 255)
pause = False

framea = -1
frameaa = 0
carid = []

# Infinite loop to process video frames
while(True):
    
    ret, frame = cap.read()

    if ret is True:
         # Make copy of original frame
        orig_frame = copy.copy(frame)  
        framea = framea + 1

        if (framea % 5 == 0):
            frameaa = framea / 5
            # Capture frame-by-frame

            # Skip initial frames that display logo
        if (skip_frame_count < 1):
            skip_frame_count += 1
            continue
        else:
            




            df = pd.read_csv(ROOT_DIR + '/Mask_RCNN-master/images/Finalright.csv', sep=',', names=['centroidy', 'centroidx','classids','file_names'])
            iii = 0
            centers = []
            for iii in range (0,len(df)):
                if df.iat[iii,3] == (frameaa+1):
                    if df.iat[iii,2] in [3,6,8]:

                        bb = np.array([[df.iat[iii,0]],[df.iat[iii,1]]])
                        centers.append(np.round(bb))
                    iii = iii + 1

        # If centroids are detected then track them
        if (len(centers) > 0):
            
            print ('frame number=',framea)
            # Track object using Kalman Filter
            centers.sort(key=lambda x:x[1])
            
            tracker.Update(centers,frameaa+1)
            
            cv.putText(frame, str(frameaa+1),(20,20),cv.FONT_HERSHEY_DUPLEX,0.8,(255,255,255))

            for icc in range (0,len(centers)):
                cv.circle(frame, (centers[icc][0], centers[icc][1]), 5, (0, 255, 0), 2)


            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                print ('trackid=',tracker.tracks[i].track_id)
                trackidcheck = tracker.tracks[i].track_id
                if (len(tracker.tracks[i].trace) > 10):
    #------------------------------------------------------------------------------------------------                
                    tracks_dataframe = pd.DataFrame(columns = ['trackID','x','y','tcolor','framenum','detectedx','detectedy'])
     
    #------------------------------------------------------------------------------------------------                   
                    for j in range(6,len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        xx1 = tracker.tracks[i].detectedx[j]
                        yy1 = tracker.tracks[i].detectedy[j]
                        xx2 = tracker.tracks[i].detectedx[j+1]
                        yy2 = tracker.tracks[i].detectedy[j+1]                            
                        clr = tracker.tracks[i].track_id % 9
                        framenummm = tracker.tracks[i].framenum[j]
                        if (len(tracker.tracks[i].trace)-1) - j < 10:
                            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),track_colors[clr], 2)
                            cv.line(frame, (int(xx1), int(yy1)), (int(xx2), int(yy2)),detectedtrackclr, 1)
    #                     tx.append(tracker.tracks[i].trace[j+1][0][0])
    #                         print (clr)
    #                         print ('x2=',tracker.tracks[i].trace[j+1][0][0])
        #                     print ('tx=',tx, 'length=',len(tx))
        #                     ty.append(tracker.tracks[i].trace[j+1][1][0])
    #                         print ('y2=',tracker.tracks[i].trace[j+1][1][0])
    #                     print ('ty=',ty,'length=',len(ty))
    #                     print ('(len(tracker.tracks[i].trace)-1)=',(len(tracker.tracks[i].trace)-1))
    #-------------------------------------------------------------------------------------------------                    
                        #export tracks
    #                     tid=np.full((len(tracker.tracks[i].trace)-1),i)
                        #with color ，记得改前面dataframe定义
                        tcolors=track_colors[clr]
                        tracks_part = pd.DataFrame([{'trackID':tracker.tracks[i].track_id,'x':x2, 'y':y2,'tcolor':clr,'framenum':framenummm,'detectedx':xx2, 'detectedy':yy2}])
    #                    print ('track=',tracks_part)
                        tracks_dataframe = pd.merge (tracks_dataframe, tracks_part, on=['trackID','x', 'y','tcolor','framenum','detectedx','detectedy'], how='outer')
                            #without color
        #                     tracks_part = pd.DataFrame([{'trackID':tracker.tracks[i].track_id,'x':x2, 'y':y2}])
        #                     tracks_dataframe = pd.merge (tracks_dataframe, tracks_part, on=['trackID','x', 'y'], how='outer')
#                    if len(tracker.tracks[i].trace)-1 > 7:
#                    cv.putText(frame, str(tracker.tracks[i].track_id),(int(tracker.tracks[i].detectedx[j+1]),int(tracker.tracks[i].detectedy[j+1])),cv.FONT_HERSHEY_DUPLEX,0.8,track_colors[tracker.tracks[i].track_id % 9])
                    cv.putText(frame, str(tracker.tracks[i].track_id),(int(x2),int(y2)),cv.FONT_HERSHEY_DUPLEX,0.8,track_colors[tracker.tracks[i].track_id % 9])

                    if len(tracks_dataframe) > 15:
                        tracks_dataframe.to_csv(ROOT_DIR+'/Tracks/'+ str(tracker.tracks[i].track_id)+'.csv',index=False,sep=',')   
    #-------------------------------------------------------------------------------------------------                     

        # Display the resulting tracking frame
        framer = cv.resize(frame,(400,600))
        cv.imshow('Tracking', frame)
        
        cv.imwrite(ROOT_DIR+'/Frames/'+str(framea)+'.jpg',frame)

        # Display the original frame
#         cv.imshow('Original', orig_frame)

        # Slower the FPS
        cv.waitKey(50)

        # Check for key strokes
        k = cv.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

        

# final_dataframe.to_csv(ROOT_DIR +'/images/test.csv',index=False,sep=',')                    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


# if __name__ == "__main__":
#     # execute main
#     main()




