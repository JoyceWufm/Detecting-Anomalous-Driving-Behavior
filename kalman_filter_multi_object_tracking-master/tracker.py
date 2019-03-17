'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from kalman_filter import KalmanFilter
from common import dprint
from scipy.optimize import linear_sum_assignment


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter()  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
#        print ('how prediction from', prediction)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        self.framenum = []
        self.originalfnum = 0
        self.detectedx = []
        self.detectedy = []


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections,fnum):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """
#        print ('here is detection', detections)
        
        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.originalfnum = fnum
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
#                    print ('predition',self.tracks[i].prediction, 'detection',detections[j])
#                    print ('difference=', diff)
                    distance = np.sqrt(diff[0][0]*diff[0][0] +
                                       diff[1][0]*diff[1][0])
#                    print ('distance=', distance)
#                    if distance > 100:
#                        cost[i][j] = 1000
#                    else: 
                    cost[i][j] = distance
                except:
                    pass

        # Let's average the squared ERROR
#        print ('cost before', cost)
#        cost = (0.5) * cost
#        print ('cost after', cost)
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        for wi in range(N):
            for wj in range(M):
                if cost[wi][wj]> 120: #sensetive
                    cost[wi][wj]= 120
#        
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
#                print ('cost=',cost[i][assignment[i]],'assignment=',assignment[i])
                print ('predition',self.tracks[i].prediction, 'detection',detections[assignment[i]])

                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
#            oframe = self.tracks[i].originalfnum
#            skframe= self.tracks[i].skipped_frames
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip) or (fnum-self.tracks[i].originalfnum)>8:  # CHECK IT!!!
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    dprint("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1            
                self.tracks.append(track)
                tracklength = len(self.tracks)
                self.tracks[tracklength-1].originalfnum = fnum

        # Update KalmanFilter state, lastResults and tracks trace
        print ('assignment list', assignment)
        for i in range(len(assignment)):
            if len(self.tracks[i].trace) < 2:
                iit = len(self.tracks[i].trace)
                bbt = detections[assignment[i]] 
            else:
                if assignment[i] == -1:
                    iit = 0
                    bbt = self.tracks[i].prediction
                else:
                    iit = len(self.tracks[i].trace)
                    bbt = self.tracks[i].prediction

            self.tracks[i].KF.predict(iit, bbt)

            if(assignment[i] != -1):
                
                self.tracks[i].skipped_frames = 0
                predictionp = self.tracks[i].prediction
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            detections[assignment[i]], 1, 0)

            else:
                predictionp = self.tracks[i].prediction
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0, 1)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]
            
#            diffp = self.tracks[i].prediction - predictionp
#            distancep = np.sqrt(diffp[0][0]*diffp[0][0] + diffp[1][0]*diffp[1][0])            
#            
#            if distancep > 50:
#                print ('pb',self.tracks[i].prediction)
#                self.tracks[i].prediction = predictionp - [0,6]
#                print('id',self.tracks[i].track_id)
#                print('p',self.tracks[i].prediction,'pp',predictionp)


            self.tracks[i].framenum.append(fnum)
            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
            self.tracks[i].originalfnum = fnum
#            print ('new x y', int(detections[assignment[i]][0]),int(detections[assignment[i]][1]))
            self.tracks[i].detectedx.append(int(detections[assignment[i]][0]))
            self.tracks[i].detectedy.append(int(detections[assignment[i]][1]))
