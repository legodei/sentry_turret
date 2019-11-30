# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:29:30 2018

@author: Haojie
"""

import serial
import cv2
from imutils.video import VideoStream
import imutils
import time
import numpy as np
import csv
from sklearn.svm import SVR
import pickle
import os.path

# =============================================================================
# useful functions
# =============================================================================

def initialize_serial():
	port = "COM3"
	baud = 1000000
	ser = serial.Serial(port,baud,timeout=1)
	return ser

def send_command(ser,command,command_value):
	cmd = "<" + command + str(command_value).zfill(4)+">"
	ser.write(cmd.encode())

def confirm_command(ser):
	while ser.in_waiting:
		s = ser.readline()
		print(s.decode("latin-1"))

def gun_laying(ser,pos):
	(x,y) = pos
	traverse = x
	elevate = y
	send_command(ser,'t',traverse)
	send_command(ser,'e',elevate)

def terminate_serial(ser):
	ser.close()

def initialize_stream(source):
	vs = VideoStream(src = source, usePiCamera = False).start()
	time.sleep(1.0)
	return vs

def terminate_stream(vs):
	vs.stop()
	vs.stream.release()
	cv2.destroyAllWindows()

def dist(coord1,coord2):
	return (coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2

class target():

	def __init__(self, pos, rect, size, targid):

		self.cur_pos = np.array(pos)
		self.prev_pos = np.array([0,0])
		self.avg_pos = np.array(pos)
		self.hp = 100
		self.rect = np.asarray(rect)
		self.match = True
		self.size = size
		self.vel = np.array([0,0])
		self.velfilt = np.array([0,0])
		self.id = targid

	def draw(self,frame):
		(rx,ry,rw,rh) = self.rect
		(cx,cy) = self.avg_pos
		cx = int(cx)
		cy = int(cy)
		cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
		cv2.circle(frame,(cx,cy),6,(0,255,0))
		cv2.circle(frame,(cx,cy),7,(0,0,255))
		cv2.arrowedLine(frame,(cx,cy),
				  (cx + int(0.3*self.velfilt[0]),cy + int(0.3*self.velfilt[1])),
				  (0,255,255),3)
		cv2.putText(frame, str(self.id),(cx-5,cy-12),cv2.FONT_HERSHEY_PLAIN,
		 0.9, (0,255,0),1)
		#cv2.putText(frame, str(self.id)+" HP:"+str(self.hp) ,(cx-10,cy-10),cv2.FONT_HERSHEY_PLAIN,
		# 0.9, (0,255,0),1)

	def match_update(self,pos,rect,size):
		self.cur_pos = np.array(pos)
		self.rect = rect
		self.size = size

	def update(self):
		if self.match == False:
			self.hp -= 1
		else:
			self.hp = 100
		self.match = False
		self.vel = 50*(np.array(self.cur_pos)-np.array(self.prev_pos))
		self.velfilt = np.clip(self.velfilt*0.85+self.vel*0.1,-300,300)
		self.avg_pos = self.avg_pos*0.8+self.cur_pos*0.2
		self.prev_pos = self.cur_pos

class fps_monitor():

	def __init__(self):
		self.t1 = time.time()
		self.t0 = 0
		self.avgfps = 0
		self.elapsed = 0.001
		self.fps = 0
		self.avgfps = 60

	def update(self):
		self.t0 = self.t1
		self.t1 = time.time()
		self.elapsed = self.t1 - self.t0
		if self.elapsed == 0:
			self.elapsed = 0.001
		self.fps = 1/self.elapsed
		self.avgfps = self.avgfps*0.98+self.fps*0.02
		return self.avgfps




# =============================================================================
# main loop
# =============================================================================
# initialize the first frame in the video stream

if __name__ == "__main__":
	data = []
	elev = []
	trav = []
	fx = []
	fy = []

	seract = False
	blurval = 25
	threshval = 50
	kersize = 85
	keriter = 3
	vellimx = 60
	vellimy = 10
	x = 1400
	y = 1400
	vx0 = 0
	vy0 = 0
	OPEN = 2020
	CLOSE = 1700
	SHOOT = False
	frame_size = 500
	pulse = True
	rate = 0
	group = 0
	power_max = 1500
	power = 1400
	
	if os.path.exists('elevp.pickle') and os.path.exists('travp.pickle'):
		print("loading SVM pointing data from file")
		with open('elevp.pickle', 'rb') as e_p:
			elevp = pickle.load(e_p)
		with open('travp.pickle', 'rb') as t_p:
			travp = pickle.load(t_p)
	else:
		print("no file found, recalculating SVM from raw")
		with open('calibhoriz.csv','r') as f:
			reader = csv.reader(f)
			for row in reader:
				data.append(row)
		
		for row in data:
			elev.append(float(row[0]))
			trav.append(float(row[1]))
			fx.append(float(row[2]))
			fy.append(float(row[3]))
		
		fx = np.array(fx)
		fy = np.array(fy)
		#cx = cx.reshape(-1,1)
		#cy = cy.reshape(-1,1)
		
		coord = np.array([fx,fy])
		coord = np.swapaxes(coord,0,1)
		
		#elevp = LinearRegression()
		#travp = LinearRegression()
		
		elevp = SVR(C = 1, gamma = 1e-5)
		travp = SVR(C = 10, gamma = 1e-5)
		
		elevp.fit(coord, elev)
		travp.fit(coord, trav)
		with open('elevp.pickle', 'wb') as e_p:
			pickle.dump(elevp, e_p)
		with open('travp.pickle', 'wb') as t_p:
			pickle.dump(travp, t_p)
	
	firstFrame = None
	if seract:
		ser = initialize_serial()
	vs = initialize_stream(0)
	fps = fps_monitor()
	targ_list = []
	targ_ids = []
	delay = time.time()
	targac = False

	mask = np.full((375, 500), 0, dtype=np.uint8)
	cv2.rectangle(mask,(20,25),(480,300),(255,255,255),-1)
	mask2 = np.full((375, 500), 0, dtype=np.uint8)
	cv2.rectangle(mask2,(25,73),(475,256),(255,255,255),-1)
	alpha = 0.6
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kersize, kersize))
	while True:
		process_fps = fps.update()
		frame = vs.read()
		frame = imutils.resize(frame, width=frame_size)
		frame_olay = cv2.bitwise_and(frame,frame,mask=mask2)
		cv2.addWeighted(frame_olay, alpha, frame, 1 - alpha,0, frame)
		
		gray = cv2.bitwise_and(frame, frame, mask=mask)
		gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (blurval, blurval), 0)

		# if the first frame is None, initialize it
		if firstFrame is None:
			firstFrame = gray

		# compute the absolute difference between the current frame and
		# first frame
		frameDelta = cv2.absdiff(firstFrame, gray)
		thresh = cv2.threshold(frameDelta, threshval, 255, cv2.THRESH_BINARY)[1]

		# dilate the thresholded image to fill in holes, then find contours
		# on thresholded image

		thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)


		(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_NONE)

		# filter contour by size, if it is too small, then ignore it
		cnts[:] = [c for c in cnts if cv2.contourArea(c) > 500]

		# loop over the contours
		for c in cnts:
			(rx, ry, rw, rh) = cv2.boundingRect(c)
			M = cv2.moments(c) #find centroid for matching to previous frame
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			matched = False
			mindist = 10000
			best_match = None
			for targ in targ_list:
				if targ.match == False: #if it's not matched, then we attempt to match
					cur_dist = dist(targ.cur_pos,[cx,cy])
					if cur_dist < mindist:
						matched = True
						mindist = cur_dist
						best_match = targ #closest position is best match
			if best_match:
				best_match.match = True
				best_match.match_update([cx,cy],[rx,ry,rw,rh],cv2.contourArea(c))
			else:
				targid = 0
				no_id = True
				while no_id:
					if targid in targ_ids:
						targid += 1
					else:
						no_id = False
				targ_list.append(
						target([cx,cy],[rx,ry,rw,rh],cv2.contourArea(c),targid))
				targ_ids.append(targid)
		max_targ_size = 0
		primary_target = None
		for targ in targ_list:
			if targ.hp <= 5:  #target persistence
				targ_ids.remove(targ.id)
				targ_list.remove(targ)
			else:
				targ.update()

				if targ.hp>= 97:
					targ.draw(frame)
					if targ.size>max_targ_size:
						max_targ_size = targ.size
						primary_target = targ
		if primary_target: #if we have a target
			power = np.clip(power+3,1400,power_max)
			if seract and SHOOT:
				send_command(ser,'f',power)
				send_command(ser,'p', 1)
				if targac == False:
					delay = time.time()
					targac = True

			aim = primary_target.avg_pos + 0.3*primary_target.velfilt+[0,-30] + 0.001*(abs(primary_target.velfilt)*primary_target.velfilt)
			cv2.circle(frame,(int(aim[0]),int(aim[1])),5,(0,0,0),-1)
			aim = aim*500./(frame_size)
			px = travp.predict([list(aim)])
			py = elevp.predict([list(aim)])
			print(px,py,list(aim))
			vx = px-x
			vy = py-y
			vx = np.clip(vx-vx0,-vellimx,vellimx)  +vx0
			vy = np.clip(vy-vy0,-vellimy,vellimy)  +vy0


			x = vx + x
			y = vy + y

			if seract:
				gun_laying(ser,(int(x),int(y-100)))
				confirm_command(ser)


		else:
			power = 1400
			if seract:
				send_command(ser,'p', 0)
				if time.time()-delay > 0.3:
					send_command(ser,'f', 1400)
				targac = False
				confirm_command(ser)


		# show the frame and record if the user presses a key
		cv2.rectangle
		cv2.rectangle(frame,(25,73),(476,256),(0,0,0))
		
		frame = imutils.resize(frame,width = 600)
		thresh = np.vstack((thresh,frameDelta))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
		thresh = imutils.resize(thresh,width = 300)
		frame = np.concatenate((frame,thresh),axis=1)

		fontsize = 0.4
		frame = imutils.resize(frame,width = 900)
		cv2.putText(frame, "SENTRY",(50,35),cv2.FONT_HERSHEY_DUPLEX,
			 1, (102,200,0),1)
		cv2.putText(frame, "TURRET",(50,65),cv2.FONT_HERSHEY_DUPLEX,
			 1, (102,200,0),1)
		cv2.putText(frame, "FPS: "+str(round(process_fps)),(450,65),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.putText(frame, "BLURVAL (WS): "+str(blurval),(320,45),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.putText(frame, "THRESH (ED): "+str(threshval),(320,25),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.putText(frame, "KERSIZE (RF): "+str(kersize),(450,25),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.putText(frame, "KERITER (TG): "+str(keriter),(450,45),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.putText(frame, "POWER (YH): "+str(int((power-1400)/5))+"%",(190,25),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.putText(frame, "SERVO (A): "+str(seract),(190,45),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.putText(frame, "SAFE"*(not SHOOT)+"ARMED"*(SHOOT)+" (M)",(190,65),cv2.FONT_HERSHEY_SIMPLEX,
			 fontsize, (102,255,0),1)
		cv2.imshow("Video Feed",frame)
		key = cv2.waitKey(1) & 0xFF


		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		elif key == ord("w"):
			blurval+=2
		elif key == ord("s"):
			blurval = np.clip(blurval-2,1,100)
		elif key == ord("e"):
			threshval+=1
		elif key == ord("d"):
			threshval=np.clip(threshval-1,1,255)
		elif key == ord("r"):
			kersize+=2
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kersize, kersize))
		elif key == ord("f"):
			kersize=np.clip(kersize-2,1,200)
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kersize, kersize))
		elif key == ord("t"):
			keriter+=1
		elif key == ord("g"):
			keriter=np.clip(keriter-1,1,10)
		elif key == ord("p"):
			firstFrame = gray
		elif key == ord("m"):
			SHOOT = not SHOOT
		elif key == ord("a"):
			seract = not seract
		elif key == ord("y"):
			power_max+=100
		elif key == ord("h"):
			power_max-=100
	terminate_stream(vs)
	if seract:
		terminate_serial(ser)

