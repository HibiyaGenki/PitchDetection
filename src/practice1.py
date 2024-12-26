import cv2
import numpy as np2
import os
import json

def annottaiton(cap_url, f):
    frameWidth = 800
    frameHeight = 480
    cap = cv2.VideoCapture(cap_url)
    # json_open = open(f, 'r')
    #json_load = json.load(json_open)
    print json_load
    lines = f.readlines()
    pitch_time=[]

    for i in range(0, 18, 2):  
        pair = (int(lines[i]), int(lines[i + 1]))
        pitch_time.append(pair)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_name = os.path.splitext(os.path.basename(cap_url))[0]

    for frame_num in range(total_frames):
        ret, img = cap.read()
        img = cv2.resize(img, (frameWidth,frameHeight))
        for i, (start_frame, end_frame) in enumerate(pitch_time):
            if frame_num == start_frame:
                out = cv2.VideoWriter(
                    f'/Users/hibiyagenki/Desktop/{base_name}_pitch_scene_{i + 1}.mp4',
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frameWidth, frameHeight)
                    )
                
            if start_frame <= frame_num <= end_frame:
                img = cv2.rectangle(img, 
                    (300,150),
                    (420,300),
                    (0,0,255),    
                    6
                    )
                out.write(img)
            if not ret:
                break            

        cv2.imshow('HIBIYAVideo', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
annottaiton('/Users/hibiyagenki/Downloads/tachikawa_K-W_1T.mp4',open('/Users/hibiyagenki/Desktop/annottation_time.txt', 'r'))
