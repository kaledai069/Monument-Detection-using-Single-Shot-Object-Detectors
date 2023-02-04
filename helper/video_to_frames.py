from datetime import timedelta
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

SAVING_FRAMES_PER_SECOND = 4
FRAME_COUNT = 0

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def convert_to_frames(video_file, output_dir, req_action):
    global FRAME_COUNT
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            break
        frame_duration = count / fps
        try:
            closest_duration = saving_frames_durations[0]
        except IndexError:
            break
        if frame_duration >= closest_duration:
            if req_action != 'nothing':
                frame = cv2.rotate(frame, req_action)
            cv2.imwrite(os.path.join(output_dir, f"{FRAME_COUNT}.jpg"), frame) 
            
            FRAME_COUNT += 1
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        count += 1


def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def main(video_file):
    filename, _ = os.path.splitext(video_file)
    filename += "-opencv"
    if not os.path.isdir(filename):
        os.mkdir(filename)    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            break
        frame_duration = count / fps
        try:
            closest_duration = saving_frames_durations[0]
        except IndexError:
            break
        if frame_duration >= closest_duration:
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame) 
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        count += 1
        if count == 20:
            break

if __name__ == "__main__":
    import sys
    dir_loc = sys.argv[1]
    all_video_files = [dir_loc + "\\" + video_name for video_name in os.listdir(dir_loc)]
    
    KDS_conversion_dict = {
        '1': 'nothing', 
        '2': cv2.ROTATE_180,
        '3': cv2.ROTATE_90_COUNTERCLOCKWISE,
        '4': 'nothing',
        '6': 'nothing',
        '7': cv2.ROTATE_90_COUNTERCLOCKWISE,
        '8': 'nothing',
        '9': cv2.ROTATE_180,
        '11': 'nothing',
        '12': cv2.ROTATE_180,
        '13': 'nothing'
    }

    BDS_conversion_dict = {
        '1': 'nothing',
        '2': 'nothing',
        '3': 'nothing',
        '4': 'nothing',
        '5': 'nothing',
        '6': cv2.ROTATE_90_COUNTERCLOCKWISE,
        '7': cv2.ROTATE_90_COUNTERCLOCKWISE,
        '8': cv2.ROTATE_90_COUNTERCLOCKWISE,
        '9': 'nothing',
        '10': 'nothing',
        '11': cv2.ROTATE_180,
        '12': 'nothing',
        '13': 'nothing',
        '14': 'nothing',
        '15': cv2.ROTATE_180,
        '16': 'nothing',
        '17': cv2.ROTATE_180,
        '18': cv2.ROTATE_180,
        '19': 'nothing',
        '20': 'nothing',
        '21': 'nothing',
        '22': 'nothing',
        '23': 'nothing',
        '24': 'nothing',
        '25': cv2.ROTATE_180,
        '26': cv2.ROTATE_180,
    }

    output_dir = dir_loc + "\\OUTPUT FRAMES"
    print(output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir) 
    for video_name in (all_video_files):
        file_dir = video_name.split('.')[0]
        file_name = file_dir.split('\\')[4]
        required_action = BDS_conversion_dict[file_name]
        convert_to_frames(video_name, output_dir, required_action)
        