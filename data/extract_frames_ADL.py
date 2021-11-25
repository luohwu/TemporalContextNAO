from opt import *
import tarfile
import  os
import shutil
import pandas as pd
from ast import literal_eval
from itertools import chain
import cv2

def make_dirs():
    output_dir='/media/luohwu/T7/dataset/ADL/rgb_frames/'
    video_id_list=sorted(ids_adl)
    for video_id in video_id_list:
        video_id=video_id
        target_dir=os.path.join(output_dir,video_id)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)



#given a video_id and its annotation file, output a generator containing only the names of needed frames.
def get_frames_id(video_id,basic_only=False):

    annos_file_path=os.path.join(args.data_path,annos_path,f'nao_{video_id}.csv')
    assert os.path.join(annos_file_path), f"file not exists: {annos_file_path}"

    # nao_P01P01_01.csv contain the frame and added previous frames for each train/test sample
    annos=pd.read_csv(annos_file_path,converters={"previous_frames":literal_eval})
    if annos.shape[0]==0:
        return None
    frames=annos['frame'].tolist()
    previousframes=annos['previous_frames'].tolist() # 2d list e.g. [[1,31,61], [ 17, 57, 77]]

    # flatten 2d list to 1d list
    previousframes=list(chain.from_iterable(previousframes))

    if basic_only==False:
        #concatenate frames and added previous frames into a single list
        all_frames=frames+previousframes
    else:
        # original dataset in the baseline method
        all_frames=frames

    return all_frames


def extract_frames_from_video(video_id):
    frames=get_frames_id(video_id)
    frames=sorted(frames)
    vidcap = cv2.VideoCapture(f'/media/luohwu/T7/ADL/{video_id}.mp4')
    # vidcap = cv2.VideoCapture('/home/luohwu/Videos/P01_02.mp4')
    if not vidcap.isOpened():
        print(f'failde opening video')
    success=True
    count = -1
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        count+=1
        # print(count)
        if count in frames:
            # print(count)
            assert success==True,f'failed extracting frame: {video_id}|{count} '
            image=cv2.resize(image,(456,256))
            cv2.imwrite(f'/media/luohwu/T7/dataset/ADL/rgb_frames/{video_id}/frame_{str(count).zfill(10)}.jpg',image)


if __name__=='__main__':

    #create dir for rgb_frames
    # make_dirs()
    for id in range(7,21):
        video_id=f'P_{str(id).zfill(2)}'
        print(f'start extracting frames: {video_id}')
        extract_frames_from_video(video_id)
        print('finished')



