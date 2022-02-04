from opt import *
import tarfile
import  os
import shutil
import pandas as pd
from ast import literal_eval
from itertools import chain
import cv2
from dataset import make_sequence_dataset
import glob
from PIL import Image
def make_dirs(output_dir):
    video_id_list=sorted(id)
    for video_id in video_id_list:
        participant_id=video_id[0:3]
        video_id=video_id[3:]

        target_dir1=os.path.join(output_dir,participant_id)
        target_dir2=os.path.join(output_dir,participant_id,video_id)
        if not os.path.exists(target_dir1):
            os.mkdir(target_dir1)
        if not os.path.exists(target_dir2):
            os.mkdir(target_dir2)

def move_tars():
    video_id_list=sorted(id)
    # print(video_id_list)
    data_path='/media/luohwu/T7/EpicKitchen/frames_rgb_flow/rgb'

    for video_id in video_id_list:
        participant_id=video_id[0:3]
        video_id=video_id[3:]
        file_path=os.path.join(data_path,'train',participant_id,f'{video_id}.tar')
        if not os.path.exists(file_path):
            file_path = os.path.join(data_path, 'test', participant_id, f'{video_id}.tar')
        print(file_path)
        target_file_path=os.path.join('/media/luohwu/T7/EpicKitchen/output',participant_id,f'{video_id}.tar')
        shutil.move(file_path,target_file_path)

def extract_frames_from_tar(base_only=False):
    video_id_list=sorted(id)
    # video_id_list=['P01P01_01']
    tar_data_path='/media/luohwu/T7/EpicKitchen/tarfiles'
    for video_id in video_id_list:
        participant_id=video_id[0:3]
        video_id=video_id[3:]
        file_path=os.path.join(tar_data_path,participant_id,f'{video_id}.tar')
        assert os.path.exists(file_path), f"file not exists: {file_path}"
        if base_only==True:
            target_dir = os.path.join('/media/luohwu/T7/dataset/EPIC/rgb_frames', participant_id, video_id)
        else:
            target_dir = os.path.join('/media/luohwu/T7/dataset/EPIC/rgb_frames', participant_id, video_id)
        tar=tarfile.open(file_path)
        print(f'start extracting: {file_path}')
        # only extract needed frames from tar files.
        tar.extractall(target_dir,members=py_files(tar,participant_id,video_id,base_only))
        tar.close()
        print(f'finished')


#given a video_id and its annotation file, output a generator containing only the names of needed frames.
def py_files(members,participant_id,video_id,basic_only=False):

    annos_file_path=os.path.join(args.data_path,'nao_annotations',f'nao_{video_id}.csv')
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
    #index start from 0
    # all_frames=[frame-1 if frame>2 else 1 for frame in all_frames]
    all_frames=[f'frame_{str(frame).zfill(10)}.jpg' for frame in all_frames]

    #remove redundant frames
    all_frames=set(all_frames)
    num=0
    for tarinfo in members:
        # exmaple of tarinfo.name = ./frame_0000044646.jpg
        if tarinfo.name[2:] in all_frames:
            # print(tarinfo.name[2:])
            num+=1
            yield tarinfo
    assert  num==len(all_frames),f"un-consistant numbers of frames in {video_id}"

def extract_frames_from_video():
    annos=pd.read_csv('/media/luohwu/T7/dataset/EPIC/nao_annotations/nao_P01P01_02.csv')
    frames=annos['frame']
    frames=sorted(frames)
    # vidcap = cv2.VideoCapture('/media/luohwu/T7/EpicKitchen/videos/P01/P01_16.mp4')
    vidcap = cv2.VideoCapture('/media/luohwu/T7/EpicKitchen/videos/P01/P01_02.m4v')
    # vidcap = cv2.VideoCapture('/home/luohwu/Videos/P01_02.mp4')
    if not vidcap.isOpened():
        print(f'failde opening video')
    success=True
    count = 0
    while True:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        count += 1
        if success:
            if count in frames:
                cv2.imwrite(f'/media/luohwu/T7/EpicKitchen/videos/P01/P01_02/frame_{str(count).zfill(10)}.jpg',image)

            # cv2.imshow('frame',image)
        else:
            print('fail reading')
            break
        if count==frames[-1]:
            break

        # print(count)
        # if count in frames:
        #     # print(count)
        #     if success==False:
        #         print(f'frame {count}: ', success)
        #     else:
        #         print(f'frame {count}: ')
        #         cv2.imwrite(
        #             f"/media/luohwu/T7/dataset/EPIC/output_frames_HD/P01/P01_01/frame_{str(count).zfill(10)}.jpg",
        #             image)  # save frame as JPEG file
        #         if count==frames[-1]:
        #             break



if __name__=='__main__':

    print('main')
    # m tar files
    ########################################################################
    # create dir for selected .tar files
    # make_dirs(output_dir='/media/luohwu/T7/dataset/EPIC/rgb_frames/')
    # make_dirs(output_dir='/media/luohwu/T7/dataset/EPIC/rgb_frames_resized/')

    # move only needed .tar files to our target dir
    # move_tars()

    # extract_frames_from_tar(base_only=True)

    # from video
    # extract_frames_from_video()

    root_dir='/media/luohwu/T7/dataset/EPIC/rgb_frames/'
    cnt=0
    for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
        cnt=cnt+1
        save_path=filename.replace('rgb_frames','rgb_frames_resized')
        # original_image=cv2.imread(filename)
        # resized_image=cv2.resize(original_image,(224,224))
        # cv2.imwrite(save_path,resized_image)

        original_image=Image.open(filename)
        resized_image=original_image.resize((224,224))
        resized_image.save(save_path)
        if cnt % 100==0:
            print(f'{cnt}: {save_path}')

