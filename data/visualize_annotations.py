import cv2
import numpy as np
import pandas as pd
from opt import *
import os
from ast import literal_eval


def visualize(par_video_id):
    if args.dataset=='EPIC':
        par_id=par_video_id[0:3]
        video_id=par_video_id[3:]
        annos_file_path=os.path.join(args.data_path,annos_path,f'nao_{par_video_id}.csv')
    else:
        annos_file_path=os.path.join(args.data_path,annos_path,f'nao_{par_video_id}.csv')
        video_id=par_video_id
    if os.path.exists(annos_file_path):
        print(annos_file_path)
        annos=pd.read_csv(annos_file_path)
        for i in range(0,annos.shape[0]):
            df_item = annos.iloc[i, :]
            nao_bbox=literal_eval(df_item.nao_bbox_resized)
            frame_id=df_item.frame
            frame_name=f'frame_{str(frame_id).zfill(10)}.jpg'
            if args.dataset=='EPIC':
                image_file_path = os.path.join(args.data_path, frames_path, par_id, video_id, frame_name)
            else:
                image_file_path = os.path.join(args.data_path, frames_path, video_id, frame_name)
            assert os.path.exists(image_file_path),f'file not exists: {image_file_path}'
            image=cv2.imread(image_file_path)
            #nao annotations
            cv2.rectangle(image,(nao_bbox[0],nao_bbox[1]),(nao_bbox[2],nao_bbox[3]),(0,255,0),3)

            #hand annos
            hand_bbox_list=df_item.hand_bbox
            if  hand_bbox_list is not np.nan:
                for hand_bbox in literal_eval(hand_bbox_list):
                    cv2.rectangle(image,(hand_bbox[0],hand_bbox[1]),(hand_bbox[2],hand_bbox[3]),(0,0,255),3)

            cv2.imshow(f'{frame_name}',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__=='__main__':
    par_video_id_list=['P12P12_01','P12P12_02', 'P12P12_04' ]
    # par_video_id_list=['P_11','P_02', 'P_03' ]
    for par_video_id in par_video_id_list:
        visualize(par_video_id)