from ast import literal_eval

import pandas as pd

from opt import *

def resize_bbox(row,height,width,new_height,new_width):
    bbox=row["nao_bbox"]
    new_bbox= [bbox[0]/width*new_width,bbox[1]/height*new_height,bbox[2]/width*new_width,bbox[3]/height*new_height]

    new_bbox= [round(coord) for coord in new_bbox]
    new_bbox[0] = new_width if new_bbox[0] > new_width else new_bbox[0]
    new_bbox[2] = new_width if new_bbox[2] > new_width else new_bbox[2]
    new_bbox[1] = new_height if new_bbox[1] > new_height else new_bbox[1]
    new_bbox[3] = new_height if new_bbox[3] > new_height else new_bbox[3]
    return new_bbox


# given each video's resolution and fps
# 1.resize the bounding box to 456x256 (size of donwloader RGB frames)
# 2.add frames before each train/test sample according to sample_time_length (how long before the sample frame) and
#   sample_fps ( how many frames in each sampled second)
def add_previous_frames(sample_time_length=5,sample_fps=3):
    # video_info_path = os.path.join(args.data_path, 'EPIC_video_info.csv')
    # video_info = pd.read_csv(video_info_path)
    video_id_list = sorted(id)
    for video_id in video_id_list:
        anno_file_path = os.path.join(args.data_path, annos_path, f'nao_{video_id}.csv')
        if os.path.exists(anno_file_path):
            print(f'current video id: {video_id}')
            fps = 30
            # the annotations of ADL data were made using 640x480 images
            height = 480
            width = 640
            sample_steps = fps // sample_fps
            previous_frames_helper = [-sample_steps * timestep for timestep in
                                      range(sample_fps * sample_time_length, 0, -1)]
            annotations = pd.read_csv(anno_file_path, converters={"nao_bbox": literal_eval})
            if annotations.shape[0] > 0:
                annotations['fps'] = fps
                # print(previous_frames_helper)
                annotations['previous_frames'] = annotations.apply(
                    lambda row: ([(row['frame'] + i) for i in previous_frames_helper]), axis=1)
                annotations['previous_frames'] = annotations.apply(
                    lambda row: [frame if frame > 0 else 1 for frame in row['previous_frames']], axis=1)
                annotations['nao_bbox_resized'] = annotations.apply(resize_bbox, args=[height, width,256,456], axis=1)
                annotations['nao_bbox_resized2'] = annotations.apply(resize_bbox, args=[height, width,224,224], axis=1)
                annotations=annotations.drop(columns={'nao_bbox'})
                annotations=annotations.rename(columns={'nao_bbox_resized':'nao_bbox'})
                annotations=annotations.rename(columns={'nao_bbox_resized2':'nao_bbox_resized'})
                annotations.to_csv(anno_file_path, index=False)

# original annotation files are .txt files and have different data format from those EPIC annotations
# conver annotation files to .csv files first
def convert_format_to_Epic():
    items = []
    video_id_list=id

    for video_id in sorted(video_id_list):
        img_path = os.path.join(args.data_path, frames_path, video_id)

        anno_name = 'nao_' + video_id + '.txt'
        anno_name_csv = 'nao_' + video_id + '.csv'
        anno_file_path = os.path.join(args.data_path, annos_path, anno_name)
        anno_file_path_csv = os.path.join(args.data_path, annos_path, anno_name_csv)
        assert  os.path.exists(anno_file_path)
        annos = pd.read_csv(anno_file_path, header=None,
                            delim_whitespace=True, converters={0: str},
                            names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                   'frame_id', 'is_active', 'object_label',
                                   'is_next_active'])

        annos = annos[annos['is_next_active'] == 1]
        annos['nao_bbox']=annos.apply(lambda row: [row['x1'], row['y1'], row['x2'], row['y2']],axis=1)
        annos['id']=video_id
        annos=annos.rename(columns={"frame_id":"frame","object_label":"label"})
        annos=annos[['frame','id','label','nao_bbox']]
        annos=annos.rename(columns= {'label':'class'})
        annos.to_csv(anno_file_path_csv,index=False)

if __name__=='__main__':
    convert_format_to_Epic()
    add_previous_frames(sample_time_length=10,sample_fps=1)