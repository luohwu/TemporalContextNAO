from ast import literal_eval

import pandas as pd

from opt import *

def resize_bbox(row,height,width):
    bbox=row["nao_bbox"]
    new_bbox= [bbox[0]/width*456,bbox[1]/height*256,bbox[2]/width*456,bbox[3]/height*256]

    new_bbox= [round(coord) for coord in new_bbox]
    new_bbox[0] = 455 if new_bbox[0] > 455 else new_bbox[0]
    new_bbox[2] = 455 if new_bbox[2] > 455 else new_bbox[2]
    new_bbox[1] = 255 if new_bbox[1] > 255 else new_bbox[1]
    new_bbox[3] = 255 if new_bbox[3] > 255 else new_bbox[3]
    # return bbox
    return new_bbox




# given each video's resolution and fps
# 1.resize the bounding box to 456x256 (size of donwloader RGB frames)
# 2.add frames before each train/test sample according to sample_time_length (how long before the sample frame) and
#   sample_fps ( how many frames in each sampled second)
def add_previous_frames(sample_time_length=5,sample_fps=3):
    video_info_path = os.path.join(args.data_path, 'EPIC_video_info.csv')
    video_info = pd.read_csv(video_info_path)
    all_par_video_id = sorted(id)
    for par_video_id in all_par_video_id:
        video_id = par_video_id[3:]
        anno_file_path = os.path.join(args.data_path, annos_path, f'nao_{par_video_id}.csv')
        if os.path.exists(anno_file_path):
            print(f'current video id: {video_id}')
            current_video_ino = video_info.loc[video_info['video'] == video_id]
            fps = current_video_ino.iloc[0]["fps"]
            resolution = current_video_ino.iloc[0]["resolution"]
            width = int(resolution[0:4])
            height = int(resolution[5:])
            sample_steps = fps // sample_fps
            previous_frames_helper = [-sample_steps * timestep for timestep in
                                      range(sample_fps * sample_time_length, 0, -1)]
            annotations = pd.read_csv(anno_file_path, converters={"nao_bbox": literal_eval})
            if annotations.shape[0] > 0:
                annotations['participant_id'] = annotations.apply(lambda row: row["id"][0:3], axis=1)
                annotations['video_id'] = annotations.apply(lambda row: row["id"][3:], axis=1)
                # annotations['frame'] = annotations.apply(lambda row: row["frame"]-30, axis=1)

                annotations['fps'] = fps
                # print(previous_frames_helper)
                annotations['previous_frames'] = annotations.apply(
                    lambda row: ([(row['frame'] + i) for i in previous_frames_helper]), axis=1)
                annotations['previous_frames'] = annotations.apply(
                    lambda row: [frame if frame > 0 else 1 for frame in row['previous_frames']], axis=1)
                annotations['nao_bbox_resized'] = annotations.apply(resize_bbox, args=[height, width], axis=1)
                annotations.to_csv(anno_file_path, index=False)


if __name__=='__main__':
    add_previous_frames(sample_time_length=5,sample_fps=2)