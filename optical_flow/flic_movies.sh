#!/bin/sh

root_dire="/home/ddk/dongdk/dataset/video.pose/FLIC.Motion-Ims-Src/"
im_first_path="${root_dire}itself/"
im_second_path="${root_dire}negative.2/"

out_dire="${root_dire}optical.flow/negative.2/"
mkdir -p $out_dire

frame_n=-2

dm_file="/home/ddk/dongdk/deep-matching/deep_matching/deep_matching_gpu.py"

df_file="/home/ddk/dongdk/deep-matching/deep_flow2/deepflow2"

cf_file="/home/ddk/dongdk/deep-matching/color_flow/color_flow"

rm_flo=1 # 0: not remove (default), !=0: remove

rm_mth=1 # 0: not remove (default), !=0: remove

is_disp=0 # 0: print nothing, !=0: display info

sleep_time=1

# dm: "--ngh_rad 256 --use_sparse"
# use 
python flow_pipeline.py \
		--rm_flo $rm_flo \
		--rm_mth $rm_mth \
		--is_disp $is_disp \
		--dm_file $dm_file \
		--df_file $df_file \
		--cf_file $cf_file \
		--frame_n $frame_n \
		--out_dire $out_dire \
		--im_first_path $im_first_path \
		--im_second_path $im_second_path \
		--cf_options "" \
		--df_options "-sintel " \
		--dm_options "-form_type 1 -GPU -v --downscale 1 --ngh_rad 320 " \
		--sleep_time $sleep_time \
