#!/bin/sh

# im dire
root_dire="/home/ddk/dongdk/dataset/video.pose/FLIC.Motion-Ims-Src/"
im_first_path="${root_dire}itself/"
im_second_path="${root_dire}positive.2/"

out_dire="${root_dire}optical.flow/flic.pos.2/"
mkdir -p $out_dire

frame_n=2

dm_file="/home/ddk/dongdk/deep-matching/deep_matching/deep_matching_gpu.py"

df_file="/home/ddk/dongdk/deep-matching/deep_flow2/deepflow2"

cf_file="/home/ddk/dongdk/deep-matching/color_flow/color_flow"

rm_flo=1  # 0: not remove (default), !=0: remove

rm_mth=1  # 0: not remove (default), !=0: remove

run_dm=0  # 0: run deep-matching, !=0: not run

is_disp=1 # 0: print nothing, !=0: display info

im_disp=0 # 0: not show, !=0: show im

sleep_time=4

# dm: "--ngh_rad 256(192,) --use_sparse"
python flow_pipeline.py \
		--run_dm $run_dm \
		--rm_flo $rm_flo \
		--rm_mth $rm_mth \
		--is_disp $is_disp \
		--im_disp $im_disp \
		--dm_file $dm_file \
		--df_file $df_file \
		--cf_file $cf_file \
		--frame_n $frame_n \
		--out_dire $out_dire \
		--im_first_path $im_first_path \
		--cf_options "" \
		--df_options "-sintel " \
		--dm_options "-form_type 1 -GPU -v --downscale 1 --ngh_rad 216 " \
		--sleep_time $sleep_time \
		--im_second_path $im_second_path \
