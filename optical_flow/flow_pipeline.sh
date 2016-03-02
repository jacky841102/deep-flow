#!/bin/sh

# 
im_first_path="/home/ddk/dongdk/deep-matching/images/itself/"

# im_second_path="/home/ddk/dongdk/deep-matching/images/second/"

out_dire="/home/ddk/dongdk/deep-matching/images/flow/"
mkdir -p $out_dire

frame_n=-2

dm_file="/home/ddk/dongdk/deep-matching/deep_matching/deep_matching_gpu.py"

df_file="/home/ddk/dongdk/deep-matching/deep_flow2/deepflow2"

cf_file="/home/ddk/dongdk/deep-matching/color_flow/color_flow"

rm_flo=1 # 0: remove (default), !=0: remove

is_disp=1 # 0: print nothing, !=0: display info

sleep_time=1

# use 
python flow_pipeline.py \
		--rm_flo $rm_flo \
		--is_disp $is_disp \
		--dm_file $dm_file \
		--df_file $df_file \
		--cf_file $cf_file \
		--frame_n $frame_n \
		--out_dire $out_dire \
		--im_first_path $im_first_path \
		--cf_options "" \
		--df_options "-match " \
		--dm_options "-GPU -v --downscale 1 --ngh_rad 256 --use_sparse " \
		--sleep_time $sleep_time \
