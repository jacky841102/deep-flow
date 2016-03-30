#!/bin/sh

# im_first_path="/home/ddk/cv.life/Nutstore/graduate.project/demo.images/bp1.jpg"
# im_second_path="/home/ddk/cv.life/Nutstore/graduate.project/demo.images/bp2.jpg"

im_first_path="/home/ddk/dongdk/faster-rcnn/output/person.torso/demo/mude.images1/mude.16.02.20000039.jpg"
im_second_path="/home/ddk/dongdk/faster-rcnn/output/person.torso/demo/mude.images1/mude.16.02.20000425.jpg"

out_dire="/home/ddk/cv.life/Nutstore/graduate.project/demo.images/"
mkdir -p $out_dire

cmd_choice=0 # 0: normal, 1: from bbox, other: NotImplemented

dm_file="/home/ddk/dongdk/deep-matching/deep_matching/deep_matching_gpu.py"

df_file="/home/ddk/dongdk/deep-matching/deep_flow2/deepflow2"

cf_file="/home/ddk/dongdk/deep-matching/color_flow/color_flow"

rm_flo=1  # 0: not remove (default), !=0: remove

rm_mth=1  # 0: not remove (default), !=0: remove

rm_ims=1  # 0: not remove (default), !=0: remove

run_dm=0  # 0: run deep-matching, !=0: not run

is_disp=1 # 0: print nothing, !=0: display info

im_disp=0 # 0: not show, !=0: show im

sleep_time=1

# dm: "--ngh_rad 256(192,) --use_sparse"
python flow_pipeline.py \
		--run_dm $run_dm \
		--rm_flo $rm_flo \
		--rm_mth $rm_mth \
		--rm_ims $rm_ims \
		--is_disp $is_disp \
		--im_disp $im_disp \
		--dm_file $dm_file \
		--df_file $df_file \
		--cf_file $cf_file \
		--out_dire $out_dire \
		--cmd_choice $cmd_choice \
		--im_first_path $im_first_path \
		--im_second_path $im_second_path \
		--cf_options "" \
		--df_options "-sintel " \
		--dm_options "-form_type 1 -GPU -v --downscale 1 --ngh_rad 256 " \
		--sleep_time $sleep_time \
