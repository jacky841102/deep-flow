#!/usr/bin/env python

# --------------------------------------------------------
# images search using k-nearest neighbors for dedupe project
# Written by Dengke Dong, 02.28.2016
# --------------------------------------------------------

"""Compute the optical flow of a video for each image."""

import os
import cv2
import sys
import time
import pprint
import argparse
import numpy as np
from math import *

disp_n = 2000

def _parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Compute the optical flow of a video for each image.')

  parser.add_argument('--im_first_path',    dest='im_first_path',  
  										help='images directory', required=True, type=str)
  parser.add_argument('--im_second_path',   dest='im_second_path', 
  										help='images directory', default=None, type=str)
  parser.add_argument('--out_dire',   dest='out_dire',	 
  										help='optical flow directory', type=str)
  parser.add_argument('--dm_file',    dest='dm_file',    
  										help="execued file of deep matching code", type=str)
  parser.add_argument('--df_file',    dest='df_file',    
  									  help="execued file of deep flow2 code",    type=str)
  parser.add_argument('--cf_file',    dest='cf_file',    
  										help="execued file of color flow code",    type=str)
  parser.add_argument('--bbox_file',    dest='bbox_file',    
  										help="the corresponding bbox file for input images",    
  										default=None, type=str)
  parser.add_argument('--origin_ims_dire',    dest='origin_ims_dire',    
  										help="origin images directory (not crop or ...)",    
  										default=None, type=str)
  parser.add_argument('--run_dm',     dest='run_dm',     
  										help='whether to run deep matching or not', 
  										default=0, type=int)
  parser.add_argument('--dm_options', dest='dm_options', 
  										help='params options of deep matching', 
  										default=None, type=str)
  parser.add_argument('--df_options', dest='df_options', 
  										help='params options of deep flow2',    
  										default=None, type=str)
  parser.add_argument('--cf_options', dest='cf_options', 
  										help='params options of color flow',    
  										default=None, type=str)
  parser.add_argument('--cmd_choice',    dest='cmd_choice', 	 
  										help='choice of command to run',  
  										default=0, type=int)
  parser.add_argument('--frame_n',    dest='frame_n', 	 
  										help='second frame',  default=-2, type=int)
  parser.add_argument('--rm_flo',     dest='rm_flo',     
  										help='remove flo files, if need',   
  										default=0, type=int)
  parser.add_argument('--rm_mth',     dest='rm_mth',     
  										help='remove match files, if need', 
  										default=0, type=int)
  parser.add_argument('--rm_ims',     dest='rm_ims',     
  										help='remove temp images, if need', 
  										default=0, type=int)
  parser.add_argument('--is_disp',    dest='is_disp',    
  										help='show info',  default=0, type=int)
  parser.add_argument('--im_disp',    dest='im_disp',    
  										help='show ims',   default=0, type=int)
  parser.add_argument('--sleep_time', dest='sleep_time', 
  										help='sleep time', default=3, type=int)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  print('Called with args:')
  print(args)
  print "\n\n"

  return args

def _create_dire(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def _get_test_data(im_path, im_ext=".jpg"):
	'''form (per line): `in_dire imgidx`'''
	print "\n\nGetting im_paths and imgidxs from `%s`.\n\n" % (im_path,)
	imgidxs  = []
	im_paths = []
	fh 			 = open(im_path)
	for line in fh.readlines():
		line = line.strip()
		info = line.split()

		n_info = len(info)
		assert n_info == 2

		in_dire = info[0].strip()
		imgidx  = info[1].strip()
		imgidxs.append(imgidx)
		im_path = in_dire + imgidx + im_ext
		im_paths.append(im_path)
	fh.close()
	print "\n\nGetting im_paths and imgidxs done.\n\n"

	return im_paths, imgidxs

def _im_paths(im_path, is_disp=False):
	im_path = im_path.strip()
	if os.path.isfile(im_path):
	  if im_path.endswith(".jpg") or im_path.endswith(".png") \
	      or im_path.endswith(".jpeg"): # # just an image (with other image extension?)
	    im_paths = [im_path]
	    imgidxs  = [os.path.basename(im_path).rsplit(".", 1)[0]]
	  else: # read from label file: contain im_dire imgidx
	    im_paths, imgidxs = _get_test_data(im_path)
	elif os.path.isdir(im_path):  # read from image directory
	  im_names = os.listdir(im_path)
	  assert len(im_names) >= 1
	  im_names = [im_name.strip() for im_name in im_names]
	  im_names.sort() # sort it for some convinience\
	  imgidxs  = [im_name.rsplit(".")[0] for im_name in im_names]
	  im_paths = [im_path + im_name      for im_name in im_names]
	else:
	  raise IOError(('\n\n{:s} not exist.\n\n').format(im_path))

	im_n = len(im_paths)
	assert im_n >= 1, "invalid input of `im_path`: " % (im_path,)
	assert im_n == len(imgidxs), "im_n: %s, len(imgidxs): %s" % \
																(im_n, len(imgidxs))
	if is_disp:
		print "\n\n"
		print "---- Images ----"
		print "im_path:", im_path, "\n"
		for j in xrange(im_n):
			imgidx  = imgidxs[j]
			im_path = im_paths[j]
			print imgidx, "->", im_path
		print "\n\n"

	return im_paths, imgidxs

def _im_pairs_v1(im_paths, frame_n, imgidxs1=None, is_disp=False):
	imgidxs2 = []
	im_pairs = []
	im_n 		 = len(im_paths)

	assert frame_n != 0
	assert im_n    >= 1 + abs(frame_n)

	# if frame_n < 0:
	# 	for idx in xrange(abs(frame_n)):
	# 		if idx + abs(frame_n) >= im_n:
	# 			continue
	# 		imgidxs2.append(imgidxs1[idx + abs(frame_n)])
	# 		im_pairs.append((im_paths[idx], im_paths[idx + abs(frame_n)]))

	# 	for idx in xrange(abs(frame_n), im_n):
	# 		imgidxs2.append(imgidxs1[idx + frame_n])
	# 		im_pairs.append((im_paths[idx], im_paths[idx + frame_n]))
	# else:
	# 	for idx in xrange(0, im_n - frame_n):
	# 		if idx + frame_n >= im_n:
	# 			continue
	# 		imgidxs2.append(imgidxs1[idx + frame_n])
	# 		im_pairs.append((im_paths[idx], im_paths[idx + frame_n]))

	# 	for idx in xrange(im_n - frame_n, im_n):
	# 		imgidxs2.append(imgidxs1[idx - frame_n])
	# 		im_pairs.append((im_paths[idx], im_paths[idx - frame_n]))

	for idx in xrange(im_n):
		j = idx + frame_n
		if j < 0 or j >= im_n:
			j = idx - frame_n
		imgidxs2.append(imgidxs1[j])
		im_pairs.append((im_paths[idx], im_paths[j]))

	assert len(im_pairs) >= im_n
	assert len(im_pairs) == len(imgidxs2)

	if is_disp:
		print "\n\n---- Pairs ----\n"
		for idx in xrange(im_n):
			imgidx2 = imgidxs2[idx]
			im_pair = im_pairs[idx]
			print imgidx2, "<-", im_pair
		print "\n\n"

	return im_pairs, imgidxs2

def _im_pairs_v2(im_first_paths, im_second_paths, is_disp=False):
	assert len(im_first_paths) >= 1
	assert len(im_first_paths) == len(im_second_paths)

	im_pairs = []
	im_n     = len(im_first_paths)

	for idx in xrange(im_n):
		im_first_path  = im_first_paths[idx]
		im_second_path = im_second_paths[idx]
		im_pairs.append((im_first_path, im_second_path))

	if is_disp:
		print "\n\n---- Pairs ----\n"
		for im_pair in im_pairs:
			print "im pair:", im_pair
		print "\n\n"

	return im_pairs

def _executed_files(args):
	dm_file = args.dm_file.strip()
	if not os.path.exists(dm_file) or not os.path.isfile(dm_file):
		raise IOError(('\n\n{:s} not exist.\n\n').format(dm_file))

	df_file = args.df_file.strip()
	if not os.path.exists(df_file) or not os.path.isfile(df_file):
		raise IOError(('\n\n{:s} not exist.\n\n').format(df_file))

	cf_file = args.cf_file.strip()
	if not os.path.exists(cf_file) or not os.path.isfile(cf_file):
		raise IOError(('\n\n{:s} not exist.\n\n').format(cf_file))

	return dm_file, df_file, cf_file

def _executed_options(args):
	dm_options = ""
	if args.dm_options is not None:
		dm_options = args.dm_options.strip()

	df_options = ""
	if args.df_options is not None:
		df_options = args.df_options.strip()

	cf_options = ""
	if args.cf_options is not None:
		cf_options = args.cf_options.strip()

	return dm_options, df_options, cf_options

def _rm_files(*args):
	for f in args:
		if os.path.exists(f) and os.path.isfile(f):
			os.remove(f)

def _executed_cmds(args, im_pairs, out_dire, dm_file, dm_options, df_file, df_options, cf_file, cf_options):
	''''''
	t_time     = time.time()
	im_n       = len(im_pairs)
	im_disp		 = True if args.im_disp != 0 else False
	run_dm		 = True if args.run_dm  == 0 else False
	rm_flo		 = True if args.rm_flo  != 0 else False
	rm_mth		 = True if args.rm_mth  != 0 else False
	rm_ims		 = True if args.rm_ims  != 0 else False

	for im_i in xrange(im_n):
		s_time   				= time.time()
		im_pair 				= im_pairs[im_i]
		im_path1        = im_pair[0] # main -> match
		im_name1        = os.path.basename(im_path1)
		imgidx1, im_ext = im_name1.rsplit(".", 1)
		im_path2 				= im_pair[1] # auxi
		im_name2        = os.path.basename(im_path2)
		imgidx2, _      = im_name2.rsplit(".", 1)
		im1 					  = cv2.imread(im_path1)
		im2 						= cv2.imread(im_path2)
		h1, w1, _ 			= im1.shape
		h2, w2, _ 			= im2.shape

		im_tpath1 = out_dire + imgidx1 + "_tmp1" + "." + im_ext
		im_tpath2 = out_dire + imgidx1 + "_tmp2" + "." + im_ext
		print "im_path1:", im_path1
		print "im_path2:", im_path2
		# ##############################################################
		mh, mw = None, None
		if h1 == h2 and w1 == w2:
			cv2.imwrite(im_tpath1, im1)
			cv2.imwrite(im_tpath2, im2)
			mh = h1
			mw = w1
		else:
			mh = max(h1, h2)
			mw = max(w1, w2)
			t_im1 = np.zeros((mh, mw, 3), dtype=np.uint8)
			t_im2 = np.zeros((mh, mw, 3), dtype=np.uint8)
			t_im1[:h1, :w1] = im1
			t_im2[:h2, :w2] = im2
			cv2.imwrite(im_tpath1, t_im1)
			cv2.imwrite(im_tpath2, t_im2)
		
		mth_path = out_dire + imgidx1 + "_match.txt"
		of_path1 = out_dire + imgidx1 + "_tmp.png"
		of_path2 = out_dire + imgidx1 + ".png"
		flo_path = out_dire + imgidx1 + ".flo"
		# ##############################################################
		dm_cmd   = "python %s %s %s %s -out %s" % (dm_file, im_tpath1, \
										im_tpath2, dm_options, mth_path)
		if run_dm:
			df_cmd   = "%s %s %s %s %s -match %s" % (df_file, im_tpath1, \
										im_tpath2, flo_path, df_options, mth_path)
		else:
			df_cmd   = "%s %s %s %s %s" % (df_file, im_tpath1, im_tpath2, \
										flo_path, df_options)
		cf_cmd   = "%s %s %s" % (cf_file, flo_path, of_path1)
		# ##############################################################
		if not os.path.exists(of_path2) or not os.path.isfile(of_path2):
			if run_dm:
				if not os.path.exists(flo_path) or \
					 not os.path.isfile(flo_path) or \
				   not os.path.exists(mth_path) or \
				   not os.path.isfile(mth_path):
					print "\nrun deep-matching\n"
					cmd = dm_cmd
					os.system(cmd)
				
			cmd = df_cmd
			os.system(cmd)

			if os.path.exists(flo_path) and os.path.isfile(flo_path):
				cmd = cf_cmd
				os.system(cmd)
			else:	
				raise IOError(('\n\n{:s} not exist.\n\n').format(flo_path))
			# ##############################################################
			of_im1        = cv2.imread(of_path1)
			of_h, of_w, _ = of_im1.shape
			assert mh == of_h
			assert mw == of_w
			assert h1 <= of_h
			assert w1 <= of_w
			of_im2 				= of_im1[:h1, :w1]
			cv2.imwrite(of_path2, of_im2)
		# ##############################################################
		if im_disp:
			im1 = cv2.imread(im_tpath1)
			im2 = cv2.imread(im_tpath2)
			im3 = cv2.imread(of_path2)
			cv2.imshow(imgidx1, im1)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imshow(imgidx2, im2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imshow(imgidx1, im3)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		if rm_ims:
			if os.path.exists(im_tpath1) and os.path.isfile(im_tpath1):
				os.remove(im_tpath1)
			if os.path.exists(im_tpath2) and os.path.isfile(im_tpath2):
				os.remove(im_tpath2)
			if os.path.exists(of_path1) and os.path.isfile(of_path1):
				os.remove(of_path1)
		if rm_flo and os.path.exists(flo_path) and os.path.isfile(flo_path):
			os.remove(flo_path)
		if rm_mth and os.path.exists(mth_path) and os.path.isfile(mth_path):
			os.remove(mth_path) 
		print "\n\nTakes %s seconds for %s-th image: %s\n\n" % \
				  (time.time() - s_time, im_i, im_path1)
		print "\n#################################################\n"
	# ##############################################################
	t_time = time.time() - t_time
	print "\n\nTakes %s seconds for %s images (aver_time: %s)\n\n" % \
					(t_time, im_n, t_time / im_n)
	print "\nDone\n"

def _bboxes(bbox_file):
	'''form (per line): `imgidx objidx x1 y1 x2 y2`'''
	print "\n\nGetting bbox of person from `%s`.\n\n" % (bbox_file,)
	if bbox_file is None:
		return False, None
	fh 		= open(bbox_file)
	lines = fh.readlines()
	fh.close()

	disp_c = 0
	bc     = {}
	bboxes = {}
	for line in lines:
		disp_c += 1
		if disp_c % disp_n == 0:
			print "im_i:", disp_c
		line = line.strip()
		info = line.split()
		imgidx, objidx, bbox = info[0], info[1], info[2:]
		assert len(bbox) == 4
		bbox = [int(pb) for pb in bbox]
		if imgidx not in bboxes.keys():
			bboxes[imgidx] = bbox
		else:
			raise IOError(('\n\n{:s} has been in `bboxes` dictionary.\n\n').format(imgidx))
	if disp_c % disp_n != 0:
			print "im_i:", disp_c
	print "\n\nGetting bbox of person done!\n\n"

	return True, bboxes

def _executed_cmds_bbox_file(args, im_pairs, imgidxs1, imgidxs2, out_dire, \
														 dm_file, dm_options, df_file, df_options, cf_file, \
														 cf_options, im_ext=".jpg"):
	''''''
	t_time     = time.time()
	im_n       = len(im_pairs)
	im_disp		 = True if args.im_disp != 0 else False
	run_dm		 = True if args.run_dm  == 0 else False
	rm_flo		 = True if args.rm_flo  != 0 else False
	rm_mth		 = True if args.rm_mth  != 0 else False
	rm_ims		 = True if args.rm_ims  != 0 else False

	bbox_file        = args.bbox_file 
	has_bbox, bboxes = _bboxes(bbox_file);
	origin_ims_dire  = args.origin_ims_dire
	if has_bbox:
		assert im_n   <= len(bboxes), "im_n: %s, len(bboxes): %s\n" % \
																	(im_n, len(bboxes))
		if origin_ims_dire is None:
			raise IOError(('\n\n{:s} not exist.\n\n').format(origin_ims_dire))
	print "\n\nLoading bbox_file from '%s' done.\n\n" % (bbox_file,)
	print "im_n:", im_n, "\n\n"
	time.sleep(args.sleep_time)

	for im_i in xrange(im_n):
		s_time   				= time.time()
		im_pair 				= im_pairs[im_i]
		imgidx1					= imgidxs1[im_i]
		imgidx2 				= imgidxs2[im_i]
		
		im_path1        = im_pair[0] # main -> match
		im_name1        = os.path.basename(im_path1)

		im_path2 				= im_pair[1] # auxi
		im_name2        = os.path.basename(im_path2)
		# print "im_path1:", im_path1
		# print "im_path2:", im_path2
		im1 					  = cv2.imread(im_path1)
		im2 						= cv2.imread(im_path2)
		print
		print "im_path1:", im_path1
		print "im_path2:", im_path2
		h1, w1, _ 			= im1.shape
		h2, w2, _ 			= im2.shape

		im_tpath1 = out_dire + imgidx1 + "_tmp1" + im_ext
		im_tpath2 = out_dire + imgidx1 + "_tmp2" + im_ext
		# print "im_tpath1:", im_tpath1
		# print "im_tpath2:", im_tpath2
		dire_ct   = os.path.dirname(im_tpath1)
		_create_dire(dire_ct)
		# ##############################################################
		origin_eq       = False
		mh, mw, mx, my  = None, None, None, None
		if h1 == h2 and w1 == w2:
			cv2.imwrite(im_tpath1, im1)
			cv2.imwrite(im_tpath2, im2)
			mh = h1
			mw = w1
			origin_eq = True
		else:
			if has_bbox:
				bbox1 = bboxes[imgidx1]
				bbox2 = bboxes[imgidx2]
				origin_im_path1 = origin_ims_dire + imgidx1 + im_ext
				origin_im_path2 = origin_ims_dire + imgidx2 + im_ext
				# print "origin_im_path1:", origin_im_path1
				# print "origin_im_path2:", origin_im_path2
				print 
				origin_im1      = cv2.imread(origin_im_path1)
				oh1, ow1, _     = origin_im1.shape
				origin_im2      = cv2.imread(origin_im_path2)
				oh2, ow2, _     = origin_im2.shape
				assert oh1 == oh2
				assert ow1 == ow2
				x1 = min(bbox1[0], bbox2[0])
				y1 = min(bbox1[1], bbox2[1])
				x2 = max(bbox1[2], bbox2[2])
				y2 = max(bbox1[3], bbox2[3])
				mx = x1
				my = y1
				mh = y2 - y1
				mw = x2 - x1
				print "bbox1:", bbox1
				print "bbox2:", bbox2
				print "x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2
				print "mx:", mx, "my:", my, "mh:", mh, "mw:", mw
				t_im1 = origin_im1[y1:y2, x1:x2]
				t_im2 = origin_im2[y1:y2, x1:x2]
			else:
				mh = max(h1, h2)
				mw = max(w1, w2)
				print "mh:", mh, "mw:", mw
				t_im1 = np.zeros((mh, mw, 3), dtype=np.uint8)
				t_im2 = np.zeros((mh, mw, 3), dtype=np.uint8)
				t_im1[:h1, :w1] = im1
				t_im2[:h2, :w2] = im2
			cv2.imwrite(im_tpath1, t_im1)
			cv2.imwrite(im_tpath2, t_im2)
		
		mth_path = out_dire + imgidx1 + "_match.txt"
		of_path1 = out_dire + imgidx1 + "_tmp.png"
		of_path2 = out_dire + imgidx1 + ".png"
		flo_path = out_dire + imgidx1 + ".flo"
		# ##############################################################
		dm_cmd   = "python %s %s %s %s -out %s" % (dm_file, im_tpath1, \
										im_tpath2, dm_options, mth_path)
		if run_dm:
			df_cmd   = "%s %s %s %s %s -match %s" % (df_file, im_tpath1, \
										im_tpath2, flo_path, df_options, mth_path)
		else:
			df_cmd   = "%s %s %s %s %s" % (df_file, im_tpath1, im_tpath2, \
										flo_path, df_options)
		cf_cmd   = "%s %s %s" % (cf_file, flo_path, of_path1)
		# ##############################################################
		if not os.path.exists(of_path2) or not os.path.isfile(of_path2):
			if run_dm:
				if not os.path.exists(flo_path) or \
					 not os.path.isfile(flo_path) or \
				   not os.path.exists(mth_path) or \
				   not os.path.isfile(mth_path):
					print "\nrun deep-matching\n"
					cmd = dm_cmd
					os.system(cmd)
				
			cmd = df_cmd
			os.system(cmd)

			if os.path.exists(flo_path) and os.path.isfile(flo_path):
				cmd = cf_cmd
				os.system(cmd)
			else:	
				raise IOError(('\n\n{:s} not exist.\n\n').format(flo_path))
			# ##############################################################
			of_im1          = cv2.imread(of_path1)
			of_h1, of_w1, _ = of_im1.shape
			assert mh == of_h1
			assert mw == of_w1
			# assert h1 <= of_h1, "h1: %s, of_h1: %s" % (h1, of_h1)
			# assert w1 <= of_w1, "w1: %s, of_w1: %s" % (w1, of_w1)
			of_im2  = None
			if origin_eq:
				of_im2 = of_im1
			elif has_bbox:
				bbox1 = bboxes[imgidx1]
				x1    = bbox1[0] - mx
				y1    = bbox1[1] - my
				x2    = bbox1[2] - mx
				y2    = bbox1[3] - my
				of_im2	= of_im1[y1:y2, x1:x2]
			else:
				of_im2	= of_im1[:h1, :w1]
			of_h2, of_w2, _ = of_im2.shape
			# assert of_h2 == h1
			# assert of_w2 == w1
			if of_h2 != h1 or of_w2 != w1:
				of_im2 = cv2.resize(of_im2, (w1, h1), \
														interpolation=cv2.INTER_LINEAR)
				print "non-eq shape ->", "h1:", h1, "w1:", w1, \
							"of_h2:", of_h2, "of_w2:", of_w2
			cv2.imwrite(of_path2, of_im2)
		# ##############################################################
		if im_disp:
			im1 = cv2.imread(im_tpath1)
			im2 = cv2.imread(im_tpath2)
			im3 = cv2.imread(of_path2)
			cv2.imshow(imgidx1, im1)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imshow(imgidx2, im2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imshow(imgidx1, im3)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		if rm_ims:
			if os.path.exists(im_tpath1) and os.path.isfile(im_tpath1):
				os.remove(im_tpath1)
			if os.path.exists(im_tpath2) and os.path.isfile(im_tpath2):
				os.remove(im_tpath2)
			if os.path.exists(of_path1) and os.path.isfile(of_path1):
				os.remove(of_path1)
		if rm_flo and os.path.exists(flo_path) and os.path.isfile(flo_path):
			os.remove(flo_path)
		if rm_mth and os.path.exists(mth_path) and os.path.isfile(mth_path):
			os.remove(mth_path) 
		print "\n\nTakes %s seconds for %s-th image: %s\n\n" % \
				  (time.time() - s_time, im_i, im_path1)
		print "\n#################################################\n"
	# ##############################################################
	t_time = time.time() - t_time
	print "\n\nTakes %s seconds for %s images (aver_time: %s)\n\n" % \
					(t_time, im_n, t_time / im_n)
	print "\nDone\n"

def flow_pipeline():
	''''''
	args 										 = _parse_args()

	frame_n 								 = args.frame_n

	im_first_path 	  			 = args.im_first_path.strip()

	out_dire 								 = args.out_dire.strip()

	is_disp									 = True if args.is_disp != 0 else False

	im_first_paths, imgidxs1 = _im_paths(im_first_path, is_disp)

	if args.im_second_path is None:
		print "\nPair from sequence\n"
		time.sleep(args.sleep_time)
		im_pairs, imgidxs2 = _im_pairs_v1(im_first_paths, frame_n, \
																	    imgidxs1=imgidxs1, is_disp=is_disp)
	else:
		print "\nPair from directory or files or image\n"
		time.sleep(args.sleep_time)
		im_second_path     				= args.im_second_path.strip()
		im_second_paths, imgidxs2 = _im_paths(im_second_path, is_disp)
		im_pairs         				  = _im_pairs_v2(im_first_paths, im_second_paths, \
																						 is_disp)

	dm_file, df_file, cf_file = _executed_files(args)

	dm_options, df_options, cf_options = _executed_options(args)

	time.sleep(args.sleep_time)

	cmd_choice = args.cmd_choice
	if cmd_choice == 0:
		_executed_cmds(args, im_pairs, out_dire, dm_file, dm_options, \
									 df_file, df_options, cf_file, cf_options)
	elif cmd_choice == 1:
		_executed_cmds_bbox_file(args, im_pairs, imgidxs1, imgidxs2, out_dire, \
									 					 dm_file, dm_options, df_file, df_options, \
									 					 cf_file, cf_options)
	else:
		raise IOError(('\n\ninvalid cmd_choice: {:s}.\n\n').format(cmd_choice))

if __name__ == '__main__':
	''''''
	flow_pipeline()
