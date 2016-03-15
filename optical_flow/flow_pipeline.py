#!/usr/bin/env python

# --------------------------------------------------------
# images search using k-nearest neighbors for dedupe project
# Written by Dengke Dong, 02.28.2016
# --------------------------------------------------------

"""Compute the optical flow of a video for each image."""

import os
import sys
import time
import pprint
import argparse
from math import *

def _parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Compute the optical flow of a video for each image.')

  parser.add_argument('--im_first_path',    dest='im_first_path',  help='images directory', required=True, type=str)
  parser.add_argument('--im_second_path',   dest='im_second_path', help='images directory', default=None, type=str)
  parser.add_argument('--out_dire',   dest='out_dire',	 help='optical flow directory', type=str)
  parser.add_argument('--dm_file',    dest='dm_file',    help="execued file of deep matching code", type=str)
  parser.add_argument('--df_file',    dest='df_file',    help="execued file of deep flow2 code",    type=str)
  parser.add_argument('--cf_file',    dest='cf_file',    help="execued file of color flow code",    type=str)
  parser.add_argument('--dm_options', dest='dm_options', help='params options of deep matching', default=None, type=str)
  parser.add_argument('--df_options', dest='df_options', help='params options of deep flow2',    default=None, type=str)
  parser.add_argument('--cf_options', dest='cf_options', help='params options of color flow',    default=None, type=str)
  parser.add_argument('--frame_n',    dest='frame_n', 	 help='second frame',  default=-2, type=int)
  parser.add_argument('--rm_flo',     dest='rm_flo',     help='remove flo files, if need',  default=0, type=int)
  parser.add_argument('--rm_mth',     dest='rm_mth',     help='remove match files, if need',  default=0, type=int)
  parser.add_argument('--is_disp',    dest='is_disp',    help='show info',  default=0, type=int)
  parser.add_argument('--sleep_time', dest='sleep_time', help='sleep time', default=3, type=int)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  print('Called with args:')
  print(args)
  print "\n\n"

  return args

def _im_paths(im_path, is_disp=False):
	im_path = im_path.strip()
	if os.path.isfile(im_path):
	  if im_path.endswith(".jpg") or im_path.endswith(".png") \
	      or im_path.endswith(".jpeg"): # # just an image (with other image extension?)
	    im_paths = [im_path]
	  else: # read from label file: contain im_path [label ...]
	    im_paths, _ = _get_test_data(im_path)
	elif os.path.isdir(im_path):  # read from image directory
	  im_names = os.listdir(im_path)
	  assert len(im_names) >= 1
	  im_names.sort() # sort it for some convinience
	  im_paths = [im_path + im_name.strip() for im_name in im_names]
	else:
	  raise IOError(('{:s} not exist').format(im_path))

	im_n = len(im_paths)
	assert im_n >= 1, "invalid input of `im_path`: " % (im_path,)

	if is_disp:
		print "\n\n"
		print "---- Images ----"
		print "im_path:", im_path, "\n"
		for  im_path in im_paths:
			print im_path
		print "\n\n"

	return im_paths

def _im_pairs_v1(im_paths, frame_n, is_disp=False):
	im_pairs = []
	im_n 		 = len(im_paths)

	assert frame_n != 0
	assert im_n    >= 1 + abs(frame_n)

	if frame_n < 0:
		for idx in xrange(abs(frame_n)):
			if idx + abs(frame_n) >= im_n:
				continue
			im_pairs.append((im_paths[idx], im_paths[idx + abs(frame_n)]))
		for idx in xrange(abs(frame_n), im_n):
			im_pairs.append((im_paths[idx], im_paths[idx + frame_n]))
	else:
		for idx in xrange(0, im_n - frame_n):
			if idx + frame_n >= im_n:
				continue
			im_pairs.append((im_paths[idx], im_paths[idx + frame_n]))

		for idx in xrange(im_n - frame_n, im_n):
			im_pairs.append((im_paths[idx], im_paths[idx - frame_n]))
	assert len(im_pairs) >= 1

	if is_disp:
		print "\n\n---- Pairs ----\n"
		for im_pair in im_pairs:
			print im_pair
		print "\n\n"

	return im_pairs

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
			print im_pair
		print "\n\n"

	return im_pairs

def _executed_files(args):
	dm_file = args.dm_file.strip()
	if not os.path.exists(dm_file) or not os.path.isfile(dm_file):
		raise IOError(('{:s} not exist').format(dm_file))

	df_file = args.df_file.strip()
	if not os.path.exists(df_file) or not os.path.isfile(df_file):
		raise IOError(('{:s} not exist').format(df_file))

	cf_file = args.cf_file.strip()
	if not os.path.exists(cf_file) or not os.path.isfile(cf_file):
		raise IOError(('{:s} not exist').format(cf_file))

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

def _executed_cmds(args, im_pairs, out_dire, dm_file, dm_options, df_file, df_options, cf_file, cf_options):
	''''''
	t_time     = time.time()
	im_n       = len(im_pairs)
	mth_paths  = []
	flo_paths  = []
	
	for im_i in xrange(im_n):
		s_time   = time.time()

		im_pair  = im_pairs[im_i]
		
		im_path1 = im_pair[0]
		im_name  = os.path.basename(im_path1)
		
		im_name2 = os.path.splitext(im_name)[0]
		im_path2 = im_pair[1]

		print "main im_path:", im_path1
		print "auxi im_path:", im_path2
		
		mth_path = out_dire + im_name2 + "_match.txt"
		mth_paths.append(mth_path)

		of_path  = out_dire + im_name2 + ".png"
		
		flo_path = out_dire + im_name2 + ".flo"
		flo_paths.append(flo_path)

		dm_cmd   = "python %s %s %s %s -out %s" % (dm_file, im_path1, \
										im_path2, dm_options, mth_path)
		df_cmd   = "%s %s %s %s %s -match %s" % (df_file, im_path1, im_path2, \
										flo_path, df_options, mth_path)
		cf_cmd   = "%s %s %s" % (cf_file, flo_path, of_path)

		# optical flow images
		if not os.path.exists(of_path) or not os.path.isfile(of_path):
			# flo files and match file
			if not os.path.exists(flo_path) or not os.path.isfile(flo_path) or \
				 not os.path.exists(mth_path) or not os.path.isfile(mth_path):
				cmd = dm_cmd
				os.system(cmd)
			
			cmd = df_cmd
			os.system(cmd)

			if os.path.exists(flo_path) and os.path.isfile(flo_path):
				cmd = cf_cmd
				os.system(cmd)
			else:	
				raise IOError(('{:s} not exist').format(flo_path))
		else:
			pass
		print "\n---- Takes %s seconds for %s-th image: %s----\n" % (time.time() - s_time, im_i, im_path1)

	if args.rm_flo != 0:
		for flo_path in flo_paths:
			if os.path.exists(flo_path) and os.path.isfile(flo_path):
			  os.remove(flo_path)

	if args.rm_mth != 0:
		for mth_path in mth_paths:
			if os.path.exists(mth_path) and os.path.isfile(mth_path):
			  os.remove(mth_path) 

	t_time = time.time() - t_time
	print "\n---- Takes %s seconds for %s images -- (average time: %s)----\n" % (t_time, im_n, t_time / im_n)
	print "\nDone\n"

def flow_pipeline():
	''''''
	args 							= _parse_args()

	frame_n 					= args.frame_n

	im_first_path 	  = args.im_first_path.strip()

	out_dire 					= args.out_dire.strip()

	is_disp						= True if args.is_disp != 0 else False

	im_first_paths 		= _im_paths(im_first_path, is_disp)

	if args.im_second_path is None:
		print "\nPair from sequence\n"
		time.sleep(args.sleep_time)
		im_pairs 			  = _im_pairs_v1(im_first_paths, frame_n, is_disp)
	else:
		print "\nPair from directory\n"
		time.sleep(args.sleep_time)
		im_second_path  = args.im_second_path.strip()
		im_second_paths = _im_paths(im_second_path, is_disp)
		im_pairs        = _im_pairs_v2(im_first_paths, im_second_paths, is_disp)

	dm_file, df_file, cf_file = _executed_files(args)

	dm_options, df_options, cf_options = _executed_options(args)

	time.sleep(args.sleep_time)

	_executed_cmds(args, im_pairs, out_dire, dm_file, dm_options, df_file, df_options, cf_file, cf_options)


if __name__ == '__main__':
	''''''
	flow_pipeline()
