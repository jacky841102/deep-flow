This code is used for computing optical flow between two images.

This code is mentioned only for scientific or personal use. 
Please contact `DeepFlow`, `DeepMatching` and `ColorFlow` for commercial use.

DeepFlow:
	http://lear.inrialpes.fr/src/deepflow/

DeepMatching:
	http://lear.inrialpes.fr/src/deepmatching/

ColorFlow: 
	http://vision.middlebury.edu/flow/data/

pipeline:
	0: prepare the images pairs
	1: compute matches using `DeepMatching`
	2: compute flo file using `DeepFlow`
	3: compute optical flow using `ColorFlow`


### Installation ###
1 download the code, put them into some directory
2 compiling `deep_matching`, `deep_flow2`, and `color_flow`
	please refer to `README` of each of them for more details.

The program was only tested under a 64-bit Linux distribution (Ubuntu 14.04).

### Example ###
	cd optical_flow
  sh flow_pipeline.sh

or see `cmd.example` for more examples


