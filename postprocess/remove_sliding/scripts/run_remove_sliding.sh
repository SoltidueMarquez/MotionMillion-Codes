python smpl2bvh.py --gender NEUTRAL --poses ./demo/272rpr/ --output ./demo/bvh/ --fps 30 --is_folder
python remove_sliding.py --input_dir ./demo/bvh/ --output_dir ./demo/remove_slide/ --window_length 5
