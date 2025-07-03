import os
from models.IK import remove_sliding
from os.path import join as pjoin
from parser.base import try_mkdir
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./results/bvh/')
    parser.add_argument('--output_dir', type=str, default='./results/remove_sliding/')
    parser.add_argument('--window_length', type=int, default=3)
    return parser.parse_args()

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            if f.endswith('.bvh'):
                fullname = os.path.join(root, f)
                file_path.append(fullname)
    return file_path
if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    window_length = args.window_length
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = findAllFile(input_dir)
    
    for input_file in input_files:
        output_file = input_file.replace(input_dir, output_dir).replace('predict.bvh', 'remove_sliding.bvh')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        remove_sliding(input_file, input_file, output_file, None, window_length=window_length)

