#!/usr/bin/env python3
"""Inspect the active interpreter without changing its environment."""

import argparse
from importlib import metadata
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Require CUDA and Triton')
    args = parser.parse_args()
    print('python=' + sys.version.split()[0])
    missing = []
    for package in ('torch', 'transformers', 'pytest', 'pybind11', 'pyzmq'):
        try:
            print(package + '=' + metadata.version(package))
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        parser.exit(1, 'Missing packages: ' + ', '.join(missing) + '\n')
    import torch
    print('torch_cuda=' + str(torch.version.cuda))
    print('cuda_available=' + str(torch.cuda.is_available()))
    if args.gpu:
        if not torch.cuda.is_available():
            parser.exit(1, 'CUDA is required for the GPU test.\n')
        try:
            import triton
        except ImportError:
            parser.exit(1, 'Triton is required for the GPU test.\n')
        print('triton=' + triton.__version__)
        for index in range(torch.cuda.device_count()):
            print(f'gpu[{index}]=' + torch.cuda.get_device_name(index))
    print('SPECSTREAM_ENVIRONMENT=PASS')


if __name__ == '__main__':
    main()
