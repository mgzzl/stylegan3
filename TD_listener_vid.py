# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

from cgi import print_arguments
import copy
import os
import re
import sys
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm

import legacy

# From Spout
import os
import random
from Library.Spout import Spout
import socket
import pygame

def msg_to_bytes(msg):
    return msg.encode('utf-8')

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, num_seeds:int, spout,socket, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    # # create sender
    spout.createSender('output')
    print('Spout-Sender created')
    seeds = random.sample(range(0,9999999), int(num_seeds))
    START_SEED=420
    PKL_PATH = 'C:/Users/hello/Desktop/AI/stylegan3/pretrainedModels/'
    pkl_name = 'faces-1024'
    reply = 'data'
    # seeds = random.sample(range(0,9999999), 2)
    while True:

        seeds[0] = START_SEED
        if num_keyframes is None:
            if len(seeds) % (grid_w*grid_h) != 0:
                raise ValueError('Number of input seeds must be divisible by grid W*H')
            num_keyframes = len(seeds) // (grid_w*grid_h)

        all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
        for idx in range(num_keyframes*grid_h*grid_w):
            all_seeds[idx] = seeds[idx % len(seeds)]

        if shuffle_seed is not None:
            rng = np.random.RandomState(seed=shuffle_seed)
            rng.shuffle(all_seeds)

        zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
        ws = G.mapping(z=zs, c=None, truncation_psi=psi)
        _ = G.synthesis(ws[:1]) # warm up
        ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

        # Interpolation.
        grid = []
        for yi in range(grid_h):
            row = []
            for xi in range(grid_w):
                x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
                y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
                interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
                row.append(interp)
            grid.append(row)

        # Render video.
        # video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
        for frame_idx in tqdm(range(num_keyframes * w_frames)):
            
            # for rec messages
            try:
                d = socket.recvfrom(1024)
                data = d[0]
                addr = d[1]
                reply = data.decode('utf-8')
                reply = reply.split('_')
                print(f'\nNew message recieved:\n name: {reply[0]}\n value: {reply[1]}')
            except:
                pass
                # print('No new message recieved')

            
            if reply[0] == 'menuIndex':
                reply[1] = int(reply[1].replace(" ", ""))
                if reply[1] == 0:
                    # network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl'
                    pass
                if reply[1] == 1:
                    pkl_name = '70s-scifi-1024'
                if reply[1] == 2:
                    pkl_name = 'abstractArt-1024'
                if reply[1] == 3:
                    pkl_name = 'drawing-1024'
                if reply[1] == 4:
                    pkl_name = 'textures-1024'
                if reply[1] == 5:
                    pkl_name = 'VisionaryArt-1024'
                if reply[1] == 6:
                    pkl_name = 'wikiArt-1024'
                if reply[1] == 7:
                    pkl_name = 'landscapes-256'
                if reply[1] == 8:
                    pkl_name = 'PottwalGAN-512'

                print(f' Network-Name: {pkl_name}')
                network_pkl = PKL_PATH + pkl_name + '.pkl'
                print('Loading networks from "%s"...' % network_pkl)
                device = torch.device('cuda')
                with dnnlib.util.open_url(network_pkl) as f:
                    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
                    print('Network sucessfully loaded!')
                reply = 'default' # set reply to default thus the if-statement is not called every iteration
                
                all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
                for idx in range(num_keyframes*grid_h*grid_w):
                    all_seeds[idx] = seeds[idx % len(seeds)]

                if shuffle_seed is not None:
                    rng = np.random.RandomState(seed=shuffle_seed)
                    rng.shuffle(all_seeds)

                zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
                ws = G.mapping(z=zs, c=None, truncation_psi=psi)
                _ = G.synthesis(ws[:1]) # warm up
                ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

                # Interpolation.
                grid = []
                for yi in range(grid_h):
                    row = []
                    for xi in range(grid_w):
                        x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
                        y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
                        interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
                        row.append(interp)
                    grid.append(row)
                reply = 'default'

            if reply[0] == 'exit':
                pygame.quit()
                socket.close()
                quit()

            if reply[0] == 'w-frames':
                reply[1] = int(reply[1].replace(" ", ""))
                if reply[1] == 0:
                    w_frames = 30
                else:
                    w_frames = reply[1]*60                
                reply = 'default' # set reply to default thus the if-statement is not called every iteration
            
            if reply[0] == 'num-seeds':
                print(f'orgiginal length of seeds: {len(seeds)}')
                tmp_num_seeds = int(reply[1].replace(" ", ""))
                print(f'reply num_seeds: {tmp_num_seeds}')
                if tmp_num_seeds < int(num_seeds):
                    num2rm = int(num_seeds) - tmp_num_seeds
                    seeds = seeds[:-num2rm]
                    print(f'updated length of seeds: {len(seeds)}')
                    reply = 'default' # set reply to default thus the if-statement is not called every iteration

                    # num_keyframes = len(seeds) // (grid_w*grid_h)
                else:
                    num2add = tmp_num_seeds - int(num_seeds)
                    print(tmp_num_seeds)
                    tmp_seeds = random.sample(range(0,9999999), num2add)
                    print(len(tmp_seeds))
                    seeds = seeds + tmp_seeds
                    print(f'updated length of seeds: {len(seeds)}')
                    reply = 'default' # set reply to default thus the if-statement is not called every iteration
    
                    # num_keyframes = len(seeds) // (grid_w*grid_h)
                num_seeds = tmp_num_seeds
                num_keyframes = len(seeds) // (grid_w*grid_h)
                all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
                for idx in range(num_keyframes*grid_h*grid_w):
                    all_seeds[idx] = seeds[idx % len(seeds)]

                if shuffle_seed is not None:
                    rng = np.random.RandomState(seed=shuffle_seed)
                    rng.shuffle(all_seeds)

                zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
                ws = G.mapping(z=zs, c=None, truncation_psi=psi)
                _ = G.synthesis(ws[:1]) # warm up
                ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

                # Interpolation.
                grid = []
                for yi in range(grid_h):
                    row = []
                    for xi in range(grid_w):
                        x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
                        y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
                        interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
                        row.append(interp)
                    grid.append(row)
                
            
            imgs = []
            for yi in range(grid_h):
                for xi in range(grid_w):
                    interp = grid[yi][xi]
                    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                    img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]            
                    imgs.append(img)
            spout.send(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
                   
        seeds = random.sample(range(0,9999999), int(num_seeds))
        spout.check()

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl', required=True)
@click.option('--num_seeds', help='Number of random seeds to generate', default='200', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=150)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--output', help='Output .mp4 filename', type=str, default='./video.mp4', metavar='FILE')
def generate_images(
    network_pkl: str,
    num_seeds: int,
    shuffle_seed: Optional[int],
    truncation_psi: float,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    output: str
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """
    UDP_IP = "127.0.0.1"
    UDP_PORT = 7010
    REC_PORT = 7000
    MESSAGE = "LISTENING"
    print("UDP target IP:", UDP_IP)
    print("UDP target port:", UDP_PORT)
    print("Message:", MESSAGE)
    try:
        sock = socket.socket(socket.AF_INET,
                        socket.SOCK_DGRAM)
        sock.setblocking(0)
        sock.settimeout(0.0)
        print(f'Setting up UDP on ip={UDP_IP} and port={UDP_PORT}')
    except:
        print("Failed to create socket")
        sys.exit()

    try:
        sock.bind(('', REC_PORT))
        print(f'Listening in ip={UDP_IP} and port={REC_PORT}')
    except:
        print('Bind failed!')
        sys.exit()


    sock.sendto(msg_to_bytes(MESSAGE), (UDP_IP, UDP_PORT))

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        print('Network sucessfully loaded!')

    # # create spout object
    spout = Spout(silent = True, width=1024, height=1024)
    # # create receiver
    # spout.createReceiver('input_try')
    
    gen_interp_video(G=G, mp4=output, bitrate='12M', grid_dims=grid, spout=spout, socket=sock, num_keyframes=num_keyframes, w_frames=w_frames, num_seeds=num_seeds, shuffle_seed=shuffle_seed, psi=truncation_psi)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
