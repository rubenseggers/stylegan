# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for producing a dataset using StyleGAN."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from tqdm import tqdm
import random


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)


def mix_dlatents(source=None, target=None, layers=range(18)):
    # style_ranges = range(0, 4) range(4, 8) range(8, 18)
    source[:, layers] = target[layers]
    return source


# noinspection PyArgumentList
def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Create output dir.
    os.makedirs(config.result_dir, exist_ok=True)

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    # Parameters.
    n_users = 20
    n_per_user = 10
    random.seed(420)

    # Initial set user dlatents.
    src_seeds = random.sample(range(10000), n_users)
    A_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    A_dlatents = Gs.components.mapping.run(A_latents, None)  # [user, layer, component]

    # Attribute target dlatents
    src_seeds = [1733, 1614]
    B_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    B_dlatents = Gs.components.mapping.run(B_latents, None)  # [user, layer, component]

    C_1 = mix_dlatents(source=A_dlatents[int(n_users / 2):], target=B_dlatents[0], layers=range(0, 4))
    C_2 = mix_dlatents(source=A_dlatents[:int(n_users / 2)], target=B_dlatents[1], layers=range(0, 4))

    # print(C_1.shape)
    user_base_dlatents = np.concatenate((C_1, C_2))
    # print(user_base_dlatents.shape)
    # exit()

    for person_id in tqdm(range(n_users)):
        # Create variation.
        user_all_dlatents = np.stack([user_base_dlatents[person_id]] * n_per_user)  # [variation, layer, component]
        from_layer = 8
        to_layer = 18
        src_seeds = random.sample(range(10000), n_per_user)
        D_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        D_dlatents = Gs.components.mapping.run(D_latents, None)  # [user, layer, component]
        user_all_dlatents[:, range(from_layer, to_layer)] = D_dlatents[:, range(from_layer, to_layer)]

        # Generate images.
        user_images = Gs.components.synthesis.run(user_all_dlatents, randomize_noise=False, **synthesis_kwargs)

        # Save images.
        for img_id in range(n_per_user):
            png_filename = os.path.join(config.result_dir, 'person_{}-img_{}.png'.format(person_id, img_id))
            # PIL.Image.fromarray(user_images[person_id], 'RGB').resize((64, 64)).save(png_filename)
            PIL.Image.fromarray(user_images[img_id], 'RGB').save(png_filename)


if __name__ == "__main__":
    main()
