# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating a dataset of persons with images using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from tqdm import tqdm


def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    # Parameters
    rnd = np.random.RandomState(1)
    similarity = 505  # set similarity amount within one 'person'
    n_persons = 10
    n_img_per_person = 10

    os.makedirs(config.result_dir, exist_ok=True)
    for person_id in tqdm(range(n_persons)):
        # Set base latent vector for this person.
        latent_base = rnd.randn(1, Gs.input_shape[1])
        for img_id in range(n_img_per_person):
            # Create variation on base latent vector to create various images.
            latent = rnd.randn(1, Gs.input_shape[1])
            latent[0][:similarity] = latent_base[0][:similarity]

            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)

            # Save image.
            png_filename = os.path.join(config.result_dir, 'person_{}-img_{}.png'.format(person_id, img_id))
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


if __name__ == "__main__":
    main()
