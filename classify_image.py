"""Predict aegypti versus albopictus for an image containing a mosquito."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--image', type=Path, required=True)
    parser.add_argument('-m', '--model', type=Path, default=Path('classifier/best.keras'))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image does not exist: {args.image}')
    if not args.model.is_file():
        parser.error(f'Trained classifier does not exist: {args.model}')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    classes = json.loads((args.model.parent / 'classes.json').read_text())
    with Image.open(args.image) as im:
        im = ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'), (224, 224), method=Image.Resampling.LANCZOS)
        batch = np.asarray(im, dtype=np.float32)[None, ...]
    model = tf.keras.models.load_model(args.model)
    scores = model(batch, training=False).numpy()[0]
    print('Predicted species:', classes[int(scores.argmax())])
    for name, score in zip(classes, scores):
        print(f'{name}: {score:.2%}')


if __name__ == '__main__':
    main()
