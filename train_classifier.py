"""Train an Aedes species classifier with filename-grouped evaluation splits."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import re

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

CLASSES = ['aegypti', 'albopictus']


def split_images(root, seed):
    groups = {}
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        match = re.fullmatch(r'(aegypti|albopictus)(\d+)([a-z])', path.stem)
        if not match:
            continue  # Exclude the three files explicitly named "Copy".
        species, specimen, _ = match.groups()
        groups.setdefault((species, int(specimen)), []).append(path)
    splits = {name: [] for name in ('train', 'validation', 'test')}
    rng = random.Random(seed)
    for species in CLASSES:
        keys = sorted(key for key in groups if key[0] == species)
        if len(keys) < 10:
            raise ValueError('Need at least 10 filename groups per species.')
        rng.shuffle(keys)
        n_test = max(1, round(len(keys) * .15))
        n_val = max(1, round(len(keys) * .15))
        for index, key in enumerate(keys):
            split = 'test' if index < n_test else 'validation' if index < n_test + n_val else 'train'
            splits[split].extend((p, CLASSES.index(species), f'{species}{key[1]}') for p in groups[key])
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('classifier'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Epochs and batch size must be positive.')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    splits = split_images(args.data.resolve(), args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'resized'
    cache.mkdir(exist_ok=True)

    def resize(row):
        path, label, group = row
        relative = path.relative_to(args.data.resolve())
        target = cache / relative.with_suffix('.jpg')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im = ImageOps.pad(im, (224, 224), method=Image.Resampling.LANCZOS)
                im.save(target, quality=90)
        return {'source': str(path), 'image': str(target.resolve()), 'label': label, 'group': group}

    manifest = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, rows in splits.items():
            manifest[name] = list(pool.map(resize, rows))
            print(name, dict(Counter(CLASSES[r['label']] for r in manifest[name])), flush=True)
    (args.output / 'splits.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'classes.json').write_text(json.dumps(CLASSES))

    def dataset(rows, training=False):
        paths = [r['image'] for r in rows]
        labels = [r['label'] for r in rows]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(rows), seed=args.seed)

        def load(path, label):
            im = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            im = tf.ensure_shape(im, (224, 224, 3))
            return tf.cast(im, tf.float32), label

        ds = ds.map(load, num_parallel_calls=2).batch(args.batch_size)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return ds.with_options(options).prefetch(1)

    weights_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    pretrained = tf.keras.utils.get_file(
        weights_name,
        'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/' + weights_name,
        cache_dir=str(args.output.resolve()), cache_subdir='pretrained')
    backbone = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=pretrained)
    backbone.trainable = False
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(.25)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = args.output / 'best.keras'
    weights = args.output / 'best.weights.h5'
    history = model.fit(dataset(manifest['train'], True), validation_data=dataset(manifest['validation']),
                        epochs=args.epochs, verbose=2, callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(str(weights), monitor='val_loss', save_best_only=True, save_weights_only=True),
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
    model.load_weights(weights)
    model.save(checkpoint)
    test = dataset(manifest['test'])
    loss, accuracy = model.evaluate(test, verbose=0)
    predicted = model.predict(test, verbose=0).argmax(axis=1)
    actual = np.array([r['label'] for r in manifest['test']])
    confusion = tf.math.confusion_matrix(actual, predicted, num_classes=2).numpy().tolist()
    metrics = {'test_loss': float(loss), 'test_accuracy': float(accuracy), 'classes': CLASSES,
               'confusion_matrix_true_rows_predicted_columns': confusion,
               'split_images': {k: len(v) for k, v in manifest.items()},
               'seed': args.seed, 'history': history.history,
               'split_method': '70/15/15 by species and numeric filename ID; all views and rotations stay together'}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print('Saved classifier:', checkpoint)


if __name__ == '__main__':
    main()
