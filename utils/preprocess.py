"""
preprocess.py
Builds train/validation ImageDataGenerators from the dataset folder.

Expected dataset layout:

dataset/
├── Broken Road/
│   ├── img1.jpg
│   └── ...
└── Not Broken Road/
    ├── img1.jpg
    └── ...
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_generators(dataset_dir, image_size=224, batch_size=32,
                    augmentation=True, validation_split=0.2):
    if augmentation:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=validation_split
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=validation_split
        )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen
