

#Instructions for running 
'''
# Train a new model starting from pre-trained COCO weights
python3 train.py train --dataset=/path/to/doc/dataset --weights=coco

# Resume training a model that you had trained earlier
python3 train.py train --dataset=/path/to/doc/dataset --weights=last

# Train a new model starting from ImageNet weights
python3 train.py train --dataset=/path/to/doc/dataset --weights=imagenet
'''


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage.transform import resize
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from tqdm import tqdm

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
# No. of images in train subset: 47958
# No. of images in val subset: 11245

class Config(Config):
    """Configuration for training on the doc  dataset.
    Derives from the base Config class and overrides some values.
    Go to config.py in mrcnn if you want to change other hyperparameters 
    """
    NAME = "object"
    GPU_COUNT = 3
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 6
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500   # steps_per_epoch * batch_size = number_of_rows_in_train_data
    VALIDATION_STEPS = 80
    USE_MINI_MASK = False
    WEIGHT_DECAY = 0.001
    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################

class Dataset(utils.Dataset):
    def load_data(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "TXT")
        self.add_class("object", 2, "TTL")
        self.add_class("object", 3, "LST")
        self.add_class("object", 4, "TAB")
        self.add_class("object", 5, "FIG")
        
        # categories = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
        
        # Train or validation dataset?
        assert subset in ["train", "val","test"]    
        subset_dir = os.path.join(dataset_dir, subset)
    
        json_obj = json.load(open(os.path.join(dataset_dir, subset+'.json'), 'r'))
        
        images = {}
        for image in json_obj['images']:
            images[image['id']] = {'file_name': os.path.join(subset_dir, image['file_name']), 
                                   'annotations': [],
                                   'height': image['height'], 
                                   'width': image['width']
                                   }
        for ann in json_obj['annotations']:
            images[ann['image_id']]['annotations'].append(ann)

        image_count = 0
        for _, (id, image) in enumerate(tqdm(images.items())):
            if not os.path.isfile(image['file_name']):
                continue
            
            polygons = []
            # rectangles = []
            class_ids = []
            for annotation in image['annotations']:
                polygons.append(annotation['segmentation'][0])
                # rectangles.append(annotation['bbox']) # x, y, w, h
                class_ids.append(annotation['category_id'])
            
            self.add_image(
             "object",
             image_id=id,  # use file name as a unique image id
             path=image['file_name'],
             width=image['width'], 
             height=image['height'],
             polygons=polygons,
             num_ids=class_ids)
            
            image_count += 1
            
        print('No. of images in ' + subset +' subset: ' + str(image_count))

    
    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros( [info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, poly in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            try:
                rr, cc = skimage.draw.polygon(poly[1::2], poly[0::2])
            except Exception:
                continue
            try:
                mask[rr, cc, i] = 1
            except Exception as e:
                print(e)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        
        return mask.astype(np.bool), num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # epochs = [30, 50, 65]
    epochs = [1, 1, 1]
    # Training dataset.
    dataset_train = Dataset()
    dataset_train.load_data(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Dataset()
    dataset_val.load_data(args.dataset, "val")
    dataset_val.prepare()

    # Training - Stage 1
    # Finetune task specific network heads
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=epochs[0],
                        layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs[1],
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=epochs[2],
                layers='all')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect regions in documents.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/doc/dataset/",
                        help='Directory of the doc dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = Config()
    else:
        class InferenceConfig(Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' to start training the model.".format(args.command))
