# Natural Image Contour Tracing

This repository creates stimuli for the task of contour tracing in natural images. 
Each input image is a natural image in which 2 easily identifiable markers (red and blue concentric circles) have been added onto contours. 
In some images the markers lie on the same contour, while in others they are on different contours.
For each image a binary label is provided as to whether the two markers are connected via a smooth contour. 

## Data Set Creation
1. Download Barcelona Images for Perceptual Edge Detection (BIPED) BIPED data set from https://github.com/xavysp/MBIPED
1. Once downloaded, follow the instructions to augment the training data and get the full training dataset.
1. In generate_pathfinder_dataset.py, set the location of the BIPED_DATASET_DIR to the edges subdirectory of the downloaded BIPED dataset.
1. Run generate_pathfinder_dataset.py
1. This will create a new dataset under ./data/pathfinder_natural_images_test containing 50,000 train and 5000 test images.
1. The script loops twice, first creating the train followed by the test datasets.
1. Input images are stored under ./data/pathfinder_natural_images_test/<train/test>/images directiory and classification labels in  ./data/pathfinder_natural_images_test/<train/test>/classification_labels.txt
1. In addition to this, several debug information and dataset metadata files and folders are also created. For more details refer to the create_dataset function. 

## Pytorch Data Loaders for Training
1. After the pathfinder dataset is created, the Pytorch dataset (PathfinderNaturalImages) defined in dataset_pathfinder.py can be used to train models.
1. To randomly fragment contours, random occlusion bubbles are added as a preprocessing step during training (for an example run dataset_pathfinder.py ) 
1. The occlusion bubble adding input transform (PunctureImage) is defined in utils.py
1. To add bubbles at particular locations, the same puncturing function is explicitly called with specific locations.