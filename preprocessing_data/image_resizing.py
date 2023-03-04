
import logging
import logging.config
from os import listdir
from os.path import isfile, join
import argparse
from image_resizer import ImageResizer


# This default job to 224px images, will shrink the dataset from 1,439,776,768 bytes
# to 116,813,824 bytes 91.8% size reduction

def process_all_images():
    files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
    for file in files:
        logger.debug('Processing image: ' + file)
        ImageResizer(image_width, quality, source_folder, destination_folder, file, keep_aspect_ratio).run()


if __name__ == '__main__':
    # Set the base width of the image to 200 pixels
    image_width = 224
    keep_aspect_ratio = False
    # set the quality of the resultant jpeg to 100%
    quality = 100
    parser= argparse.ArgumentParser()
    parser.add_argument('--source_folder', required=True, typr=str)
    parser.add_argument('--destination_folder', required=True, typr=str)
    args=parser.parse_args()
    source_folder=args.source_folder
    destination_folder=args.detination_folder
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('rfmid')
    process_all_images()
