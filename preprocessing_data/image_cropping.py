import logging
import logging.config
from os import listdir
from os.path import isfile, join
import argparse

from image_cropper import ImageCrop


# Note that this will alter the current training image set folder

def process_all_images():
    files = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]
    for file in files:
        logger.debug('Processing image: ' + file)
        ImageCrop(source_folder, destination_folder, file).remove_black_pixels()


if __name__ == '__main__':
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
