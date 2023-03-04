import csv
import logging.config
import os
from absl import app

from image_augmentation_strategies import DataAugmentationStrategy
from ground_truth_files import GroundTruthFiles


def write_header():
    with open(r'.\ground_truth\augmented.csv', 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['file_name','N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"])    
        return file_writer


def process_files(images, cache, files):
    total = 0
    for strategy in range(len(images)):
        images_to_process = images[strategy][0]
        samples_per_image = images[strategy][1]
        for image_index in range(images_to_process):
            image_vector = files[image_index]
            file_name = image_vector[0]

            # Only check during the first strategy
            if strategy == 0:
                if file_name not in cache:
                    cache[file_name] = 1
                else:
                    cache[file_name] = cache[file_name] * 20

            print('Processing: ' + file_name)
            augment = DataAugmentationStrategy( image_size,file_name+'.png')
            count = augment.generate_images(samples_per_image, image_vector, cache[file_name])
            total = total + count
    return total


def main(argv):
    # load the ground truth file
    files = GroundTruthFiles()
    files.populate_vectors(csv_path) 
    print('files record count order by size ASC')
    print('N ' + str(len(files.N)))
    print('DR ' + str(len(files.DR)))
    print('ARMD ' + str(len(files.ARMD)))
    print('MH ' + str(len(files.MH)))
    print('DN ' + str(len(files.DN)))
    print('MYA ' + str(len(files.MYA)))
    print('TSLN ' + str(len(files.TSLN)))
    print('ODC ' + str(len(files.ODC)))
    print('O ' + str(len(files.O)))


 
    images_N = [[len(files.N), 1], [20, 6]]
    images_DR = [[len(files.DR), 1], [50, 6]]
    images_ARMD = [[len(files.ARMD), 3], [100, 6]]
    images_MH = [[len(files.MH), 1], [100, 6]]
    images_DN = [[len(files.DN), 3], [20, 6]]
    images_MYA = [[len(files.MYA), 3], [100, 6]]
    images_TSLN = [[len(files.TSLN), 2], [40, 6]]
    images_ODC = [[len(files.ODC), 1], [100, 6]]
    images_O = [[len(files.O), 1], [10, 6]]


    
    
    # Delete previous file
    exists = os.path.isfile(r'\ground_truth\augmented.csv')
    if exists:
        os.remove(r'D:\THESIS\rfmid\ground_truth\augmented.csv')

    write_header()

    images_processed = {}

    total_N = process_files(images_N, images_processed, files.N)
    total_DR = process_files(images_DR, images_processed, files.DR)
    total_ARMD = process_files(images_ARMD, images_processed, files.ARMD)
    total_MH = process_files(images_MH, images_processed, files.MH)
    total_DN = process_files(images_DN, images_processed, files.DN)
    total_MYA = process_files(images_MYA, images_processed, files.MYA)
    total_TSLN = process_files(images_TSLN, images_processed, files.TSLN)
    total_ODC = process_files(images_ODC, images_processed, files.ODC)
    total_others = process_files(images_O, images_processed, files.O)
  

    print("total generated N: " + str(total_N))
    print("total generated DR: " + str(total_DR))
    print("total generated ARMD: " + str(total_ARMD))
    print("total generated MH: " + str(total_MH))
    print("total generated DN: " + str(total_DN))
    print("total generated MYA: " + str(total_MYA))
    print("total generated TSLN: " + str(total_TSLN))
    print("total generated ODC: " + str(total_ODC))
    print("total generated others: " + str(total_others))
    print("total",str(total_ARMD+total_DN+ total_DR+ total_TSLN+total_others+total_ODC+ total_N+ total_MH+ total_MYA) )

if __name__ == '__main__':
    # create logger
    logging.config.fileConfig(r'logging.conf')
    logger = logging.getLogger('rfmid')
    image_size = 224
    csv_path = r'dataset\data_preprocessing\Training_Set\Training_Labels.csv'
    app.run(main)