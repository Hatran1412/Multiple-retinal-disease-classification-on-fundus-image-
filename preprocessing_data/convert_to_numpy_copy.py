from absl import app
import logging
import logging.config
import time
import csv
import cv2
import os
import numpy as np
import glob



class NumpyDataGenerator:
    def __init__(self, training_path, validation_path,testing_path, csv_training_path, csv_validation_path,csv_testing_path, augmented_path, csv_augmented_file):
        self.training_path = training_path
        self.validation_path= validation_path
        self.testing_path = validation_path
        self.csv_training_path = csv_training_path
        self.csv_validation_path = csv_validation_path
        self.csv_testing_path= csv_testing_path
        self.total_records_training = 0
        self.total_records_testing = 0
        self.total_records_augmented=0
        self.total_records_validation =0 
        self.csv_augmented_path = csv_augmented_file
        self.augmented_path = augmented_path
        self.logger = logging.getLogger('rfmid')
 
        
        
    def npy_training_files(self, file_name_training, file_name_training_labels):
        training = []
        training_labels = []

        self.logger.debug("Opening CSV file")
        with open(self.csv_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_training = 0
            for row in csv_reader:
                file_name = row[0]
                N=row[1]
                DR = row[2]
                ARMD = row[3]
                MH = row[4]
                DN = row[5]
                MYA = row[6]
                TSLN = row[7]
                ODC = row[8]
                O = row[9]
           
                # just discard the first row
                if file_name != "ID":
                    self.logger.debug("Processing image: " + file_name)
                    # load first the image from the folder
                    eye_image = os.path.join(self.training_path, file_name + '.png')
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    training.append(image)
                    training_labels.append([N,DR,ARMD, MH,DN,MYA,TSLN,ODC, O])
                    self.total_records_training = self.total_records_training + 1

        training = np.array(training, dtype='uint8')
        training_labels = np.array(training_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        training = np.reshape(training, [training.shape[0], training.shape[1], training.shape[2], training.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_training, training)
        self.logger.debug("Saving NPY File: " + file_name_training)
        np.save(file_name_training_labels, training_labels)
        self.logger.debug("Saving NPY File: " + file_name_training_labels)
        self.logger.debug("Closing CSV file")

        
    def npy_validation_files(self, file_name_validation, file_name_validation_labels):
        validation = []
        validation_labels = []

        self.logger.debug("Opening CSV file")
        with open(self.csv_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_validation = 0
            for row in csv_reader:
                file_name = row[0]
                N=row[1]
                DR = row[2]
                ARMD = row[3]
                MH = row[4]
                DN = row[5]
                MYA = row[6]
                TSLN = row[7]
                ODC = row[8]
                O = row[9]
           
                # just discard the first row
                if file_name != "ID":
                    self.logger.debug("Processing image: " + file_name)
                    # load first the image from the folder
                    eye_image = os.path.join(self.training_path, file_name + '.png')
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    validation.append(image)
                    validation_labels.append([N,DR,ARMD, MH,DN,MYA,TSLN,ODC, O])
                    self.total_records_validation = self.total_records_validation + 1

        validation = np.array(validation, dtype='uint8')
        validation_labels = np.array(validation_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        validation = np.reshape(validation, [validation.shape[0], validation.shape[1], validation.shape[2], validation.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_validation, validation)
        self.logger.debug("Saving NPY File: " + file_name_training)
        np.save(file_name_validation_labels, validation_labels)
        self.logger.debug("Saving NPY File: " + file_name_validation_labels)
        self.logger.debug("Closing CSV file")


    def npy_testing_files(self, file_name_testing, file_name_testing_labels):
        testing = []
        testing_labels = []

        self.logger.debug("Opening CSV file")
        with open(self.csv_testing_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_testing = 0
            for row in csv_reader:
                file_name = row[0]
                N=row[1]
                DR = row[2]
                ARMD = row[3]
                MH = row[4]
                DN = row[5]
                MYA = row[6]
                TSLN = row[7]
                ODC = row[8]
                O = row[9]
                # just discard the first row
                if file_name != "ID":
                    self.logger.debug("Processing image: " + file_name + ".png")
                    # load first the image from the folder
                    eye_image = os.path.join(self.testing_path, file_name + '.png')
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    testing.append(image)
                    testing_labels.append([N,DR,ARMD, MH,DN,MYA,TSLN,ODC, O])
                    self.total_records_testing = self.total_records_testing + 1

        testing = np.array(testing, dtype='uint8')
        testing_labels = np.array(testing_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        testing = np.reshape(testing, [testing.shape[0], testing.shape[1], testing.shape[2], testing.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_testing, testing)
        self.logger.debug("Saving NPY File: " + file_name_testing)
        np.save(file_name_testing_labels, testing_labels)
        self.logger.debug("Saving NPY File: " + file_name_testing_labels)
        self.logger.debug("Closing CSV file")

    
    def npy_augmented_files(self, file_name_augmented, file_name_augmented_labels):
        augmented = []
        augmented_labels = []

        self.logger.debug("Opening CSV file")
        with open(self.csv_augmented_path) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            self.total_records_testing = 0
            for row in csv_reader:
                file_name = row[0]
                N=row[1]
                DR = row[2]
                ARMD = row[3]
                MH = row[4]
                DN = row[5]
                MYA = row[6]
                TSLN = row[7]
                ODC = row[8]
                O = row[9]
                # just discard the first row
                if file_name != "ID":
                    self.logger.debug("Processing image: " + file_name + ".png")
                    # load first the image from the folder
                    eye_image = os.path.join(self.augmented_path, file_name )
                    image = cv2.imread(eye_image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    augmented.append(image)
                    augmented_labels.append([N,DR,ARMD, MH,DN,MYA,TSLN,ODC, O])
                    self.total_records_augmented = self.total_records_augmented + 1

        augmented = np.array(augmented, dtype='uint8')
        augmented_labels = np.array(augmented_labels, dtype='uint8')
        # convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
        # for example (6069 * 28 * 28 * 3)-> (6069 x 2352) (14,274,288)
        augmented = np.reshape(augmented, [augmented.shape[0], augmented.shape[1], augmented.shape[2], augmented.shape[3]])

        # save numpy array as .npy formats
        np.save(file_name_augmented, augmented)
        self.logger.debug("Saving NPY File: " + file_name_augmented)
        np.save(file_name_augmented_labels, augmented_labels)
        self.logger.debug("Saving NPY File: " + file_name_augmented_labels)
        self.logger.debug("Closing CSV file")



def main(argv):
    start = time.time()
    image_width = 224
    training_path = r'dataset\data_preprocessing\training_set\training_resized'
    validation_path = r'dataset\data_preprocessing\validation_Set\Validation_resized'
    testing_path = r'dataset\data_preprocessing\testing_Set\testing_resized'
    augmented_path = r'dataset\data_preprocessing\training_set\training_augmented'
    csv_training_file = r'dataset\data_preprocessing\training_set\Training_Labels.csv'
    csv_augmented_file = r'\.ground_truth\augmented.csv'
    csv_validation_file = r'dataset\data_preprocessing\valuation_set\Validation_Labels.csv'
    csv_testing_file = r'dataset\data_preprocessing\testing_set\Testing_Labels.csv'
    logger.debug('Generating npy files')
    generator = NumpyDataGenerator(training_path, testing_path,validation_path, csv_training_file, csv_testing_file, csv_validation_file, augmented_path,
                                   csv_augmented_file)

    # Generate testing file
    generator.npy_validation_files('validation' , 'validation_labels')
    generator.npy_training_files('training' , 'training_labels')
    generator.npy_augmented_files('augmented', 'augmented_labels')
    generator.npy_testing_files('testing' , 'testing_labels')

   
    end = time.time()
    logger.debug('Training Records ' + str(generator.total_records_training))
    logger.debug('Testing Records ' + str(generator.total_records_testing))
    logger.debug('Augmented Records ' + str(generator.total_records_augmented))
    logger.debug('Validation Records ' + str(generator.total_records_validation))
    logger.debug('All Done in ' + str(end - start) + ' seconds')


if __name__ == '__main__':
    # create logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('rfmid')
    app.run(main)
