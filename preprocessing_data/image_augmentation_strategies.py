import csv
import os
import cv2
from image_augmentation import ImageTreatment


class DataAugmentationStrategy:
    def __init__(self, image_size, file_name):
        self.base_image = file_name
        self.treatment = ImageTreatment(image_size)
        self.file_path = r'dataset\data_preprocessing\training_set\training_resized' 
        self.saving_path = r'dataset\data+prprocessing\training_set\training_augmented'
        self.file_id = file_name.replace('.png', '')

    def save_image(self, original_vector, image, sample):
        central = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        file = self.file_id + '_'+str(sample)+'.png'
        file_name = os.path.join(self.saving_path, file)
        exists = os.path.isfile(file_name)
        if exists:
            print("duplicate file found: " + file_name)

        status = cv2.imwrite(file_name, central)

        with open(r'.\ground_truth\augmented.csv', 'a', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow([file, original_vector[1],original_vector[2], original_vector[3], original_vector[4],original_vector[5],
             original_vector[6], original_vector[7], original_vector[8], original_vector[9]
            ])

        #print(file_name + " written to file-system : ", status)

    def generate_images(self, number_samples, original_vector, weights):
        eye_image = os.path.join(self.file_path, self.base_image)
        image = cv2.imread(eye_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image
        saved = 0

        # For any repeating elements, just give the other output
        # We are only expecting up to 3 repetitions
        if weights == 20:
            original_image = self.treatment.rot90(original_image, 2)
        if weights == 400:
            original_image = self.treatment.rot90(original_image, 3)
        if weights > 401:
            print(str(self.file_id) + ' samples:' + str(number_samples))
            raise ValueError('this cannot happen')
            

        # for the sample type 6, just generate 1 image and leave the method
        if number_samples == 6: 
            central = self.treatment.rot90(original_image, 1)
            self.save_image(original_vector, central, weights+6)
            saved = saved +1
            return saved

        if number_samples > 0:
            central = self.treatment.rescale_intensity(original_image)
            self.save_image(original_vector, central, weights+0)
            saved = saved + 1

        if number_samples > 1:
            central = self.treatment.contrast(original_image, 2)
            self.save_image(original_vector, central, weights+1)
            saved = saved + 1

        if number_samples > 2:
            central = self.treatment.saturation(original_image, 0.5)
            self.save_image(original_vector, central, weights+2)
            saved = saved + 1

        if number_samples > 3:
            central = self.treatment.gamma(original_image, 0.5)
            self.save_image(original_vector, central, weights+3)
            saved = saved + 1

        if number_samples > 4:
            central = self.treatment.hue(original_image, 0.2)
            self.save_image(original_vector, central, weights+4)
            saved = saved + 1
        
        return saved


        