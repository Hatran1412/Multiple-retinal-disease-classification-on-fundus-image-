import csv
import os


class Prediction:
    def __init__(self, prediction, num_images_test, folder = ""):
        self.prediction = prediction
        self.num_images_test = num_images_test
        self.folder = folder

    def save(self):
        """Generate a CSV that contains the output of all the classes.
        Args:
          No arguments are required.
        Returns:
          File with the output
        """
        # The process here is to generate a CSV file with the content of the data annotations file
        # and also the total of labels per eye. This will help us later to process the images
        if self.folder != "":
            folder_to_save = os.path.join(self.folder, 'predictions.csv')
        else:
            folder_to_save = 'predictions.csv'
        with open(folder_to_save, 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['file_name','N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"])
            count = 0
            for sub in self.prediction:
                N = sub[0]
                DR = sub[1]
                ARMD = sub[2]
                MH = sub[3]
                DN = sub[4]
                MYA = sub[5]
                TSLN = sub[6]
                ODC = sub[7]
                O=sub[8]
                file_writer.writerow([count, N, DR, ARMD,MH,DN,MYA,TSLN,ODC,O])
                count = count + 1

    def save_all(self, y_test):
        """Generate a CSV that contains the output of all the classes.
        Args:
          No arguments are required.
        Returns:
          File with the output
        """
        # The process here is to generate a CSV file with the content of the data annotations file
        # and also the total of labels per eye. This will help us later to process the images
        if self.folder != "":
            folder_to_save = os.path.join(self.folder, 'predictions.csv')
        else:
            folder_to_save = 'predictions.csv'
        with open(folder_to_save, 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['file_name','N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"])
            count = 0
            for i in range(self.num_images_test):
                N = self.prediction[i][0]
                DR = self.prediction[i][1]
                ARMD = self.prediction[i][2]
                MH = self.prediction[i][3]
                DN = self.prediction[i][4]
                MYA = self.prediction[i][5]
                TSLN = self.prediction[i][6]
                ODC = self.prediction[i][7]
                O=self.prediction[i][8]
                file_writer.writerow([count, N, DR, ARMD,MH,DN,MYA,TSLN,ODC,O])
                count = count + 1

        if self.folder != "":
            folder_to_save = os.path.join(self.folder, 'ground_truth.csv')
        else:
            folder_to_save = 'ground_truth.csv'
        with open(folder_to_save, 'w', newline='') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['file_name','N','DR','ARMD',"MH",'DN','MYA','TSLN',"ODC","O"])
            count = 0
            for i in range(self.num_images_test):
                N2 = self.prediction[i][0]
                DR2 = self.prediction[i][1]
                ARMD2 = self.prediction[i][2]
                MH2 = self.prediction[i][3]
                DN2 = self.prediction[i][4]
                MYA2 = self.prediction[i][5]
                TSLN2 = self.prediction[i][6]
                ODC2 = self.prediction[i][7]
                O2=self.prediction[i][8]

                file_writer.writerow([count, N2, DR2, ARMD2,MH2,DN2,MYA2,TSLN2,ODC2,O2])
                count = count + 1