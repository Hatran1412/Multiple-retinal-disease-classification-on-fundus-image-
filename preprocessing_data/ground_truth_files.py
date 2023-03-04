import csv


class GroundTruthFiles:
    def __init__(self):
        self.N=[]
        self.DR = []
        self.ARMD = []
        self.MH = []
        self.DN = []
        self.MYA = []
        self.TSLN = []
        self.ODC = []
        self.O = []

    def populate_vectors(self, ground_truth_file):
        with open(ground_truth_file) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
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
                if  file_name!= "ID":
                    print("Processing image: " + file_name + ".png")
                    if N=='1':
                        self.N.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])
                    if DR == '1':
                        self.DR.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])  
                    if ARMD == '1':
                        self.ARMD.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])  
                    if MH == '1':
                        self.MH.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])  
                    if DN == '1':
                        self.DN.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])  
                    if MYA == '1':
                        self.MYA.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])  
                    if TSLN == '1':
                        self.TSLN.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])    
                    if ODC == '1':
                        self.ODC.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])   
                    if O == '1':
                        self.O.append([file_name,N,DR,ARMD, MH,DN,MYA,TSLN, ODC,O])  