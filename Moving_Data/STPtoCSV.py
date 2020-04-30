#Author: Tahereh Arabghalizi
#Date: 08/14/2019
#University of Pittsburgh
#Convert stp to CSV format - Run on Historical Data from Port Authority of Allegheny County

import re
import os


def toCSV(stppath, csvpath, filename):
    stppath = stppath + filename + '.stp'
    csvpath = csvpath + filename + '.csv'
    regex = '([A-Z]{1,2}\d{4,5}|999\d{1}|00009999) (.*?) \d{6}'

    with open(csvpath,'w') as file:
        with open(stppath) as fp:
            fp.readline()
            header = fp.readline()
            header = re.sub('Patter Block','Patter_Block',header)
            header = re.sub('Node ID','Node_ID',header)
            header = re.sub('WS1S2','W S1 S2',header)
            header = re.sub('TPofTP','TPo fTP',header)
            header = re.sub(' +',' ',header)
            header = re.sub(' ',',' , header)
            file.write(header)
            for line in fp:
                line = re.sub(' +',' ',line)
                matches = re.finditer(regex,line)
                for matchNum, match in enumerate(matches):
                    matchNum = matchNum + 1
                    match = match.group(2)
                    if(match.find('.')==-1):
                        rep = re.sub(' ', '_', match)
                        line = re.sub(match, rep , line)

                line = re.sub(' ',',',line)
                line = re.sub('_',' ',line)
                file.write(line)


if __name__ == '__main__':

    #specify folder paths for stp and csv files here
    stppath = '1909.stp'
    csvpath = '1909.csv'

    for filee in os.listdir(stppath):
        print('stp filename= ', filee)
        filename = filee.split('.')[0]
        toCSV(stppath, csvpath, filename)
