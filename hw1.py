import cv2 as cv
import time
import numpy as np
import os
# import jbf
import myfilter as fil
from argparse import ArgumentParser as parser
from matplotlib import pyplot as plt

parser = parser(description='Computer Vision 2018 Homework 1')
parser.add_argument('-i', '--input_dir', 
    default='./testdata', 
    help = 'where to find input images, default: ./testdata')
parser.add_argument('-o', '--output_dir', 
    default='./output_imgs',
    help = 'where to save output images, default: ./output_imgs')
parser.add_argument('-v', '--verbose', 
    action = 'store_true', 
    help = 'print detail information during execution')
args = parser.parse_args()

verbose = args.verbose
image_dir = args.input_dir
output_dir = args.output_dir
output_file = output_dir + '/result.txt'

images = []
for i in os.listdir(image_dir):
    if i.startswith('1'):
        images = images + [os.path.join(image_dir, i)]

if not os.path.exists(output_dir):
    os.system('mkdir {}'.format(output_dir))

if os.listdir(output_dir):
    os.system('rm {}/*'.format(output_dir))
    
index = 0
candidateParameters = []
candidateImgs = []
sigmaS = [1, 2, 3]
sigmaR = [0.05, 0.1, 0.2]

candidateParameters = {}
for i in range(11):
    left = 11 - i
    for j in range(left):
        candidateParameters[(i, j, left - j - 1)] = [0,0,0]

# sigmaS = [1]
# sigmaR = [0.05]
# candidateParameters = {(0,0,10):[0,0,0], (0,1,9):[0,0,0], (0,2,8):[0,0,0]}

countImg = 0
for i in images:
    countImg += 1
    imgOrigin = cv.imread(i, cv.IMREAD_COLOR)
    b, g, r = cv.split(imgOrigin)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    conventionalImg = np.int16(y)
    cv.imwrite(output_dir + '/' + i[i.find('.',1) - 2:i.find('.',1)] + "_y.png", conventionalImg)

    for element in candidateParameters:
        index += 1
        y = element[0] / 10 * b + element[1] / 10 * g + element[2] / 10 * r
        y = np.int16(y)
        candidateParameters[element][0] = 0
        candidateParameters[element][1] = y

    winner = []
    countContest = 0
    for _sigmaS in sigmaS:
        for _sigmaR in sigmaR:
            countContest += 1
            print('----------- contest %d ( sigma_s = %d, sigma_r = %.2f ) -----------' % (countContest, _sigmaS, _sigmaR))
            print('.......... computing jbf of original image ..........')
            if verbose:
                tStart = time.time()
            jbfImgOrigin = fil.bilateral(imgOrigin, _sigmaS, _sigmaR, imgOrigin)
            if verbose:
                tEnd = time.time()
                print('time: %.2f' % (tEnd - tStart))
            
            # if verbose:
            #     tStart = time.time()
            # jbfImgOrigin1 = jbf.jbf(imgOrigin, imgOrigin, _sigmaS, _sigmaR)
            # if verbose:
            #     tEnd = time.time()
            #     print('time: %.2f' % (tEnd - tStart))
            
            countCandidate = 0
            for element in candidateParameters:
                candidateImg = candidateParameters[element][1]
                countCandidate += 1
                print('.......... computing candidate %d ( wb = %02.1f, wg = %02.1f, wr = %02.1f ) ..........' % (countCandidate, element[0] / 10, element[1] / 10, element[2] / 10))

                if verbose:
                    tStart = time.time()
                jbfCandidate = fil.bilateral(imgOrigin, _sigmaS, _sigmaR, candidateImg)
                candidateParameters[element][2] = np.sum(np.abs(jbfImgOrigin - jbfCandidate))
                if verbose:
                    tEnd = time.time()
                    print('time: %.2f' % (tEnd - tStart))
                    print('score: %.2f' % (candidateParameters[element][2]))
            
                # if verbose:
                #     tStart = time.time()
                # jbfCandidate1 = jbf.jbf(imgOrigin, candidateImg, _sigmaS, _sigmaR)
                # if verbose:
                #     tEnd = time.time()
                #     print('time: %.2f' % (tEnd - tStart))
            
            print('.......... computing local minimal ..........')
            for candidate in candidateParameters:
                wb = candidate[0]
                wg = candidate[1]
                wr = candidate[2]
                isLocalMin = True
                localMin = candidateParameters[candidate][2]

                element = (wb, wg-1, wr+1)
                if element in candidateParameters and candidateParameters[element][2] < localMin:
                    isLocalMin = False
                element = (wb, wg+1, wr-1)
                if element in candidateParameters and candidateParameters[element][2] < localMin:
                    isLocalMin = False
                element = (wb-1, wg, wr+1)
                if element in candidateParameters and candidateParameters[element][2] < localMin:
                    isLocalMin = False
                element = (wb+1, wg, wr-1)
                if element in candidateParameters and candidateParameters[element][2] < localMin:
                    isLocalMin = False
                element = (wb-1, wg+1, wr)
                if element in candidateParameters and candidateParameters[element][2] < localMin:
                    isLocalMin = False
                element = (wb+1, wg-1, wr)
                if element in candidateParameters and candidateParameters[element][2] < localMin:
                    isLocalMin = False
                if isLocalMin:
                    candidateParameters[candidate][0] += 1

            count = 0
            orderdCandidate = []
            for element in candidateParameters:
                count += 1
                if verbose:
                    print('score of candidate %d: %d' % (count, candidateParameters[element][0]))
                orderdCandidate.append([ candidateParameters[element][0], candidateParameters[element][1], element ])
            print('')
            winner = sorted(orderdCandidate, reverse = True, key = lambda x : x[0])

    count = 0
    fout = open(output_file, 'a+')
    for element in winner:
        count += 1
        if count <= 3 and element[0] != 0:
            pngName = '/' + i[i.find('.',1) - 2:i.find('.',1)] + '_y' + str(count) + '.png'
            cv.imwrite(output_dir + pngName, element[1], [int(cv.IMWRITE_PNG_COMPRESSION), 0])
            fout.write('weight combination of {}: wb = {:02.1f}, wg = {:02.1f}, wr = {:02.1f}\n'.format(pngName, element[2][0] / 10, element[2][1] / 10, element[2][2] / 10))
    fout.write('\n')
    fout.close()
