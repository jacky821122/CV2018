import math
import numpy as np
import cv2 as cv

def bilateral(img, sigS, sigR, guide):
    isGray = False
    if len(guide.shape) == 2:
        isGray = True
    img = img / 255
    guide = guide / 255
    r = 3 * sigS
    w = r * 2 + 1
    B, G, R = cv.split(img)
    height = len(B)
    width = len(B[0])
    if isGray:
        guide_pad = np.pad(guide, ((r,r), (r,r)), 'constant')
    else:
        guide_B, guide_G, guide_R = cv.split(guide)
        guide_b_pad = np.pad(guide_B, ((r,r), (r,r)), 'constant')
        guide_g_pad = np.pad(guide_G, ((r,r), (r,r)), 'constant')
        guide_r_pad = np.pad(guide_R, ((r,r), (r,r)), 'constant')
    b_pad = np.pad(B, ((r,r), (r,r)), 'constant')
    g_pad = np.pad(G, ((r,r), (r,r)), 'constant')
    r_pad = np.pad(R, ((r,r), (r,r)), 'constant')

    hs = np.zeros((w,w), dtype = int)
    for i in range(w):
        for j in range(w):
            hs[i, j] = ((j-r) ** 2 + (i-r) ** 2) / (2 * (sigS ** 2))

    newImg_b = np.zeros((height, width), dtype=float)
    newImg_g = np.zeros((height, width), dtype=float)
    newImg_r = np.zeros((height, width), dtype=float)

    for i in range(height):
        for j in range(width):
            bWindow = b_pad[i : i + w, j : j + w]
            gWindow = g_pad[i : i + w, j : j + w]
            rWindow = r_pad[i : i + w, j : j + w]
            if isGray:
                guideWindow = guide_pad[i : i + w, j : j + w]
                center = guideWindow[r,r]
                centerWindow = np.full((w,w), center)
                hr = ((guideWindow - centerWindow) ** 2) / (2 * (sigR ** 2))
            else:
                guideBWindow = guide_b_pad[i : i + w, j : j + w]
                guideGWindow = guide_g_pad[i : i + w, j : j + w]
                guideRWindow = guide_r_pad[i : i + w, j : j + w]
                centerB = guideBWindow[r,r]
                centerG = guideGWindow[r,r]
                centerR = guideRWindow[r,r]
                centerBWindow = np.full((w,w), centerB)
                centerGWindow = np.full((w,w), centerG)
                centerRWindow = np.full((w,w), centerR)
                hr = (((guideBWindow - centerBWindow) ** 2) + ((guideGWindow - centerGWindow) ** 2) + ((guideRWindow - centerRWindow) ** 2)) / (2 * (sigR ** 2))
            expWindow = np.exp((hs + hr) * -1)
            W = np.sum(expWindow)
            newImg_b[i,j] = (np.sum(expWindow * bWindow) / W)
            newImg_g[i,j] = (np.sum(expWindow * gWindow) / W)
            newImg_r[i,j] = (np.sum(expWindow * rWindow) / W)                

    newImg = cv.merge([newImg_b, newImg_g, newImg_r])
    return (newImg) * 255
