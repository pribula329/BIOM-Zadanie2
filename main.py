import os

import numpy as np
import cv2
from time import sleep
from mtcnn import MTCNN
from sklearn.metrics import mean_squared_error as mse


def vizVideaAnotacie(dir,file, typ):
    if typ == 1:
        videoFile = np.load(dir + file)
        colorImages = videoFile['colorImages_original']
        colorImagesMedium = videoFile['colorImages_medium']
        colorImagesSevere = videoFile['colorImages_severe']
        boundingBox = videoFile['boundingBox']
        landmarks2D = videoFile['landmarks2D']

        return colorImages, colorImagesMedium, colorImagesSevere, boundingBox, landmarks2D
    else:
        videoFile = np.load(dir + file)
        colorImages = videoFile['colorImages']
        colorImagesMedium = videoFile['colorImages']
        colorImagesSevere = videoFile['colorImages']
        boundingBox = videoFile['boundingBox']
        landmarks2D = videoFile['landmarks2D']

        return colorImages, colorImagesMedium, colorImagesSevere, boundingBox, landmarks2D


def vytvorVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium, colorImagesSevere, step):

    i = 0
    while True:
        try:
            img = colorImages[:, :, :, i]
        except:
            break
        img2 = colorImagesMedium[:, :, :, i]
        img3 = colorImagesSevere[:, :, :, i]
        bbox = boundingBox[:, :, i]
        mark = landmarks2D[:, :, i]

        lavyHorny = (int(bbox[0][0]), int(bbox[0][1]))
        pravyDolny = (int(bbox[3][0]), int(bbox[3][1]))

        image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image1 = cv2.rectangle(image1, lavyHorny, pravyDolny, (255, 0, 0), 2)

        image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        image2 = cv2.rectangle(image2, lavyHorny, pravyDolny, (255, 0, 0), 2)

        image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        image3 = cv2.rectangle(image3, lavyHorny, pravyDolny, (255, 0, 0), 2)

        for x in range(0, 65):
            bod = (int(mark[x][0]), int(mark[x][1]))
            image1 = cv2.circle(image1, bod, 1, (0, 0, 255), 1)
            image2 = cv2.circle(image2, bod, 1, (0, 0, 255), 1)
            image3 = cv2.circle(image3, bod, 1, (0, 0, 255), 1)

        imageSpolu = np.concatenate((image1, image2, image3), axis=1)
        cv2.imshow('frame', imageSpolu)
        sleep(0.25)
        i += step
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


def iouFace(faces, bbox, image, typ):

    detekovalo = 0
    preImg = []
    if typ ==1:
        for (x, y, w, h) in faces:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detekovalo += 1

            bA = [int(bbox[0][0]), int(bbox[0][1]), int(bbox[3][0]), int(bbox[3][1])]
            bB = [int(x), int(y), int(x + w), int(y + h)]

            iou = calculate_iou(bA, bB)
            preImg.append(iou)
    else:
        for detec in faces:
            x = int(detec['box'][0])
            y = int(detec['box'][1])
            w = int(detec['box'][2])
            h = int(detec['box'][3])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detekovalo += 1

            bA = [int(bbox[0][0]), int(bbox[0][1]), int(bbox[3][0]), int(bbox[3][1])]
            bB = [int(x), int(y), int(x + w), int(y + h)]

            iou = calculate_iou(bA, bB)
            preImg.append(iou)
    return preImg, image

def mseVypocet(moje, origo):

    return mse(moje, origo)


def eyesMTCNN(face, mark, image):
    ### mtcnn oci
    vypMSE = 100 # ak nenaslo nic

    for detection in face:
        xp, yp = detection['keypoints']['right_eye']
        xl, yl = detection['keypoints']['left_eye']
        image = cv2.circle(image, (xp, yp), 5, (0, 255, 0), 2)
        image = cv2.circle(image, (xl, yl), 5, (0, 255, 0), 2)
        # oko z landmarks
        prave, lave = stredOkaZlandmark(mark)

        image = cv2.circle(image, prave, 5, (255, 0, 0), 2)
        image = cv2.circle(image, lave, 5, (255, 0, 0), 2)
        origo = [prave[0], prave[1], lave[0], lave[1]]
        detekcia = [xp, yp, xl, yl]
        vypMSE = mseVypocet(origo, detekcia)



    return vypMSE, image



def eyesViola(eyes, image):

    for (x, y, w, h) in eyes:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return image

def spojInfo(sirka, p, r, image, precPriemer, recPriemer):
    info = np.zeros((100, sirka, 3), np.uint8)
    cv2.putText(info, 'Precision: ' + str(p), (0, 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info, 'P ave.: ' + str(sum(precPriemer)/len(precPriemer)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info, 'Recall: ' + str(r), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info, 'R ave: ' + str(sum(recPriemer)/len(recPriemer)), (0, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    image = np.concatenate((info, image), axis=0)

    return image


def violaVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium, colorImagesSevere, step):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    i = 0
    prec1 = []
    prec2 = []
    prec3 = []

    rec1 = []
    rec2 = []
    rec3 = []

    while True:
        # aby som videl aj origo
        try:
            img = colorImages[:, :, :, i]
            sirka = colorImages.shape[1]
        except:
            break

        img2 = colorImagesMedium[:, :, :, i]
        img3 = colorImagesSevere[:, :, :, i]

        bbox = boundingBox[:, :, i]
        mark = landmarks2D[:, :, i]

        lavyHorny = (int(bbox[0][0]), int(bbox[0][1]))
        pravyDolny = (int(bbox[3][0]), int(bbox[3][1]))

        image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#viola
        image1 = cv2.rectangle(image1, lavyHorny, pravyDolny, (255, 0, 0), 2)

        image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # viola
        image2 = cv2.rectangle(image2, lavyHorny, pravyDolny, (255, 0, 0), 2)

        image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)  # viola
        image3 = cv2.rectangle(image3, lavyHorny, pravyDolny, (255, 0, 0), 2)

        for x in range(0, 65):
            bod = (int(mark[x][0]), int(mark[x][1]))
            image1 = cv2.circle(image1, bod, 1, (0, 0, 255), 1)
            image2 = cv2.circle(image2, bod, 1, (0, 0, 255), 1)
            image3 = cv2.circle(image3, bod, 1, (0, 0, 255), 1)

        faces1 = face_cascade.detectMultiScale(image1, 1.1, 4) # tvare viola
        faces2 = face_cascade.detectMultiScale(image2, 1.1, 4)  # tvare viola
        faces3 = face_cascade.detectMultiScale(image3, 1.1, 4)  # tvare viola

        #iou a vkreslene face
        preImg1, image1 = iouFace(faces1, bbox, image1,1)
        preImg2, image2 = iouFace(faces2, bbox, image2,1)
        preImg3, image3 = iouFace(faces3, bbox, image3,1)

        # prec a rec pre kazde video
        p1, r1 = precisionAndRecall(preImg1)
        p2, r2 = precisionAndRecall(preImg2)
        p3, r3 = precisionAndRecall(preImg3)
        prec1.append(p1)
        prec2.append(p2)
        prec3.append(p3)
        rec1.append(r1)
        rec2.append(r2)
        rec3.append(r3)

        ## eyes
        eyes1 = eyes_cascade.detectMultiScale(image1, scaleFactor=1.1, minNeighbors=20)
        eyes2 = eyes_cascade.detectMultiScale(image2, scaleFactor=1.1, minNeighbors=20)
        eyes3 = eyes_cascade.detectMultiScale(image3, scaleFactor=1.1, minNeighbors=20)

        image1 = eyesViola(eyes1, image1)

        image2 = eyesViola(eyes2, image2)
        image3 = eyesViola(eyes3, image3)


        ### spojenie info
        image1 = spojInfo(sirka, p1, r1, image1, prec1, rec1)

        image2 = spojInfo(sirka, p2, r2, image2, prec2, rec2)
        image3 = spojInfo(sirka, p3, r3, image3, prec3, rec3)

        imageSpolu = np.concatenate((image1, image2, image3), axis=1)
        cv2.imshow('viola', imageSpolu)

        i += step
        sleep(0.25)
        if i >= 140:
            break
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    #return sum(prec1)/len(prec1),sum(prec2)/len(prec2),sum(prec3)/len(prec3),sum(rec1)/len(rec1),sum(rec2)/len(rec2),sum(rec3)/len(rec3)
    return sum(prec1)/len(prec1),sum(rec1)/len(rec1)


def stredOkaZlandmark(mark):
    xprave = 0
    yprave = 0
    for prave in range(42,48):
        xprave += mark[prave][0]
        yprave += mark[prave][1]

    xlave = 0
    ylave = 0
    for lave in range(36,42):
        xlave += mark[lave][0]
        ylave += mark[lave][1]

    return (int(xprave/6), int(yprave/6)), (int(xlave/6), int(ylave/6))


def mtcnnVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium, colorImagesSevere , step):

    detector = MTCNN()
    i = 0
    prec1 = []
    prec2 = []
    prec3 = []
    rec1 = []
    rec2 = []
    rec3 = []
    mseAv1 = []
    mseAv2 = []
    mseAv3 = []

    while True:
        # aby som videl aj origo
        try:
            img = colorImages[:, :, :, i]
            sirka = colorImages.shape[1]
        except:
            break
        img2 = colorImagesMedium[:, :, :, i]
        img3 = colorImagesSevere[:, :, :, i]
        bbox = boundingBox[:, :, i]
        mark = landmarks2D[:, :, i]

        lavyHorny = (int(bbox[0][0]), int(bbox[0][1]))
        pravyDolny = (int(bbox[3][0]), int(bbox[3][1]))

        image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#mtcnn
        image1 = cv2.rectangle(image1, lavyHorny, pravyDolny, (255, 0, 0), 2)

        image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # mtcnn
        image2 = cv2.rectangle(image2, lavyHorny, pravyDolny, (255, 0, 0), 2)

        image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)  # mtcnn
        image3 = cv2.rectangle(image3, lavyHorny, pravyDolny, (255, 0, 0), 2)

        for x in range(0, 65):
            bod = (int(mark[x][0]), int(mark[x][1]))
            image1 = cv2.circle(image1, bod, 1, (0, 0, 255), 1)
            image2 = cv2.circle(image2, bod, 1, (0, 0, 255), 1)
            image3 = cv2.circle(image3, bod, 1, (0, 0, 255), 1)

        face1 = detector.detect_faces(image1) #mtcnn tvar
        face2 = detector.detect_faces(image2)  # mtcnn tvar
        face3 = detector.detect_faces(image3)  # mtcnn tvar

        ### tvar
        # iou a vkreslene face
        preImg1, image1 = iouFace(face1, bbox, image1,2)
        preImg2, image2 = iouFace(face2, bbox, image2,2)
        preImg3, image3 = iouFace(face3, bbox, image3,2)

        ### oci
        mse1, image1 = eyesMTCNN(face1, mark, image1)
        mse2, image2 = eyesMTCNN(face2, mark, image2)
        mse3, image3 = eyesMTCNN(face3, mark, image3)
        mseAv1.append(mse1)
        mseAv2.append(mse2)
        mseAv3.append(mse3)

        #pre tvar
        # prec a rec pre kazde video
        p1, r1 = precisionAndRecall(preImg1)
        p2, r2 = precisionAndRecall(preImg2)
        p3, r3 = precisionAndRecall(preImg3)
        prec1.append(p1)
        prec2.append(p2)
        prec3.append(p3)
        rec1.append(r1)
        rec2.append(r2)
        rec3.append(r3)

        ### spojenie info
        image1 = spojInfo(sirka, p1, r1, image1, prec1, rec1)
        image2 = spojInfo(sirka, p2, r2, image2, prec2, rec2)
        image3 = spojInfo(sirka, p3, r3, image3, prec3, rec3)

        cv2.putText(image1, 'mse: ' + str(mse1), (0, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image1, 'mse ave: ' + str(sum(mseAv1)/len(mseAv1)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image2, 'mse: ' + str(mse2), (0, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image2, 'mse ave: ' + str(sum(mseAv2) / len(mseAv2)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image3, 'mse: ' + str(mse3), (0, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image3, 'mse ave: ' + str(sum(mseAv3) / len(mseAv3)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        image = np.concatenate((image1, image2, image3), axis=1)

        cv2.imshow('mtcnn', image)

        i += step
        #sleep(0.05)
        if i >= 140:
            break
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    #return sum(prec1) / len(prec1), sum(prec2) / len(prec2), sum(prec3) / len(prec3), sum(rec1) / len(rec1), sum(
    #    rec2) / len(rec2), sum(rec3) / len(rec3), sum(mseAv1) / len(mseAv1), sum(mseAv2) / len(mseAv2), sum(mseAv3) / len(mseAv3)
    return sum(prec1) / len(prec1), sum(rec1) / len(rec1),sum(mseAv1) / len(mseAv1)

def precisionAndRecall(precPreImg):
    ### vypocet precision
    nasloAsponJedno = 0
    truePositivePrecision = 0
    falsePositivePrecision = 0
    for pre in precPreImg:
        if pre >= 0.50:
            if nasloAsponJedno == 0:
                truePositivePrecision += 1
                nasloAsponJedno += 1
            else:
                falsePositivePrecision += 1
        else:
            falsePositivePrecision += 1

    try:
        precision = truePositivePrecision / (truePositivePrecision + falsePositivePrecision)
    except:
        precision = 0
    falseNegative = 1 - truePositivePrecision  # len 1 tvar
    recall = truePositivePrecision / (truePositivePrecision + falseNegative)
    return precision, recall


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def camera():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()

        image = frame

        faces = face_cascade.detectMultiScale(image, 1.1, 4) # tvare viola
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('camera', image)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

#viola
precisionViola1 = []
recallViola1 = []
precisionViola2 = []
recallViola2 = []
precisionViola3 = []
recallViola3 = []
####mtcnn
precisionMTCNN1 = []
recallMTCNN1 = []
mseMTCNN1 = []
precisionMTCNN2 = []
recallMTCNN2 = []
mseMTCNN2 = []
precisionMTCNN3 = []
recallMTCNN3 = []
mseMTCNN3 = []

problemoveRecallViola = []
problemovePrecisionViola = []
problemoveMTCNNMSE = []
problemoveRecallMTCNN = []
problemovePrecisionMTCNN = []
"""
dir_list = os.listdir("C:/Users/Luky/Documents/Skola/BIOM/viz_vzorka/")
for file in dir_list:
    colorImages, colorImagesMedium, colorImagesSevere, boundingBox, landmarks2D = vizVideaAnotacie("C:/Users/Luky/Documents/Skola/BIOM/viz_vzorka/",file,1)
    vytvorVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere, 3)
    pv1, pv2, pv3, rv1, rv2, rv3 = violaVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere, 3)
    pm1, pm2, pm3, rm1, rm2, rm3, mm1, mm2, mm3 =mtcnnVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere, 3)
    precisionViola1.append(pv1)
    recallViola1.append(rv1)
    precisionViola2.append(pv2)
    recallViola2.append(rv2)
    precisionViola3.append(pv3)
    recallViola3.append(rv3)

    precisionMTCNN1.append(pm1)
    recallMTCNN1.append(rm1)
    mseMTCNN1.append(mm1)
    precisionMTCNN2.append(pm2)
    recallMTCNN2.append(rm2)
    mseMTCNN2.append(mm2)
    precisionMTCNN3.append(pm3)
    recallMTCNN3.append(rm3)
    mseMTCNN3.append(mm3)
    if pv1<0.5:
        problemovePrecisionViola.append([file,pv1])
    if rv1<0.5:
        problemoveRecallViola.append([file,rv1])
    if pm1<0.5:
        problemovePrecisionMTCNN.append([file,pm1])
    if rm1<0.5:
        problemoveRecallMTCNN.append([file,rm1])
    if mm1>50:
        problemoveMTCNNMSE.append([file,mm1])

print("VIOLA PRIEMER")
print("***************************")
print("Precision 1:" + str(sum(precisionViola1)/len(precisionViola1)))
print("Recall 1:" + str(sum(recallViola1)/len(recallViola1)))
print("***************************")
print("Precision 2:" + str(sum(precisionViola2)/len(precisionViola2)))
print("Recall 2:" + str(sum(recallViola2)/len(recallViola2)))
print("***************************")
print("Precision 3:" + str(sum(precisionViola3)/len(precisionViola3)))
print("Recall 3:" + str(sum(recallViola3)/len(recallViola3)))
print("***************************")

print("MTCNN PRIEMER")
print("***************************")
print("Precision 1:" + str(sum(precisionMTCNN1)/len(precisionMTCNN1)))
print("Recall 1:" + str(sum(recallMTCNN1)/len(recallMTCNN1)))
print("MSE 1:" + str(sum(mseMTCNN1)/len(mseMTCNN1)))
print("***************************")
print("Precision 2:" + str(sum(precisionMTCNN2)/len(precisionMTCNN2)))
print("Recall 2:" + str(sum(recallMTCNN2)/len(recallMTCNN2)))
print("MSE 1:" + str(sum(mseMTCNN1)/len(mseMTCNN1)))
print("***************************")
print("Precision 3:" + str(sum(precisionMTCNN3)/len(precisionMTCNN3)))
print("Recall 3:" + str(sum(recallMTCNN3)/len(recallMTCNN3)))
print("MSE 1:" + str(sum(mseMTCNN1)/len(mseMTCNN1)))
print("***************************")

dir_list = os.listdir("D:/BIOM/videos-K-O.tar/videos-K-O/")
for file in dir_list:
    print(file)
    colorImages, colorImagesMedium, colorImagesSevere, boundingBox, landmarks2D = vizVideaAnotacie("D:/BIOM/videos-K-O.tar/videos-K-O/",file,2)
    vytvorVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere)
    pv1, rv1= violaVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere, 4)
    pm1,rm1,mm1 = mtcnnVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium, colorImagesSevere, 4)
    precisionViola1.append(pv1)
    recallViola1.append(rv1)
    precisionMTCNN1.append(pm1)
    recallMTCNN1.append(rm1)
    mseMTCNN1.append(mm1)
    if pv1<0.5:
        problemovePrecisionViola.append([file,pv1])
    if rv1<0.5:
        problemoveRecallViola.append([file,rv1])
    if pm1<0.5:
        problemovePrecisionMTCNN.append([file,pm1])
    if rm1<0.5:
        problemoveRecallMTCNN.append([file,rm1])
    if mm1>50:
        problemoveMTCNNMSE.append([file,mm1])

print("VIOLA PRIEMER")
print("***************************")
print("Precision 1:" + str(sum(precisionViola1)/len(precisionViola1)))
print("Recall 1:" + str(sum(recallViola1)/len(recallViola1)))
print("MTCNN PRIEMER")
print("***************************")
print("Precision 1:" + str(sum(precisionMTCNN1)/len(precisionMTCNN1)))
print("Recall 1:" + str(sum(recallMTCNN1)/len(recallMTCNN1)))
print("MSE 1:" + str(sum(mseMTCNN1)/len(mseMTCNN1)))
print("***************************")
print("***************************")
print("***************************")

print("PROBLEMOVE VIDEA VIOLA")
print("Precision")
print(problemovePrecisionViola)
print("Recall")
print(problemoveRecallViola)


print("PROBLEMOVE VIDEA MTCNN")
print("Precision")
print(problemovePrecisionMTCNN)
print("Recall")
print(problemoveRecallMTCNN)
print("MSE")
print(problemoveMTCNNMSE)
"""

#camera()


#ok
file = 'Kirsten_Dunst_2.npz'
colorImages, colorImagesMedium, colorImagesSevere, boundingBox, landmarks2D = vizVideaAnotacie("D:/BIOM/videos-K-O.tar/videos-K-O/",file,2)
vytvorVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere,5)
pv1, rv1= violaVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere, 5)
pm1,rm1,mm1 = mtcnnVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium, colorImagesSevere, 5)
print(file)
print("VIOLA PRIEMER")
print("***************************")
print("Precision 1:" + str(pv1))
print("Recall 1:" + str(rv1))
print("MTCNN PRIEMER")
print("***************************")
print("Precision 1:" + str(pm1))
print("Recall 1:" + str(rm1))
print("MSE 1:" + str(mm1))
print("--------------------------------------------")
#zle Viola
file = 'Nobuyuki_Idei_1.npz'
colorImages, colorImagesMedium, colorImagesSevere, boundingBox, landmarks2D = vizVideaAnotacie("D:/BIOM/videos-K-O.tar/videos-K-O/",file,2)
vytvorVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere,5)
pv1, rv1= violaVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere, 5)
print(file)
print("VIOLA PRIEMER")
print("***************************")
print("Precision 1:" + str(pv1))
print("Recall 1:" + str(rv1))
print("--------------------------------------------")
#zleMTCNN
file = 'Katharine_Hepburn_4.npz'
colorImages, colorImagesMedium, colorImagesSevere, boundingBox, landmarks2D = vizVideaAnotacie("D:/BIOM/videos-K-O.tar/videos-K-O/",file,2)
vytvorVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere,5)
pv1, rv1= violaVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium,colorImagesSevere, 5)
pm1,rm1,mm1 = mtcnnVidea(colorImages, boundingBox, landmarks2D, colorImagesMedium, colorImagesSevere, 5)
print(file)

print("MTCNN PRIEMER")
print("***************************")
print("Precision 1:" + str(pm1))
print("Recall 1:" + str(rm1))
print("MSE 1:" + str(mm1))




#source
#https://gist.github.com/pknowledge/b8ba734ae4812d78bba78c0a011f0d46
#https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
#https://www.mathwords.com/c/centroid_formula.htm