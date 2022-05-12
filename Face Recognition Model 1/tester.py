from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import cv2
import os
import numpy as np
from scipy import rand
import faceRecognition as fr

pred = []
# Comment belows lines when running this program second time.Since it saves training.yml file in directory
faces, faceID = fr.labels_for_training_data('trainingImages')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write('trainingData.yml')
# This module takes images  stored in diskand performs face recognition
for i in range(1, 12):
    test_img = cv2.imread('TestImages/test'+str(i)+'.jpg')  # test_img path
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("faces_detected:", faces_detected)

    # Uncomment below line for subsequent runs
    # face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    # face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

    # creating dictionary containing names for each label
    name = {0: "Bhavya", 1: "Dishant", 2: "Jayant"}

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(
            roi_gray)  # predicting the label of given image
        print("confidence:", confidence)
        print("label:", label)
        print("person:", name[label])
        if(label == None):
            label = rand.randint(0, 1)

        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        if confidence != 0 and (confidence > 100 or confidence < 10):
            # If confidence less than 37 then don't print predicted face text on screen
            predicted_name = 'UNKNOWN'
            label = -1
        pred.append(label)
        fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (1000, 1000))
    cv2.imshow("face dtecetion tutorial", resized_img)
    cv2.waitKey(0)  # Waits indefinitely until a key is pressed
actual = [2, 1, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, -1, 2]
print("\n\n\nActual/True Classifcation: ", actual)
print("Predicted Classifcation: ", pred)
print("Accuracy: ", accuracy_score(actual, pred)*100, '%')
print("Confusion Matrix: \n", confusion_matrix(actual, pred))
confusion_matrix = confusion_matrix(actual, pred)
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)
print("\nFalse Positives: ", FP)
print("False Negatives: ", FN)
print("True Positives: ", TP)
print("True Negatives: ", TN)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("\nSensitivity: ", TPR)
print("Specificity: ", TNR)
print("Precision: ", PPV)
print("Negative Predictive Value: ", NPV)
print("False Positive Rate: ", FPR)
print("False Negative Rate: ", FNR)
print("False Discovery Rate: ", FDR)
print("Overall Accuracy: ", ACC)
cv2.destroyAllWindows
