import dash
import dash_core_components as dcc
import dash_html_components as html
import os
#import selfie_verification_20190423_1237

import traceback
import sys

app = dash.Dash(__name__)

from flask import request

data_cache = {}

app.layout = html.Div(style=dict(fontFamily='Calibri'), children = [
    html.Div(html.H1(children='Welcome to Visual ID Verification Process', style={'textAlign': 'center'})),
    html.Div(html.H2(children='Kindly follow 3 simple steps to get your ID verified', style={'textAlign': 'center', 'color': 'red'})),
    html.Div(html.H4(children='Step1: Face the camera, and place your ID card just underneath your face', style={'color': 'grey'})),
    html.Div(html.H4(children='Step2: Face the camera, and smile'), style={'color': 'grey'}),
    html.Div(html.H4(children='Step3: Face the camera, and blink your eyes', style={'color': 'grey'})),
    html.Div(html.H2(children='Enter your Email id to continue. ''', style={'color': 'red'})),
    html.Div(dcc.Input(id='input-box', type='text'), style={'columnCount': 0}),
    html.Button('Submit', id='Button'),
    dcc.Loading(id="loading-1", children=[html.Div(id='output-container-calc-start')], type="default")
])

app.css.append_css({'external_url':  'https://codepen.io/chriddyp/pen/bWLwgP.css'})

@app.callback(
    dash.dependencies.Output('output-container-calc-start', 'children'),
    [dash.dependencies.Input('Button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    #print("update_output")
    if n_clicks is not None and value is not None:
        try:
            send_to_email=value.strip()
            var1="Success"
            var2="Failed"

            # coding: utf-8

            # In[13]:

            # import the necessary packages
            import time
            from scipy.spatial import distance as dist
            from imutils.video import FileVideoStream
            from imutils.video import VideoStream
            from imutils import face_utils
            import argparse
            import imutils
            import time
            import dlib
            import cv2

            #def draw_text(frame, text, x, y, color=(255,0,0), thickness=4, size=5):
                    #if x is not None and y is not None:
                            #cv2.putText(frame, text, (300,300), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

            ######################################################################################################
            ################### 1. First part of Face Detection (Developed by Udit)
            ######################################################################################################

            init_time = time.time()
            test_timeout = init_time+15
            final_timeout = init_time+15
            counter_timeout_text = init_time+1
            counter_timeout = init_time+1
            counter = 15

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # capture frames from a camera
            cap = cv2.VideoCapture(0)

            # loop runs if capturing has been initialized.
            while 1:
                    # reads frames from a camera
                    ret, img = cap.read()

                    # convert to gray scale of each framese
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Detects faces of different sizes in the input image
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    if (time.time() > counter_timeout_text and time.time() < test_timeout):
                            cv2.putText(img, "Timer: {:.0f}".format(counter), (450, 450),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            #draw_text(frame, str(counter), center_x, center_y)
                            counter_timeout_text+=0.03333
                    if (time.time() > counter_timeout and time.time() < test_timeout):
                            counter-=1
                            counter_timeout+=1

                    num_face=0
                    for (x,y,w,h) in faces:
                            # To draw a rectangle in a face
                            num_face=num_face+1
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            roi_gray = gray[y:y+h, x:x+w]
                            roi_color = img[y:y+h, x:x+w]
                            img_item=".jpg"
                            cv2.imwrite("Applicant" + str(num_face) + img_item, roi_gray)

                    #Display an image in a window
                    cv2.imshow('img',img)

                    # Wait for q key to stop
                    if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > final_timeout):
                            cap.release()
                            cv2.destroyAllWindows()
                            break

            #Close the window
            cap.release()
            cv2.destroyAllWindows()

            ######################################################################################################
            #--------------- 1. End of First part of Face Detection (Developed by Udit)
            ######################################################################################################

            #### things to check here or output to be considered - *****  num_face  ****

            ######################################################################################################
            #--------------- 2. Second Part Smile Detection Starts Here (Developed by Udit)
            ######################################################################################################

            init_time = time.time()
            test_timeout = init_time+15
            final_timeout = init_time+15
            counter_timeout_text = init_time+1
            counter_timeout = init_time+1
            counter = 15

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

            cap = cv2.VideoCapture(0)
            font=cv2.FONT_HERSHEY_COMPLEX_SMALL
            g=0

            while True:

                    ret, img = cap.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #faces, faceRejectLevels, faceLevelWeights= face_cascade.detectMultiScale3(gray, 1.3, 5, outputRejectLevels=True)

                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    f=0

                    #frame = cap.read()
                    #img = imutils.resize(img, width=450)

                    #center_x = int(frame.shape[0]/2)
                    #center_y = int(frame.shape[0]/2)
                    if (time.time() > counter_timeout_text and time.time() < test_timeout):
                            cv2.putText(img, "Timer: {:.0f}".format(counter), (450, 450),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            #draw_text(frame, str(counter), center_x, center_y)
                            counter_timeout_text+=0.03333
                    if (time.time() > counter_timeout and time.time() < test_timeout):
                            counter-=1
                            counter_timeout+=1
                    for(x,y,w,h) in faces:
                            #if (round(faceLevelWeights[f][0],3)) <= 5: continue
                            #print(round(faceLevelWeights[f][0],3))         To Display Face Confidence
                            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                    #        cv2.putText(img, str(round(faceLevelWeights[f][0],3)), (x,y),font, 1, (255,255,255), 2)
                            roi_gray = gray[y:y+h, x:x+w]
                            roi_color = img[y:y+h, x:x+w]
                            eyes, rejectLevels, levelWeights = smile_cascade.detectMultiScale3(roi_gray, outputRejectLevels=True)
                            i=0
                            for(ex,ey,ew,eh) in eyes:
                                if(round(levelWeights[i][0],3)>=3.5):
                                    #print(round(levelWeights[i][0],3))       To Display Smile Confidence
                                    g+=1
                                    cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                                    cv2.putText(roi_color,str(round(levelWeights[i][0],3)),(ex,ey), font,1,(255,255,255),2)
                                i+=1
                            cv2.putText(roi_color, "Smile: {}".format(g), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            f+=1
                    cv2.imshow('img',img)

                    if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > final_timeout):
                            cap.release()
                            cv2.destroyAllWindows()
                            #vs.stop()
                            break

            cap.release()
            cv2.destroyAllWindows()
            #vs.stop()

            ######################################################################################################
            #--------------- 2. End of Smile Detection Here (Developed by Udit)
            ######################################################################################################

            #### things to check here or output to be considered - *****  g (number of smiles)  ****

            ######################################################################################################
            #--------------- 3. Start of Eye Blink Detection Here (Developed by Udit)
            ######################################################################################################

            init_time = time.time()
            test_timeout = init_time+20
            final_timeout = init_time+20
            counter_timeout_text = init_time+1
            counter_timeout = init_time+1
            counter = 20
            cap = cv2.VideoCapture(0)

            while(cap.isOpened()):
                    def eye_aspect_ratio(eye):
                            # compute the euclidean distances between the two sets of
                            # vertical eye landmarks (x, y)-coordinates
                            A = dist.euclidean(eye[1], eye[5])
                            B = dist.euclidean(eye[2], eye[4])

                            # compute the euclidean distance between the horizontal
                            # eye landmark (x, y)-coordinates
                            C = dist.euclidean(eye[0], eye[3])

                            # compute the eye aspect ratio
                            ear = (A + B) / (2.0 * C)

                            # return the eye aspect ratio
                            return ear

                    # construct the argument parse and parse the arguments
                    ap = argparse.ArgumentParser()
                    #ap.add_argument("-p", "Users/ankitamunshi/PycharmProjects/bakwaas/facedetection/shape_predictor_68_face_landmarks.dat", required=True,
                     #       help="/Users/ankitamunshi/PycharmProjects/bakwaas/facedetection/shape_predictor_68_face_landmarks.dat")
                    ap.add_argument("-v", "--video", type=str, default="",
                           help="path to input video file")
                    args = vars(ap.parse_args())

                    # define two constants, one for the eye aspect ratio to indicate
                    # blink and then a second constant for the number of consecutive
                    # frames the eye must be below the threshold
                    EYE_AR_THRESH = 0.3
                    EYE_AR_CONSEC_FRAMES = 3

                    # initialize the frame counters and the total number of blinks
                    COUNTER = 0
                    TOTAL = 0

                    # initialize dlib's face detector (HOG-based) and then create
                    # the facial landmark predictor
                    print("[INFO] loading facial landmark predictor...")
                    detector = dlib.get_frontal_face_detector()
                    predictor = dlib.shape_predictor("C:\\Users\\user\\Desktop\\shape_predictor_68_face_landmarks.dat")

                    # grab the indexes of the facial landmarks for the left and
                    # right eye, respectively
                    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

                    # start the video stream thread
                    print("[INFO] starting video stream thread...")
                    vs = FileVideoStream(args["video"]).start()
                    fileStream = True
                    vs = VideoStream(src=0).start()
                    #vs = VideoStream(usePiCamera=True).start()
                    fileStream = False
                    time.sleep(1.0)

                    # loop over frames from the video stream
                    while True:
                            # if this is a file video stream, then we need to check if
                            # there any more frames left in the buffer to process
                            if fileStream and not vs.more():
                                    break

                            # grab the frame from the threaded video file stream, resize
                            # it, and convert it to grayscale
                            # channels)
                            frame = vs.read()
                            frame = imutils.resize(frame, width=450)
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            # detect faces in the grayscale frame
                            rects = detector(gray, 0)

                            # loop over the face detections
                            for rect in rects:
                                    # determine the facial landmarks for the face region, then
                                    # convert the facial landmark (x, y)-coordinates to a NumPy
                                    # array
                                    shape = predictor(gray, rect)
                                    shape = face_utils.shape_to_np(shape)

                                    # extract the left and right eye coordinates, then use the
                                    # coordinates to compute the eye aspect ratio for both eyes
                                    leftEye = shape[lStart:lEnd]
                                    rightEye = shape[rStart:rEnd]
                                    leftEAR = eye_aspect_ratio(leftEye)
                                    rightEAR = eye_aspect_ratio(rightEye)

                                    # average the eye aspect ratio together for both eyes
                                    ear = (leftEAR + rightEAR) / 2.0

                                    # compute the convex hull for the left and right eye, then
                                    # visualize each of the eyes
                                    leftEyeHull = cv2.convexHull(leftEye)
                                    rightEyeHull = cv2.convexHull(rightEye)
                                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                                    # check to see if the eye aspect ratio is below the blink
                                    # threshold, and if so, increment the blink frame counter
                                    if ear < EYE_AR_THRESH:
                                            COUNTER += 1

                                    # otherwise, the eye aspect ratio is not below the blink
                                    # threshold
                                    else:
                                            # if the eyes were closed for a sufficient number of
                                            # then increment the total number of blinks
                                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                                    TOTAL += 1

                                            # reset the eye frame counter
                                            COUNTER = 0

                                    center_x = int(frame.shape[0]/2)
                                    center_y = int(frame.shape[0]/2)
                                    if (time.time() > counter_timeout_text and time.time() < test_timeout):
                                            cv2.putText(frame, "Timer: {:.0f}".format(counter), (300, 300),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                            #draw_text(frame, str(counter), center_x, center_y)
                                            counter_timeout_text+=0.03333
                                    if (time.time() > counter_timeout and time.time() < test_timeout):
                                            counter-=1
                                            counter_timeout+=1
                                    cv2.imshow('frame', frame)

                                    # draw the total number of blinks on the frame along with
                                    # the computed eye aspect ratio for the frame
                                    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # show the frame
                            cv2.imshow("Frame", frame)
                            #key = cv2.waitKey(1) & 0xFF


                            if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > final_timeout):
                                    break
                                    cv2.destroyAllWindows()
                                    vs.stop()

                            # if the `q` key was pressed, break from the loop
                            #if key == ord("q"):
                                   #break

                    # do a bit of cleanup
                    cap.release()
                    cv2.destroyAllWindows()
                    vs.stop()

            ######################################################################################################
            #--------------- 3. End of Eye Blink Detection Here (Developed by Udit)
            ######################################################################################################

            #### things to check here or output to be considered - *****  TOTAL (number of blinks)  ****

            ######################################################################################################
            #--------------- 4. Face Comparison Starts Here (Developed by Udit)
            ######################################################################################################

            import cv2
            import tensorflow
            import numpy.core.multiarray
            from keras.layers import Softmax
            from scipy import ndimage, misc

            image1 = cv2.imread('Applicant1.jpg')
            image2 = cv2.imread('Applicant2.jpg')

            image1_1 = misc.imresize(image1, (150, 150))
            image2_1 = misc.imresize(image2, (150, 150))

            #convert the images to grayscale
            image1_1 = cv2.cvtColor(image1_1, cv2.COLOR_BGR2GRAY)
            image2_1 = cv2.cvtColor(image2_1, cv2.COLOR_BGR2GRAY)

            def mse(imageA, imageB):
                # the 'Mean Squared Error' between the two images is the
                # sum of the squared difference between the two images
                # NOTE: the two images must have the same dimension
                err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
                err /= float(imageA.shape[0] * imageA.shape[1])
                # return the MSE, the lower the error, the more "similar"
                # the two images are
                return err

            def compare_images(imageA, imageB, title):
                # compute the mean squared error and structural similarity
                # index for the images
                m = mse(imageA, imageB)
                s = measure.compare_ssim(imageA, imageB)
                # setup the figure
                fig = plt.figure(title)
                plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
                # show first image
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(imageA, cmap = plt.cm.gray)
                plt.axis("off")
                # show the second image
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(imageB, cmap = plt.cm.gray)
                plt.axis("off")
                # show the images
                plt.show()

            #from skimage.measure import structural_similarity as ssim
            from skimage import measure
            import matplotlib.pyplot as plt
            import numpy as np
            import cv2

            # initialize the figure
            fig = plt.figure("Images")
            images = ("Image1", image1_1), ("Image2", image2_1)
            # loop over the images
            for (i, (name, image)) in enumerate(images):
                # show the image
                ax = fig.add_subplot(1, 3, i + 1)
                ax.set_title(name)
                plt.imshow(image, cmap = plt.cm.gray)
                plt.axis("off")

            #show the figure
            #plt.show()

            # compare the images
            #compare_images(id_2, id_2, "ID vs. ID")
            compare_images(image1_1, image2_1, "ID vs. selfie")

            match_rate=measure.compare_ssim(image1_1, image2_1)

            ######################################################################################################
            #--------------- 4. End of Face Comparison Here (Developed by Udit)
            ######################################################################################################

            #### things to check here or output to be considered - *****  match_rate  ****

            ######################################################################################################
            #--------------- 5. Start of Mail Sending (Developed by Udit)
            ######################################################################################################

            import random
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            card_number=random.randint(4000000000000000,4999999999999999)
            #print(card_number)

            card_number1=str(card_number)
            cropped_number1 = card_number1[:4]
            cropped_number2 = card_number1[4:8]
            cropped_number3 = card_number1[8:12]
            cropped_number4 = card_number1[12:]

            expiry_year=random.randint(20,25)
            #print(expiry_year)

            expiry_month=random.randint(1,12)
            #print(expiry_month)

            cvv=str(random.randint(100,999))
            #print(cvv)

            #print(cropped_number4)

            cc_num = str(cropped_number1 + " " + cropped_number2 + " " + cropped_number3 + " " + cropped_number4)
            expiry = str(expiry_month) + "/" + str(expiry_year)
            #print("Your virtual card number is : ", cc_num)
            #print("Your virtual card expires on : ", expiry)
            #print("Your virtual card cvv no is : ", cvv)

            email = 'dish.ag87@googlemail.com'
            password = 'uditagar1'
            #send_to_email = 'sdm.values@gmail.com'
            subject = 'Your virtual credit card is approved'
            message = 'Dear Customer,\nWe thank you for choosing HSBC Retail Banking services\nWe are excited to have you on board!\nCongratulations on your recent credit card approval\nYour virtual credit card no. is : ' + cc_num + '\nYour virtual card expires on : ' + expiry + '\nYour virtual card cvv no is : ' + cvv

            subject1 = 'Your Selfie Verification Failed'
            message1 = 'Selfie Verification Failed\nOur Sales Representative will reach out to you for in-person verification within 48 working hours!\nYou can also call phone banking on 12345'

            if TOTAL > 2:
                    if g > 1 :
                            if num_face > 1:
                                    if match_rate > 0.3:
                                            msg = MIMEMultipart()
                                            msg['From'] = email
                                            msg['To'] = send_to_email
                                            msg['Subject'] = subject
                                            var=var1

                                            msg.attach(MIMEText(message, 'plain'))

                                            server = smtplib.SMTP('smtp.gmail.com', 587)
                                            server.starttls()
                                            server.login(email, password)
                                            text = msg.as_string()
                                            server.sendmail(email, send_to_email, text)
                                            server.quit()
                                    else:
                                            msg = MIMEMultipart()
                                            msg['From'] = email
                                            msg['To'] = send_to_email
                                            msg['Subject'] = subject1
                                            var=var2

                                            msg.attach(MIMEText(message1, 'plain'))

                                            server = smtplib.SMTP('smtp.gmail.com', 587)
                                            server.starttls()
                                            server.login(email, password)
                                            text = msg.as_string()
                                            server.sendmail(email, send_to_email, text)
                                            server.quit()
                            else:
                                    msg = MIMEMultipart()
                                    msg['From'] = email
                                    msg['To'] = send_to_email
                                    msg['Subject'] = subject1
                                    var=var2

                                    msg.attach(MIMEText(message1, 'plain'))

                                    server = smtplib.SMTP('smtp.gmail.com', 587)
                                    server.starttls()
                                    server.login(email, password)
                                    text = msg.as_string()
                                    server.sendmail(email, send_to_email, text)
                                    server.quit()
                    else:
                            msg = MIMEMultipart()
                            msg['From'] = email
                            msg['To'] = send_to_email
                            msg['Subject'] = subject1
                            var=var2

                            msg.attach(MIMEText(message1, 'plain'))

                            server = smtplib.SMTP('smtp.gmail.com', 587)
                            server.starttls()
                            server.login(email, password)
                            text = msg.as_string()
                            server.sendmail(email, send_to_email, text)
                            server.quit()
            else:
                    msg = MIMEMultipart()
                    msg['From'] = email
                    msg['To'] = send_to_email
                    msg['Subject'] = subject1
                    var=var2

                    msg.attach(MIMEText(message1, 'plain'))

                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(email, password)
                    text = msg.as_string()
                    server.sendmail(email, send_to_email, text)
                    server.quit()






            return var

        except Exception as e:
            print(e)
            return "Enter valid Email id"
        else:
            return "Enter your Email id to continue"

if __name__ == '__main__':
    app.run_server(debug=True, port=5001)
