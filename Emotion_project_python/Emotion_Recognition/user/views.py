from tkinter import filedialog, Tk, Button

from django.db.models import Count
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
import cv2
import numpy as np
import dlib
from imutils import face_utils
import face_recognition
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# Create your views here.
from user.models import RegisterModel, RecognitionModel


def login(request):

    if request.method=="POST":
        usid=request.POST.get('your_email')
        pswd = request.POST.get('password')
        try:
            check = RegisterModel.objects.get(email=usid,password=pswd)
            request.session['userd_id']=check.id
            return redirect('mydetails')
        except:
            pass
    return render(request,'login.html')




def register(request):
    if request.method == 'POST':
        firstname = request.POST.get('first_name')
        lastname = request.POST.get('last_name')

        password = request.POST.get('password')
        comfirm_password = request.POST.get('comfirm_password')
        email = request.POST.get('your_email')


        if password == comfirm_password:
            RegisterModel.objects.create(firstname=firstname, lastname=lastname, password=password,
                                         repassword=comfirm_password, email=email, )
        else:
            return HttpResponse("Password Not Match")

        return redirect('login')
    return render(request, 'register.html')


def mydetails(request):
    name = request.session['userd_id']
    ted = RegisterModel.objects.get(id=name)

    return render(request, 'mydetails.html',{'objects':ted})



def recognition(request):
    ind=''
    def select_image1():
        # grab a reference to the image panels
        global panelA, panelB

        # open a file chooser dialog and allow the user to select an input
        # image
        path = filedialog.askopenfilename()

        USE_WEBCAM = False  # If false, loads video file source

        # parameters for loading data and images
        emotion_model_path = './models/emotion_model.hdf5'
        emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        frame_window = 10
        emotion_offsets = (20, 40)

        # loading models
        detector = dlib.get_frontal_face_detector()
        emotion_classifier = load_model(emotion_model_path)

        # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        emotion_window = []

        # starting video streaming

        cv2.namedWindow('window_frame')
        video_capture = cv2.VideoCapture(0)

        # Select video or webcam feed
        cap = None

        cap = cv2.VideoCapture(path)  # Video file source

        while cap.isOpened():  # True:
            ret, bgr_image = cap.read()

            # bgr_image = video_capture.read()[1]
            if bgr_image is None:
                break

            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            faces = detector(rgb_image)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                name = request.session['userd_id']
                userObj = RegisterModel.objects.get(id=name)
                user = userObj.firstname
                RecognitionModel.objects.create(username=user, result=emotion_text, path=path)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))


                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
                draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('window_frame', bgr_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    # initialize the window toolkit along with the two image panels
    root = Tk()
    panelA = None
    panelB = None

    # create a button, then when pressed, will trigger a file chooser
    # dialog and allow the user to select an input image; then add the
    # button the GUI
    btn = Button(root, text="Select Your Video", command=select_image1)
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
    root.mainloop()

    Vobj = RecognitionModel.objects.all()
    return render(request, 'recognition.html', {'v': Vobj})






def charts(request,chart_type):
    chart = RecognitionModel.objects.values('result').annotate(dcount=Count('result'))
    return render(request,'charts.html',{'chart_type':chart_type,'form':chart})



