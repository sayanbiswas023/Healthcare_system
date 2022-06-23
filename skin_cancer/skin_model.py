from model import build_model
from re import L
import efficientnet.tfkeras as efn
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
import cv2
from tkinter import *
import customtkinter
import tkinter.font as tkFont
from PIL import Image,ImageTk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import numpy as np
from multiprocessing import Queue


WIDTH=700
HEIGHT=500


customtkinter.set_appearance_mode("System") 
customtkinter.set_default_color_theme("blue") 

win=customtkinter.CTk()
win.title("Dhoom Dhadaka Healthcare")
win.geometry(f"{WIDTH}x{HEIGHT}")
image = Image.open("./images/bg1.jpg").resize((WIDTH+650, HEIGHT+760))
win.bg_image = ImageTk.PhotoImage(image)
win.image_label =Label(master=win, image=win.bg_image)
win.image_label.place(relx=0.15, rely=0.92, anchor=CENTER)


##FUNCTIONS

# global entry_img = numpy.zeros(shape=(256,256,3))
def upload_image():
    f_types = [('jpg Files', '*.jpg'),('png Files', '*.png'),('jpeg Files', '*.jpeg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    # img = ImageTk.PhotoImage(file=filename)

    img =cv2.imread(filename)
    # print(filename)
    # disp_img=Image.open(filename)
    # disp_img=img.resize((100, 100))

    # frame = Frame(win, width=100, height=100)
    # frame.grid(row=15, column=3,padx=5,pady=10)
    # label = Label(frame, image = disp_img)
    yesLb = customtkinter.CTkLabel(master=win, text="✅ image uploaded",justify=LEFT)
    yesLb.place(relx=0.9, rely=0.4, anchor=CENTER)

    # img=np.array(img)
    # fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    # visual = np.log(fft_mag)
    # visual = (visual - visual.min()) / (visual.max() - visual.min())
    # result = Image.fromarray((visual * 255).astype(np.uint8))
    cv2.imwrite('./skin_cancer/skin.jpg',img)
    # print('image saved')

def click_image():
    print("Clicking Image")
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    cv2.imwrite('./skin_cancer/skin.jpg',image)
    del(camera)
    yesLb = customtkinter.CTkLabel(master=win, text="✅ image uploaded",justify=LEFT)
    yesLb.place(relx=0.9, rely=0.4, anchor=CENTER)

def predict():
    showoutput.delete('1.0', END)
    prediction=['Benign(Non Cancerous)','Malignant(Cancerous)']

    #loading model
    model = build_model(256, 6)
    print("...Model Built...") ##
    model.load_weights('./skin_cancer/weights' + '.h5')
    print("...Weights Loaded...") ##

    #loading image
    img=cv2.imread('./skin_cancer/skin.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img = np.expand_dims(img, axis=0)

    print(img.shape)
    pred = model.predict(img)
    output=prediction[int(pred[0][0])]

    # showing prediction
    showoutput.delete('1.0', END)
    showoutput.insert(END,output)




head_font1=tkFont.Font(family="Times New Roman",size=30,weight="bold",slant="italic")
head_font2=tkFont.Font(family="Times New Roman",size=20,weight="bold",slant="italic")
head_font3=tkFont.Font(family="Arial",size=10)

#title
TitleLb =customtkinter.CTkLabel(master=win, text="~ Skin Cancer Detection ~",justify=LEFT)
TitleLb.configure(font=head_font1)
TitleLb.grid(row=2, column=1)

InfoLb= customtkinter.CTkLabel(master=win, text="Skin Cancer is totally curable if detected at an early stage.Wishing good luck!!", fg_color="dark slate gray")
InfoLb.grid(row=4, column=1, pady=10,rowspan=2, sticky=W)

#patient name label
NameLb = customtkinter.CTkLabel(master=win, text="Name of the Patient",justify=LEFT,fg_color="dodger blue")
NameLb.grid(row=6, column=0, pady=15, sticky=W)

#patient name entry
Name = StringVar()
NameEn = Entry(win, textvariable=Name)
NameEn.grid(row=6, column=1)

#upload button
Upload = Button(win, text="Upload Image", command=upload_image,bg="dark slate gray",fg="yellow")
Upload.grid(row=11, column=1,padx=10,pady=10)

#click button
Click = Button(win, text="Click an Image", command=click_image,bg="dark slate gray",fg="yellow")
Click.grid(row=15, column=1,padx=10,pady=10)

#predict button
predict = Button(win, text="Predict Disease", command=predict,bg="light sea green",fg="yellow")
predict.grid(row=17, column=1,padx=10,pady=20)

#predictionlabel
NameLb = customtkinter.CTkLabel(master=win, text="Prediction",justify=LEFT,fg_color="dodger blue")
NameLb.grid(row=19, column=0, pady=15, sticky=W)

#show output
showoutput = Text(win, height=1, width=40,bg="light goldenrod",fg="blue4")
showoutput.grid(row=19, column=1 , padx=10,pady=20)

#credits
creditsLb =customtkinter.CTkLabel(master=win, text="Made with love by icecream.",justify=LEFT)
creditsLb.configure(font=head_font3)
creditsLb.place(relx=0.5, rely=0.97, anchor=CENTER)

win.mainloop()