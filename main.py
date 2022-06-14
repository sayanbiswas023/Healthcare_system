import sys
import os
from tkinter import *
import customtkinter
from PIL import Image,ImageTk
import tkinter.font as tkFont

WIDTH=550
HEIGHT=250


customtkinter.set_appearance_mode("System") 
customtkinter.set_default_color_theme("blue") 

window=customtkinter.CTk()
window.title("Dhoom Dhadaka Healthcare")
window.geometry(f"{WIDTH}x{HEIGHT}")
image = Image.open("./images/bg1.jpg").resize((WIDTH+450, HEIGHT+210))
window.bg_image = ImageTk.PhotoImage(image)
window.image_label =Label(master=window, image=window.bg_image)
window.image_label.place(relx=0.15, rely=0.92, anchor=CENTER)

head_font3=tkFont.Font(family="Arial",size=8)

def health_detect():
    os.system('python3 ./disease/health_detect.py')

def skin_cancer():
   os.system('python3 ./skin_cancer/skin_model.py')

def emotional_support():
    os.system('python3 ./Chatbot/app.py')

TitleLb = customtkinter.CTkLabel(master=window, text="WELCOME TO DHOOM DHADAKA HEALTHCARE", justify=LEFT)
TitleLb.pack(pady=12, padx=10)

btn_disease = customtkinter.CTkButton(master=window,corner_radius=6,  text="Predict disease", command=health_detect)
btn_disease.place(relx=0.5, rely=0.3, anchor=CENTER)

btn_check_skin_cancer = customtkinter.CTkButton(master=window,corner_radius=6,  text="Check Skin Cancer",command=skin_cancer)
btn_check_skin_cancer.place(relx=0.5, rely=0.5, anchor=CENTER)

btn_emotional_support = customtkinter.CTkButton(master=window, corner_radius=6, text="Need Emotional Support ?",command=emotional_support)
btn_emotional_support.place(relx=0.5, rely=0.7, anchor=CENTER)

creditsLb =customtkinter.CTkLabel(master=window, text="Made with love by icecream.",justify=LEFT)
creditsLb.configure(font=head_font3)
creditsLb.place(relx=0.5, rely=0.95, anchor=CENTER)

window.mainloop()