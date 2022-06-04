from tkinter import *
import numpy as np
import pandas as pd
import customtkinter
from PIL import Image,ImageTk
import tkinter.font as tkFont


l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("./disease/Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("./disease/Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),Symptom6.get(),Symptom7.get(),Symptom8.get(),Symptom9.get(),Symptom10.get()]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),Symptom6.get(),Symptom7.get(),Symptom8.get(),Symptom9.get(),Symptom10.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),Symptom6.get(),Symptom7.get(),Symptom8.get(),Symptom9.get(),Symptom10.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

# gui_stuff------------------------------------------------------------------------------------

WIDTH=700
HEIGHT=900
root=customtkinter.CTk()
root.title("Dhoom Dhadaka Healthcare")
root.geometry(f"{WIDTH}x{HEIGHT}")
image = Image.open("./images/bg1.jpg").resize((WIDTH+650, HEIGHT+760))
root.bg_image = ImageTk.PhotoImage(image)
root.image_label =Label(master=root, image=root.bg_image)
root.image_label.place(relx=0.15, rely=0.92, anchor=CENTER)


head_font1=tkFont.Font(family="Times New Roman",size=30,weight="bold",slant="italic")
head_font2=tkFont.Font(family="Times New Roman",size=20,weight="bold",slant="italic")
head_font3=tkFont.Font(family="Arial",size=10)


# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Symptom6 = StringVar()
Symptom6.set(None)
Symptom7 = StringVar()
Symptom7.set(None)
Symptom8 = StringVar()
Symptom8.set(None)
Symptom9 = StringVar()
Symptom9.set(None)
Symptom10 = StringVar()
Symptom10.set(None)
Name = StringVar()

# Heading
# w2 = customtkinter.CTkLabel(master=root, justify=LEFT, text="Disease Predictor using Machine Learning", fg="white", bg="blue")
# w2.config(font=("Elephant", 30))
# w2.grid(row=1, column=0, columnspan=2, padx=100)
# w2 = customtkinter.CTkLabel(master=root, justify=LEFT, text="A Project by Yaswanth Sai Palaghat", fg="white", bg="blue")
# w2.config(font=("Aharoni", 30))
# w2.grid(row=2, column=0, columnspan=2, padx=100)

# labels
TitleLb =customtkinter.CTkLabel(master=root, text="~ Disease Prediction ~",justify=LEFT)
TitleLb.configure(font=head_font1)
TitleLb.grid(row=2, column=1)

NameLb = customtkinter.CTkLabel(master=root, text="Name of the Patient",justify=LEFT,fg_color="dodger blue")
NameLb.grid(row=6, column=0, pady=15, sticky=W)


S1Lb = customtkinter.CTkLabel(master=root, text="Symptom 1", fg_color="dark slate gray")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = customtkinter.CTkLabel(master=root, text="Symptom 2", fg_color="dark slate gray")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = customtkinter.CTkLabel(master=root, text="Symptom 3", fg_color="dark slate gray")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = customtkinter.CTkLabel(master=root, text="Symptom 4", fg_color="dark slate gray")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = customtkinter.CTkLabel(master=root, text="Symptom 5", fg_color="dark slate gray")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

S6Lb = customtkinter.CTkLabel(master=root, text="Symptom 6", fg_color="dark slate gray")
S6Lb.grid(row=12, column=0, pady=10, sticky=W)

S7Lb = customtkinter.CTkLabel(master=root, text="Symptom 7", fg_color="dark slate gray")
S7Lb.grid(row=13, column=0, pady=10, sticky=W)

S8Lb = customtkinter.CTkLabel(master=root, text="Symptom 8", fg_color="dark slate gray")
S8Lb.grid(row=14, column=0, pady=10, sticky=W)

S9Lb = customtkinter.CTkLabel(master=root, text="Symptom 9", fg_color="dark slate gray")
S9Lb.grid(row=15, column=0, pady=10, sticky=W)

S10Lb = customtkinter.CTkLabel(master=root, text="Symptom 10", fg_color="dark slate gray")
S10Lb.grid(row=16, column=0, pady=10, sticky=W)


lrLb = customtkinter.CTkLabel(master=root, text="DecisionTree", fg_color="DeepSkyBlue4")
lrLb.grid(row=20, column=0, pady=10,sticky=W)

destreeLb = customtkinter.CTkLabel(master=root, text="RandomForest", fg_color="DeepSkyBlue4")
destreeLb.grid(row=21, column=0, pady=10, sticky=W)

ranfLb = customtkinter.CTkLabel(master=root, text="NaiveBayes", fg_color="DeepSkyBlue4")
ranfLb.grid(row=22, column=0, pady=10, sticky=W)

# entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

S6En = OptionMenu(root, Symptom6,*OPTIONS)
S6En.grid(row=12, column=1)

S7En = OptionMenu(root, Symptom7,*OPTIONS)
S7En.grid(row=13, column=1)

S8En = OptionMenu(root, Symptom8,*OPTIONS)
S8En.grid(row=14, column=1)

S9En = OptionMenu(root, Symptom9,*OPTIONS)
S9En.grid(row=15, column=1)

S10En = OptionMenu(root, Symptom10,*OPTIONS)
S10En.grid(row=16, column=1)


dst = Button(root, text="DecisionTree", command=DecisionTree,bg="light sea green",fg="yellow")
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="Randomforest", command=randomforest,bg="light sea green",fg="yellow")
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes,bg="light sea green",fg="yellow")
lr.grid(row=10, column=3,padx=10)

predictionLb =customtkinter.CTkLabel(master=root, text="PREDICTION",justify=LEFT)
predictionLb.configure(font=head_font2)
predictionLb.grid(row=18, column=1)

creditsLb =customtkinter.CTkLabel(master=root, text="Made with love by icecream.",justify=LEFT)
creditsLb.configure(font=head_font3)
creditsLb.place(relx=0.5, rely=0.97, anchor=CENTER)

#textfileds
t1 = Text(root, height=1, width=40,bg="light goldenrod",fg="blue4")
t1.grid(row=20, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="light goldenrod",fg="blue4")
t2.grid(row=21, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="light goldenrod",fg="blue4")
t3.grid(row=22, column=1 , padx=10)


root.mainloop()
