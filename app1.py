import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import wikipedia

def app():
    st.header("Landmark Recognition")
    names =  ['Acropolis_of_Athens', 'Arc de Triomphe', 'Big_Ben', 'BlueMosque', 'Brandenburg_Gate', 'CN_Tower', 'Casa_Mila', 'Christ the Redeemer', 'Colosseum', 'Deoksugung', 'Eiffel_Tower', 'Forbidden_City', 'Gardens_by_the_Bay', 'Great_Wall_of_China', 'HollyWood Sign', 'Jerusalem', 'Leaning_Tower_of_Pisa', 'London_Eye', 'Marina Bay', 'Opera_House', 'Pantheon', 'Rialto_Bridge', 'Statue_of_Liberty_National_Monument', 'Taj Mahal', 'pyramid']

    pimage = st.file_uploader("Choose an image...", type="jpg")
    submit = st.button('Predict')
    if submit:
        if pimage is not None:
            file_bytes = np.asarray(bytearray(pimage.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            model = YOLO("best.pt")
            results = model(opencv_image)
            st.write(results[0].boxes.data)
            r = results[0].boxes.data
            r1 = r[-1]-1
            st.write(names[r1])
            t = wikipedia.summary(names[r1], sentences=10)
            st.write(t)


