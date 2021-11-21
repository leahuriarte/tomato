"remember pipreqs for requirements.txt"


from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st

import gc
gc.enable()

class Predict:
    def __init__(self, filename):

        st.title('Is Your Tomato Plant Healthy?')

        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()

        
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Submit a Photo Below:",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500))

    def get_prediction(self):

        if st.button('Classify It!'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            pred = pred.capitalize()
            percentage = probs[pred_idx]*100
            #st.header(f'My Prediction is {pred}, with a confidence of {percentage:.02f}%')
            if pred == "Lateblight":
                st.subheader('It looks like your tomato plant has late blight. Late blight is highly contagious and can kill your plant. To prevent late blight from spreading to the rest of your garden, you must destroy this plant.')
            elif pred == "Earlyblight":
                st.subheader("It looks like your tomato plant has early blight, a common fungal disease. To protect your garden, remove all potentially infected plant debris and destoy it. ")
            elif pred == "Septorialeafspot":
                st.subheader("It looks like your tomato plant has septoria leaf spot, a common fungal disease. This fungus can attack a wide range of plants, so remove and destroy any infected leaves to halt its spread.")
            elif pred == "Spidermites":
                st.subheader("It looks like your tomato plant has spider mites. Spider mites are rapid breeders, so it is essential you treat them early before their population rises, using insecticidal soap or neem oil.")
            elif pred == "Targetspot":
                st.subheader("It looks like your tomato plant has target spot, a disease caused by the fungus Corynespora cassiicola. Target spot can remain in plant residue for up to two years, so remove old plant disease at the end of the growing season. ")
            elif pred == "Yellowleafcurlvirus":
                st.subheader("it looks like your tomato plant has yellow lead curl virus, a DNA virus. The virus can infect a wide variety of plants, and is typically spread by whiteflies.")
            elif pred == "Leafmold":
                st.subheader("It looks like your tomato plant has leaf mold. Leaf mold is a fungal virus that is commonly developed by tomatoes in very humid environments. Remove all infected plant material from your garden.")
            elif pred == "Mosaicvirus":
                st.subheader("It looks like your tomato plant has mosaic virus. Mosaic virus has no cure, so remove and destoy all infected plant materials and do not compost them. Inspect nearby plants for signs of infection and remember to disinfect your garden tools.")
            else:
                st.subheader('Congratulations, it looks like your tomato plant is healthy!')
        #else: 
            #st.write(f'Click the button to classify') 

if __name__=='__main__':

    file_name='tomato.pkl'

    predictor = Predict(file_name)

    gc.collect()