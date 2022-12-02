import sys

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import requests
from bs4 import BeautifulSoup
from googletrans import Translator
def check_url(url):
    r = requests.head(url)
    return r.status_code == 200

def get_text_from_site(url):
    try:
        response = requests.get(url)
        bs = BeautifulSoup(response.text, 'html.parser')
        part_of_text = bs.find('div', 'c-detail text').text
        index = part_of_text.lower().find("приметы")
        i = part_of_text.find(":", index)
        result_sentence = part_of_text[i+1:part_of_text.find("\n", i+1)].replace(".", "")
        translator = Translator()
        translation = translator.translate(result_sentence, src='ru', dest='en')
        result_sentence = translation.text
    except:
        result_sentence = ""
    return result_sentence

def generate_arcane(text):
    model_id = "nitrosocke/Arcane-Diffusion"
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cpu")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    prompt = "arcane style, " + text
    images = pipe(prompt).images
    return images[0]
    #for image in images:
        #image.save(f"./result{images.index(image)}.png")

if __name__ == '__main__':
    st.title('Внимание, розыск в Аркейн!')
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    text_input = st.text_input(
        "Вставьте ссылку из Следственного комитета РФ по городу Москве сюда 👇"
    )
    if text_input != "":
        if check_url(text_input):
            text = get_text_from_site(text_input)
            if(text != ""):
                print(text)
                img = generate_arcane(text)
                st.image(img)
            else:
                st.text("Invalid URL")