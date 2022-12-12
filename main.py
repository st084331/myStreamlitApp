import validators
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

MODEL_ID = "nitrosocke/Arcane-Diffusion"


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def init_pipe():
    if torch.cuda.is_available():
        try:
            pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
        except:
            pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
            pipe = pipe.to("cpu")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
        pipe = pipe.to("cpu")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    return pipe


PIPE = init_pipe()


def generate_arcane(text, num_inference_steps=50, guidance_scale=7.5):
    return PIPE(prompt="arcane style, " + text, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale).images


def find_info(url):
    response = requests.get(url)
    bs = BeautifulSoup(response.text, 'html.parser')
    part_of_text = bs.find('div', 'c-detail text').text
    return part_of_text


def get_text_from_site(url):
    try:
        part_of_text = find_info(url)
    except:
        return [1, "Невозможно найти информацию. Попробуйте другую ссылку."]
    index = part_of_text.lower().find("приметы")
    if index != -1:
        i = part_of_text.find(":", index)
        if i != -1:
            result_sentence = part_of_text[i + 1:part_of_text.find("\n", i + 1)].replace(".", "")
            try:
                result_sentence = GoogleTranslator(source='ru', target='en').translate(result_sentence)
            except:
                return [2, "Перевод не завершился. Попробуйте еще раз или попробуйте другую ссылку."]
        else:
            return [1, "Невозможно найти информацию. Попробуйте другую ссылку."]
    else:
        return [1, "Невозможно найти информацию. Попробуйте другую ссылку."]

    return [0, result_sentence]


if __name__ == '__main__':
    st.title('Внимание, розыск в Аркейн!')

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    text_input = st.text_input(
        "Вставьте ссылку из Следственного комитета РФ по городу Москве сюда 👇"
    )

    num_inference_steps = st.slider("Число шагов (влияет на скорость работы)", value=50, min_value=1, max_value=800, step=1)

    guidance_scale = st.slider("Степень приверженности (больше или меньше фантазии)", value=7.5, min_value=1.1,
                               max_value=15.0, step=0.1)

    if st.button("Сгенерировать!"):

        if text_input != "":

            if validators.url(text_input):

                text_and_code = get_text_from_site(text_input)

                if not text_and_code[0]:
                    st.empty()
                    images = generate_arcane(text_and_code[1], num_inference_steps, guidance_scale)
                    st.image(images[0])
                else:
                    st.text(text_and_code[1])
            else:
                st.text("Это не ссылка. Попробуйте вставить ссылку.")
