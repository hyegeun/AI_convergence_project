import streamlit as st
from PIL import Image
import os
import cv2
import image_sift as type_1
import numpy as np
import subprocess
import shutil
import glob

# 폴더 삭제하고 다시 만드는 함수 
def make_dir(new_folder):
    if os.path.isdir(new_folder):
        shutil.rmtree(new_folder)
        os.makedirs(new_folder)
    else:
        os.makedirs(new_folder)

# 업로드된 이미지를 OpenCV 이미지로 변환하는 함수
def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image) 
    # RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

# OpenCV 이미지를 Pillow 이미지로 변환하는 함수
def cv2_to_pil(cv2_image):
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    return pil_image


def save_query(images):
    # query_path = './unicom/data/aiproject/image_query'
    query_path = './image_query'
    make_dir(query_path)

    for idx, image in enumerate(images):
        image.convert("RGB").save(os.path.join(query_path, f"{idx:04d}.jpg"))



# 페이지 구성 설정
st.set_page_config(layout="wide")

# 페이지 전환을 위한 함수
def set_page(page):
    st.session_state.page = page

# 페이지 중앙으로 정렬
def center_alignment():
    """페이지를 가로 방향으로 중앙으로 정렬하고 미디어 쿼리를 적용합니다."""
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            @media (max-width: 1200px) {{
                .reportview-container .main .block-container{{max-width: 100%;}}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# 이미지를 업로드하는 함수
def upload_image():

    uploaded_files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files is not None:
        st.session_state.uploaded_images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]
        # if len(st.session_state.uploaded_images) == 1:
            # st.image(st.session_state.uploaded_images[0], caption='Uploaded Image', width = 600)
        if st.button('Next'):
            set_page('inging')
            st.experimental_rerun()


# st.set_page_config(layout="wide")


# 메인 페이지
def main_page():
    st.header('IS THIS A STOLEN IMAGE?', divider='rainbow')

    main_img = Image.open('./st_images/main_image.png')
    st.image(main_img)
    upload_image()


# 진행 중 페이지
def processing_page():
    st.header('Searching for stolen images.', divider='rainbow')
    # st.markdown('### 잠시만 기다려주세요.')
    
    process_img = Image.open('./st_images/processing_image.png')

    save_query(st.session_state.uploaded_images)
    cmd = 'torchrun unicom/retrieval.py --eval --dataset inshop --model_name ./unicom/output/sop_2.pt'
    with st.spinner('Wait for it...'):
        st.image(process_img)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
        stdout, stderr = process.communicate()

    set_page('result')
    st.experimental_rerun()


# 결과 이미지를 보여주는 함수
# @st.cache(allow_output_mutation=True)
def show_result_images():
    result_folder = "./result"  # 결과 이미지가 저장된 폴더
    result_images = os.listdir(result_folder)[:10]  # 결과 이미지 파일 목록을 가져옴
    images = []
    for image_file in result_images:
        image_path = os.path.join(result_folder, image_file)
        images.append(Image.open(image_path))
    return images


# 결과 페이지
def result_page():
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            display: flex; /* 탭 아이템들을 가로로 나란히 배치 */
            gap: 2px;
        }

        .stTabs [data-baseweb="tab"] {
            white-space: nowrap; /* 탭 아이템이 한 줄로 길어질 때 줄바꿈 방지 */
            gap: 1px;
            padding: 10px;
            font-size: 16px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
        }
    </style>
    """, unsafe_allow_html=True)


    with open('./css/result.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.header('Result of uploaded Images', divider='rainbow')

    query_list = sorted(os.listdir('result'))

    for pid, tab in zip(query_list, st.tabs(query_list)):
        with tab:
            st.subheader('Uploaded Image', divider='blue')
            st.image(Image.open(f'image_query/{pid}.jpg'), width=450)

            result_images = sorted(glob.glob(f'./result/{pid}/*.jpg'))
            st.subheader('Similar Images', divider='orange')
            num_cols = 2

            num_rows = len(result_images) // num_cols + (1 if len(result_images) % num_cols > 0 else 0)
            for row in range(num_rows):
                columns = st.columns(num_cols)  # 이미지를 행렬 형태로 배치
                for col in range(num_cols):
                    index = row * num_cols + col
                    if index < len(result_images):
                        columns[col].markdown(f'#### __Rank {index+1}__')
                        columns[col].image(result_images[index], width=450)
                st.divider()

    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:4rem;
        }
    </style>
    '''


center_alignment()
if st.button('🏠Home'):
    set_page('main')

# 애플리케이션 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'main'
    

# 페이지 라우팅
if st.session_state.page == 'main':
    main_page()
elif st.session_state.page == 'inging':
    processing_page()
elif st.session_state.page == 'result':
    result_page()
