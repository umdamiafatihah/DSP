import streamlit as st
import re

def normal_text(content: str, style=""):
    bold_pattern = re.compile(r"\*\*(.*?)\*\*")
    content = bold_pattern.sub(r"<b>\1</b>", content)
    container = st.container()
    container.markdown(f"<div class='text' style='{style}'><span>{content}</span></div>", unsafe_allow_html=True)

def ordered_list(items):
    container = st.container()
    bold_pattern = r"\*\*(.*?)\*\*"
    list_items = []
    for item in items:
        bold_matches = re.findall(bold_pattern, item)
        for bold_word in bold_matches:
            item = item.replace(f"**{bold_word}**", f"<b>{bold_word}</b>")
        list_items.append(f"<li><span>{item}</span></li>")

    concat = "".join(list_items)
    container.markdown(f"<div class=text><ol>{concat}</ol><div>", unsafe_allow_html=True)

def subtitle(content:str, size=2, style=""):
    container = st.container()
    container.markdown(f"<div class='text h{size}'><span>{content}<span></div>", unsafe_allow_html=True)

def image(img_path: str, parent=None, title=None, alt=None, style=""):
    if parent is not None:
        container = parent.container()
    else:
        container = st.container()
    elem_list = [f"<span class='h5 mb'>{title}</span>" if title is not None else None, f"<img src='app/static/{img_path}' style={style} alt='rsna logo'/>"]
    cleaned = [elem for elem in elem_list if elem is not None]
    container.markdown(f"<div class=dataset-provider>{''.join(cleaned)}</div>", unsafe_allow_html=True)

def vertical_gap(size:int):
    for i in range(size):
        st.markdown("<break>  \n", unsafe_allow_html=True)

def info_card(highlight:str, highlight_desc:str, parent=None):
    if parent is not None:
        container = parent.container()
    else:
        container = st.container()
    
    container.markdown(f"<div class=info-card-container><div class=info-card><span classname=h4>{highlight}</span><span class=text>{highlight_desc}</span></div></div>", unsafe_allow_html=True)