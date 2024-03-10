# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:25:42 2024

@author: ASHISH KUMAR
"""


import nltk
from gtts import gTTS
import os
import pandas as pd


txt= 'Hi this is ashish'

language = 'en'

myobj = gTTS(text= txt, lang=language, slow=False)


myobj.save("read_article.mp3")

os.system("start read_article.mp3")
