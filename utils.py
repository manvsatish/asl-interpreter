from gtts import gTTS
import os
import platform

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "output.mp3"
    tts.save(filename)
    
    system = platform.system()
    if system == "Windows":
        os.system(f'start {filename}')
    elif system == "Darwin":
        os.system(f'afplay {filename}')
    else:
        os.system(f'mpg123 {filename}')