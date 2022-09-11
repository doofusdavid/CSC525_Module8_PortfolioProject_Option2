from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import os

def sayWords(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
r = sr.Recognizer()

with sr.Microphone() as source:
    print("Adjusting for ambient noise (1 second)")
    r.adjust_for_ambient_noise(source)         # listen for 1 second to calibrate the energy threshold for ambient noise levels

    i = 0
    while True:
        print("listening")
        audio_input = r.listen(source)                   # now when we listen, the energy threshold is already set to a good value, and we can reliably catch speech right away
        print("done")
        try:
            text = r.recognize_google(audio_input)
        except UnknownValueError:
            print("I didn't get that, please try again")
            continue
        if text == "exit":
            print("Goodbye")
            sayWords("Goodbye")
            break

        print("You: {}".format(text))

        input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if i > 0 else input_ids

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        sayWords(output)
        print(">> Bot: {}".format(output))
