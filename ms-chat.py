from email.mime import audio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gtts
from playsound import playsound
import speech_recognition as sr

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
r = sr.Recognizer()


# tts = gtts.gTTS("Hello, I am a chatbot. How are you?", lang="en")
# tts.save("output.mp3")
# playsound("output.mp3")

with sr.Microphone() as source:
    print("Adjusting for ambient noise (1 second)")
    r.adjust_for_ambient_noise(source)         # listen for 1 second to calibrate the energy threshold for ambient noise levels
    print("recording")
    audio_input = r.listen(source)                   # now when we listen, the energy threshold is already set to a good value, and we can reliably catch speech right away
    print("done")
    print(r.recognize_google(audio_input))

exit()
for i in range(5):
    text = input(">> User:")
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
    tts = gtts.gTTS(output)
    tts.save("output.mp3")
    playsound("output.mp3")
    print(">> Bot: {}".format(output))
