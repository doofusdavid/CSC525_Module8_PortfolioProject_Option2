"""
Portfolio Project - Option 2
David Edwards
CSC 525 - Principles of Machine Learning
Dr. Issac Gang
9/11/2022
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import os


def say_words(phrase: str) -> None:
    """
    Takes in a phrase and speaks it out loud

    :type phrase: str
    :rtype: None
    :param phrase: string containing the phrase to be spoken
    """
    tts = gTTS(text=phrase, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")
    os.remove("output.mp3")


def main():
    # Disable parallelism, as it causes issues with playsound
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load the model and tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Initiate the recognizer
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting for ambient noise (1 second)")
        print("Speak the word 'exit' to end the program")
        # listen for 1 second to calibrate the energy threshold for ambient noise levels
        r.adjust_for_ambient_noise(source)

        i = 0
        while True:
            print("listening")
            audio_input = r.listen(source)
            print("done")
            try:
                text = r.recognize_google(audio_input)
            except sr.UnknownValueError:
                print("I didn't get that, please try again")
                continue
            if text == "exit":
                print("Goodbye")
                say_words("Goodbye")
                break

            # Display what the user said
            print("You: {}".format(text))

            # Encode the user input, add the eos_token and return a tensor in Pytorch
            input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
            bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if i > 0 else input_ids

            # Ensure we keep the chat history to inform the next response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode the response
            output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            # Display, and say, the response
            say_words(output)
            print(">> Bot: {}".format(output))


if __name__ == "__main__":
    main()
