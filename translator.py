from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import soundcard as sc 
import soundfile as sf 

output_file_name = "AudioCaptured.wav"
samplerate = 16000 #need to be 16k cuz the whisper asked it to be 16khz
model_size = "small"
record_sec = 3
model_name ='Helsinki-NLP/opus-mt-ja-en'

model_translator = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors = 'pt', padding = True, truncation = True)

    translation_ids = model_translator.generate(
        input_ids,
        max_length = 100,
        num_beams = 5,
        length_penalty = 1.0,
        no_repeat_ngram_size = 5,
        top_k = 50,
        top_p = 0.95,
        early_stopping = False,
        do_sample = True
    )

    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens = True)

    return translated_text


# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")
accumulated_Transcription = " "

try:
    print("Starting...")
    while True:

        with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback= True).recorder(samplerate=samplerate) as mic:
            data = mic.record(numframes=samplerate*record_sec)
            segments, info = model.transcribe(data[:,0], beam_size=5)

            for segment in segments:
                # print(segment.text)
                translated_text = translate_text(segment.text)
                print(translated_text)
                accumulated_Transcription += translated_text + "\n"

except KeyboardInterrupt:

    print("Stopping...")

    with open("log.txt", "w") as log_file:
        log_file.write(accumulated_Transcription)