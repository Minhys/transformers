from transformers import pipeline
import scipy.io.wavfile
import torch

print("Initializing pipeline...")
# Explicitly specify the device to use GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

synthesiser = pipeline(
    "text-to-audio",
    "facebook/musicgen-small",
    device=device
)

prompt = "A sad jazz piece with a slow saxophone solo and gentle piano."

print(f"Generating music for prompt: '{prompt}'")
# The generation process can be lengthy, especially on CPU.
# The model will be downloaded on the first run.
# We increase max_new_tokens to generate a longer audio clip.
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 512})

output_filename = "generated_music_2.wav"
print(f"Saving audio to '{output_filename}'...")

# The output from the pipeline is a dictionary containing the audio waveform and the sampling rate.
audio_data = music["audio"]
sampling_rate = music["sampling_rate"]

scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_data)

print("Script finished successfully.")
