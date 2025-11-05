from celery import Celery
import os
import torch
from transformers import pipeline
import scipy.io.wavfile

# Configure Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Initialize the synthesis pipeline globally.
# This is a heavy object, so we want to load it only once per worker process.
print("Initializing music generation pipeline...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
synthesiser = pipeline(
    "text-to-audio",
    "facebook/musicgen-small",
    device=device
)
print("Pipeline initialized.")


@celery_app.task(name='create_music_task')
def create_music_task(prompt: str):
    """
    A Celery task to generate music using the real model.
    """
    print(f"Received music generation task for prompt: '{prompt}'")

    # Generate music using the pre-loaded pipeline
    print("Generating music...")
    music = synthesiser(prompt, forward_params={"do_sample": True})

    # Prepare the output directory and filename
    output_dir = "music_results"
    os.makedirs(output_dir, exist_ok=True)
    task_id = create_music_task.request.id
    output_filename = os.path.join(output_dir, f"{task_id}.wav")

    # Save the audio file
    print(f"Saving audio to {output_filename}")
    audio_data = music["audio"]
    sampling_rate = music["sampling_rate"]
    scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_data)

    print(f"Music generation complete for task {task_id}.")

    # Return the path to the generated file
    return output_filename
