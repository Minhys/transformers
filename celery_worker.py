from celery import Celery
import os
import torch
from transformers import pipeline
import scipy.io.wavfile

# A dictionary to hold our loaded models. This allows us to load them once
# per worker and reuse them across tasks.
PIPELINES = {}

def initialize_pipelines():
    """Loads all supported models into the global PIPELINES dictionary."""
    print("Initializing supported pipelines...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load MusicGen
    print("Loading facebook/musicgen-small...")
    PIPELINES["musicgen"] = pipeline(
        "text-to-audio",
        "facebook/musicgen-small",
        device=device
    )
    print("facebook/musicgen-small loaded.")

    # Placeholder for SongGeneration - we don't load it to avoid resource issues.
    # In a real production environment, you would load it here.
    PIPELINES["songgeneration"] = None
    print("Pipelines initialized.")

# Configure Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Initialize pipelines when the worker starts.
initialize_pipelines()


@celery_app.task(name='create_music_task')
def create_music_task(prompt: str, model_name: str = "musicgen"):
    """
    A Celery task to generate music using a specified model.
    """
    print(f"Received task for model '{model_name}' with prompt: '{prompt}'")

    if model_name not in PIPELINES:
        raise ValueError(f"Model '{model_name}' is not supported.")

    if model_name == "songgeneration":
        # This is the placeholder logic.
        print("Model 'songgeneration' is not fully integrated due to high resource requirements.")
        raise NotImplementedError("Model 'songgeneration' requires a dedicated GPU environment and is not available.")

    # Get the appropriate pipeline
    synthesiser = PIPELINES[model_name]

    # Generate music
    print(f"Generating music with {model_name}...")
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

    return output_filename
