import json
import time
from pathlib import Path
import mlflow
import torch
import torchaudio
from faster_whisper import WhisperModel
from jiwer import wer, cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_FILE = PROJECT_DIR / "data/benchmark/metadata.json"
AUDIO_FOLDER = PROJECT_DIR / "data/benchmark/audio"
RESULTS_FOLDER = PROJECT_DIR / "experiments"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "darija_asr_benchmark"

# IMPORTANT:
# The benchmark sends data to the MLflow SERVER.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WHISPER_DECODE_PARAMS = {
    "language": "ar",
    "task": "transcribe",
    "beam_size": 5,
    "vad_filter": True,
}

# Models to compare
MODELS = [
    # {
    #     "name": "whisper-small",
    #     "type": "faster-whisper",
    #     "model_id": "small",
    # },
    # {
    #     "name": "whisper-medium",
    #     "type": "faster-whisper",
    #     "model_id": "medium",
    # },
    {
        "name": "whisper-large-v3",
        "type": "faster-whisper",
        "model_id": "large-v3",
    },
#     {
#         "name": "whisper-large-v3-turbo",
#         "type": "faster-whisper",
#         "model_id": "large-v3-turbo",
#     },
#     {
#         "name": "algerian-dialect",
#         "type": "transformers",
#         "model_id": "MohammedNasri/whisper-algerian-dialect",
#     },
#      {
#     "name": "algerian-darja-stage1",
#     "type": "transformers-peft",
#     "model_id": "attoucheaziz/whisper-algerian-darja-stage1",
# },
# {
#     "name": "algerian-darja-small",
#     "type": "transformers-peft",
#     "model_id": "touati-kamel/whisper-algerian-darja-small",
# },
]


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 70)
print("MLflow configuration")
print("=" * 70)
print(f"Tracking URI : {mlflow.get_tracking_uri()}")
print(f"Experiment   : {EXPERIMENT_NAME}")
print(f"Device       : {DEVICE}")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):
    """
    Supports:
      - JSON array
      - JSONL
    """

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # JSON array
    if content.startswith("["):
        return json.loads(content)

    # JSONL
    data = []

    for line in content.splitlines():
        line = line.strip()

        if line:
            data.append(json.loads(line))

    return data


dataset = load_dataset(DATASET_FILE)

print(f"\nLoaded {len(dataset)} samples")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Minimal normalization.

    IMPORTANT:
    We do NOT transliterate Arabic/Darija.
    We keep French/English/Arabic content intact.
    """

    if text is None:
        return ""

    text = str(text)

    # Normalize whitespace
    text = " ".join(text.split())

    # Lowercase
    text = text.lower()

    return text.strip()


# ============================================================
# LOAD FASTER-WHISPER MODEL
# ============================================================

def load_faster_whisper(model_id):

    compute_type = "float16" if DEVICE == "cuda" else "int8"

    print(f"\nLoading Faster-Whisper: {model_id}")
    print(f"Device: {DEVICE}")
    print(f"Compute type: {compute_type}")

    model = WhisperModel(
        model_id,
        device=DEVICE,
        compute_type=compute_type,
    )

    return model


# ============================================================
# LOAD TRANSFORMERS MODEL
# ============================================================

def load_transformers_model(model_id):

    print(f"\nLoading Transformers model: {model_id}")

    processor = WhisperProcessor.from_pretrained(model_id)

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id
    )

    model = model.to(DEVICE)

    model.eval()

    return processor, model


def load_peft_model(model_id):
    from peft import PeftConfig, PeftModel

    print(f"\nLoading Whisper PEFT adapter: {model_id}")
    config = PeftConfig.from_pretrained(model_id)
    base_model_id = config.base_model_name_or_path
    if not base_model_id:
        raise ValueError(f"Adapter {model_id} does not specify a base model")

    # Adapter repositories may omit the processor files.
    try:
        processor = WhisperProcessor.from_pretrained(model_id)
    except OSError:
        processor = WhisperProcessor.from_pretrained(base_model_id)

    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, model_id)
    model = model.to(DEVICE)
    model.eval()
    return processor, model


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(audio_path):

    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Convert stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample -> 16 kHz
    if sample_rate != 16000:

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )

        waveform = resampler(waveform)

    # Convert to numpy
    audio = waveform.squeeze(0).numpy()

    return audio


# ============================================================
# TRANSCRIBE WITH FASTER-WHISPER
# ============================================================

def transcribe_faster_whisper(model, audio_path):

    segments, info = model.transcribe(
        str(audio_path),
        **WHISPER_DECODE_PARAMS,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()


# ============================================================
# TRANSCRIBE WITH TRANSFORMERS
# ============================================================

def transcribe_transformers(processor, model, audio_path):

    audio = load_audio(audio_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    # Match the audio encoder, including when PEFT adapter weights use
    # a different dtype from the underlying Whisper model.
    encoder_weight = model.get_encoder().conv1.weight
    input_features = inputs.input_features.to(
        device=encoder_weight.device,
        dtype=encoder_weight.dtype,
    )

    with torch.no_grad():

        predicted_ids = model.generate(
            input_features
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    return transcription.strip()


# ============================================================
# RUN MODEL BENCHMARK
# ============================================================

def benchmark_model(model_config):

    model_name = model_config["name"]
    model_type = model_config["type"]
    model_id = model_config["model_id"]

    print("\n")
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = None
    processor = None

    load_start = time.perf_counter()

    if model_type == "faster-whisper":

        model = load_faster_whisper(model_id)

    elif model_type == "transformers":

        processor, model = load_transformers_model(
            model_id
        )

    elif model_type == "transformers-peft":
        processor, model = load_peft_model(model_id)

    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")

    load_time = time.perf_counter() - load_start

    print(f"Model loading time: {load_time:.2f} seconds")

    # --------------------------------------------------------
    # MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(run_name=model_name) as run:

        print(f"\nMLflow Run ID: {run.info.run_id}")

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model_name": model_name,
                "model_type": model_type,
                "model_id": model_id,
                "device": DEVICE,
                "dataset": str(DATASET_FILE),
                "num_samples": len(dataset),
            }
        )

        if model_type == "faster-whisper":
            mlflow.log_params(WHISPER_DECODE_PARAMS)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        total_reference = []
        total_prediction = []

        sample_results = []

        total_inference_time = 0

        successful_samples = 0

        # ----------------------------------------------------
        # Process every audio
        # ----------------------------------------------------

        for index, item in enumerate(dataset):

            # Expected metadata:
            #
            # {
            #     "audio": "SAPS-01-START.wav",
            #     "text": "..."
            # }
            #
            # OR:
            #
            # {
            #     "file": "SAPS-01-START.wav",
            #     "text": "..."
            # }

            audio_filename = (
                item.get("audio")
                or item.get("file")
                or item.get("audio_file")
            )

            reference_text = (
                item.get("text")
                or item.get("transcript")
                or item.get("reference")
            )

            if not audio_filename:
                print(
                    f"[{index + 1}/{len(dataset)}] "
                    f"Skipping: no audio filename"
                )
                continue

            if reference_text is None:
                print(
                    f"[{index + 1}/{len(dataset)}] "
                    f"Skipping: no reference transcript"
                )
                continue

            audio_path = AUDIO_FOLDER / audio_filename

            if not audio_path.exists():

                print(
                    f"[{index + 1}/{len(dataset)}] "
                    f"Audio not found: {audio_path}"
                )

                continue

            print(
                f"\n[{index + 1}/{len(dataset)}] "
                f"{audio_filename}"
            )

            # ------------------------------------------------
            # Transcription
            # ------------------------------------------------

            start_time = time.perf_counter()

            try:

                if model_type == "faster-whisper":

                    prediction = transcribe_faster_whisper(
                        model,
                        audio_path,
                    )

                else:

                    prediction = transcribe_transformers(
                        processor,
                        model,
                        audio_path,
                    )

            except Exception as e:

                print(f"ERROR: {e}")

                prediction = ""

            inference_time = (
                time.perf_counter() - start_time
            )

            total_inference_time += inference_time

            print(f"Reference : {reference_text}")
            print(f"Prediction: {prediction}")

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            reference_normalized = normalize_text(
                reference_text
            )

            prediction_normalized = normalize_text(
                prediction
            )

            # ------------------------------------------------
            # Calculate per sample metrics
            # ------------------------------------------------

            try:

                sample_wer = wer(
                    reference_normalized,
                    prediction_normalized,
                )

            except Exception:

                sample_wer = 1.0

            try:

                sample_cer = cer(
                    reference_normalized,
                    prediction_normalized,
                )

            except Exception:

                sample_cer = 1.0

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            result = {
                "audio": audio_filename,
                "reference": reference_text,
                "prediction": prediction,
                "wer": sample_wer,
                "cer": sample_cer,
                "inference_time": inference_time,
            }

            sample_results.append(result)

            total_reference.append(
                reference_normalized
            )

            total_prediction.append(
                prediction_normalized
            )

            successful_samples += 1

            print(
                f"WER: {sample_wer:.4f} | "
                f"CER: {sample_cer:.4f} | "
                f"Time: {inference_time:.2f}s"
            )

        # ----------------------------------------------------
        # Overall metrics
        # ----------------------------------------------------

        if successful_samples > 0:

            overall_wer = wer(
                total_reference,
                total_prediction,
            )

            overall_cer = cer(
                total_reference,
                total_prediction,
            )

        else:

            overall_wer = 1.0
            overall_cer = 1.0

        average_inference_time = (
            total_inference_time / successful_samples
            if successful_samples > 0
            else 0
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                "wer": overall_wer,
                "cer": overall_cer,
                "total_inference_time": total_inference_time,
                "average_inference_time": average_inference_time,
                "successful_samples": successful_samples,
                "model_load_time": load_time,
            }
        )

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        result_file = (
            RESULTS_FOLDER
            / f"{model_name}_results.json"
        )

        with open(
            result_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                sample_results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ----------------------------------------------------
        # Log result JSON to MLflow
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(result_file),
            artifact_path="predictions",
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        print(f"Samples       : {successful_samples}")
        print(f"WER           : {overall_wer:.4f}")
        print(f"CER           : {overall_cer:.4f}")
        print(
            f"Avg time      : "
            f"{average_inference_time:.2f}s"
        )
        print(
            f"Total time    : "
            f"{total_inference_time:.2f}s"
        )

        print(
            f"Results saved : {result_file}"
        )

        print(
            f"MLflow Run ID : {run.info.run_id}"
        )

        print("-" * 70)

    # --------------------------------------------------------
    # Free GPU memory
    # --------------------------------------------------------

    del model
    del processor

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "wer": overall_wer,
        "cer": overall_cer,
        "avg_time": average_inference_time,
        "total_time": total_inference_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DARija / French / English ASR BENCHMARK")
    print("=" * 70)

    print(f"Dataset : {DATASET_FILE}")
    print(f"Audio   : {AUDIO_FOLDER}")
    print(f"Device  : {DEVICE}")
    print(f"MLflow  : {MLFLOW_TRACKING_URI}")

    results = []

    # --------------------------------------------------------
    # Run all models
    # --------------------------------------------------------

    for model_config in MODELS:

        try:

            result = benchmark_model(
                model_config
            )

            results.append(result)

        except Exception as e:

            print("\n")
            print("=" * 70)
            print(
                f"FAILED: {model_config['name']}"
            )
            print("=" * 70)

            print(e)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    if not results:

        print("No models completed successfully.")

        return

    # Sort by WER
    results = sorted(
        results,
        key=lambda x: x["wer"],
    )

    print(
        f"{'Model':<30}"
        f"{'WER':<12}"
        f"{'CER':<12}"
        f"{'Avg Time':<12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['model']:<30}"
            f"{result['wer']:<12.4f}"
            f"{result['cer']:<12.4f}"
            f"{result['avg_time']:<12.2f}"
        )

    print("=" * 70)

    best_model = results[0]

    print(
        f"\nBEST MODEL BY WER: "
        f"{best_model['model']}"
    )

    print(
        f"WER: {best_model['wer']:.4f}"
    )

    print(
        "\nOpen MLflow UI:"
    )

    print(
        "http://127.0.0.1:5000"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
