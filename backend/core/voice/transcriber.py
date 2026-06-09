import os
import tempfile

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not installed. Voice transcription disabled.")


class VoiceTranscriber:
    """
    Transcribes audio files to text using OpenAI Whisper.
    Supports MP3, MP4, WAV, M4A, WEBM audio formats.
    """

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: whisper model size
                        tiny (fastest), base, small, medium, large (most accurate)
                        Use "base" for balance of speed and accuracy
        """
        self.model = None
        self.model_size = model_size

        if WHISPER_AVAILABLE:
            try:
                print(f"Loading Whisper {model_size} model...")
                self.model = whisper.load_model(model_size)
                print("Whisper model loaded.")
            except Exception as e:
                print(f"Whisper model load failed: {e}")

    def transcribe(self, audio_file_path: str) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_file_path: path to audio file on disk

        Returns:
            dict with:
            {
                transcript: str,
                language: str (detected language code),
                duration_seconds: float,
                success: bool
            }
        """
        if not self.model:
            return {
                "transcript": "",
                "language": "en",
                "duration_seconds": 0.0,
                "success": False,
                "error": "Whisper model not available"
            }

        if not os.path.exists(audio_file_path):
            return {
                "transcript": "",
                "language": "en",
                "duration_seconds": 0.0,
                "success": False,
                "error": f"Audio file not found: {audio_file_path}"
            }

        try:
            result = self.model.transcribe(
                audio_file_path,
                fp16=False,           # Use fp32 for CPU
                verbose=False
            )

            return {
                "transcript": result["text"].strip(),
                "language": result.get("language", "en"),
                "duration_seconds": 0.0,
                "success": True
            }

        except Exception as e:
            print(f"Transcription error: {e}")
            return {
                "transcript": "",
                "language": "en",
                "duration_seconds": 0.0,
                "success": False,
                "error": str(e)
            }

    def transcribe_bytes(self, audio_bytes: bytes, file_extension: str = ".wav") -> dict:
        """
        Transcribe audio from bytes (e.g. from API upload).

        Args:
            audio_bytes: raw audio bytes
            file_extension: file extension to use when saving temp file

        Returns:
            same dict as transcribe()
        """
        with tempfile.NamedTemporaryFile(
            suffix=file_extension,
            delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = self.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

        return result
