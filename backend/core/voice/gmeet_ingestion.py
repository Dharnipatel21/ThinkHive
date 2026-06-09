import os
import tempfile
from .transcriber import VoiceTranscriber

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Warning: google-api-python-client not installed. Gmeet ingestion disabled.")


class GmeetIngestion:
    """
    Fetches Google Meet recordings from Google Drive,
    transcribes them using Whisper, and prepares the
    transcript for ingestion into the RAG pipeline.
    """

    def __init__(self, google_credentials=None):
        """
        Args:
            google_credentials: Google OAuth2 credentials object
                                 obtained from google-auth-oauthlib
        """
        self.credentials = google_credentials
        self.drive_service = None
        self.transcriber = VoiceTranscriber(model_size="base")

        if GOOGLE_API_AVAILABLE and google_credentials:
            try:
                self.drive_service = build("drive", "v3", credentials=google_credentials)
            except Exception as e:
                print(f"Google Drive API init failed: {e}")

    def fetch_recording(self, file_id: str) -> str:
        """
        Download a Google Meet recording from Drive to a temp file.

        Args:
            file_id: Google Drive file ID of the recording

        Returns:
            local path to downloaded audio file
        """
        if not self.drive_service:
            raise RuntimeError("Google Drive service not initialised")

        try:
            import io

            request = self.drive_service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            # Save to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(buffer.getvalue())
            tmp.close()

            return tmp.name

        except Exception as e:
            raise RuntimeError(f"Failed to download recording: {e}")

    def ingest_meeting(self, file_id: str, org_id: str, meeting_metadata: dict) -> dict:
        """
        Full ingestion of a Google Meet recording.
        Downloads, transcribes, and returns ready-to-index content.

        Args:
            file_id: Google Drive file ID
            org_id: organisation ID
            meeting_metadata: dict with meeting_title, meeting_date, attendees

        Returns:
            dict with:
            {
                transcript: str,
                language: str,
                document_metadata: dict (ready for MongoDB),
                success: bool
            }
        """
        tmp_path = None

        try:
            # Download recording
            tmp_path = self.fetch_recording(file_id)

            # Transcribe
            transcription = self.transcriber.transcribe(tmp_path)

            if not transcription["success"]:
                return {
                    "transcript": "",
                    "language": "en",
                    "document_metadata": {},
                    "success": False,
                    "error": transcription.get("error", "Transcription failed")
                }

            # Build metadata for MongoDB document record
            document_metadata = {
                "filename": f"meeting_{meeting_metadata.get('meeting_date', 'unknown')}.txt",
                "file_type": "meeting_transcript",
                "org_id": org_id,
                "meeting_title": meeting_metadata.get("meeting_title", "Untitled Meeting"),
                "meeting_date": meeting_metadata.get("meeting_date"),
                "attendees": meeting_metadata.get("attendees", []),
                "google_drive_file_id": file_id,
                "language": transcription["language"],
                "word_count": len(transcription["transcript"].split())
            }

            return {
                "transcript": transcription["transcript"],
                "language": transcription["language"],
                "document_metadata": document_metadata,
                "success": True
            }

        except Exception as e:
            return {
                "transcript": "",
                "language": "en",
                "document_metadata": {},
                "success": False,
                "error": str(e)
            }

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
