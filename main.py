import os
import json
import base64
import asyncio
import websockets
import pyaudio
import wave
import threading
import signal
import queue
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate."
)
VOICE = "alloy"
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono
RATE = 24000  # Increase from 16000 to 24000 Hz (OpenAI prefers this)
CHUNK = 1600  # Adjust buffer size proportionally
SPEAKING_THRESHOLD = 1500  # May need adjustment with the new rate

# Global variables
recording = False
playing = False
audio_queue = queue.Queue()
openai_ws = None
is_assistant_speaking = False
response_start_time = None
last_assistant_item = None

if not OPENAI_API_KEY:
    raise ValueError("Missing the OpenAI API key. Please set it in the .env file.")

# PyAudio setup
audio = pyaudio.PyAudio()


# Functions for audio handling
def convert_audio_for_openai(audio_data):
    """Convert PyAudio format to OpenAI's expected format (base64 encoded, correct format)"""
    # For this implementation, we need to convert from int16 PCM to ulaw
    # This is a simplified conversion - you may need a more accurate conversion
    return base64.b64encode(audio_data).decode("utf-8")


def convert_audio_from_openai(base64_audio):
    """Convert OpenAI's audio format back to PyAudio format"""
    return base64.b64decode(base64_audio)


def is_speaking(audio_data):
    """Simple voice activity detection"""
    # Convert bytes to int16 samples
    int_data = []
    for i in range(0, len(audio_data), 2):
        if i + 1 < len(audio_data):
            sample = int.from_bytes(
                audio_data[i : i + 2], byteorder="little", signed=True
            )
            int_data.append(sample)

    # Calculate RMS (Root Mean Square)
    if len(int_data) == 0:
        return False

    rms = sum([abs(x) for x in int_data]) / len(int_data)
    return rms > SPEAKING_THRESHOLD


def record_audio():
    """Record audio from microphone and send to OpenAI"""
    global recording, openai_ws, is_assistant_speaking

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Recording started. Speak into the microphone.")

    while recording:
        try:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)

            # Send only if we have an active OpenAI connection
            if openai_ws and openai_ws.open:
                # Check if the user is speaking
                user_is_speaking = is_speaking(audio_data)

                # If the assistant is speaking and the user starts speaking, trigger interruption
                if is_assistant_speaking and user_is_speaking:
                    loop.run_until_complete(handle_interruption())

                audio_base64 = convert_audio_for_openai(audio_data)

                # Use the thread's event loop to send the message
                loop.run_until_complete(
                    openai_ws.send(
                        json.dumps(
                            {"type": "input_audio_buffer.append", "audio": audio_base64}
                        )
                    )
                )
        except Exception as e:
            print(f"Error in recording: {e}")

    stream.stop_stream()
    stream.close()
    print("Recording stopped.")


def play_audio():
    """Play audio from queue"""
    global playing, audio_queue

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK,
    )

    print("Playback started.")

    while playing:
        try:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                stream.write(audio_data)
            else:
                time.sleep(0.1)  # Small sleep to prevent CPU hogging
        except Exception as e:
            print(f"Error in playback: {e}")

    stream.stop_stream()
    stream.close()
    print("Playback stopped.")


async def handle_interruption():
    """Handle user interruption while assistant is speaking"""
    global is_assistant_speaking, response_start_time, last_assistant_item

    if is_assistant_speaking and last_assistant_item and response_start_time:
        elapsed_time = int((time.time() - response_start_time) * 1000)  # Convert to ms
        print(f"User interrupted! Truncating assistant response after {elapsed_time}ms")

        truncate_event = {
            "type": "conversation.item.truncate",
            "item_id": last_assistant_item,
            "content_index": 0,
            "audio_end_ms": elapsed_time,
        }
        await openai_ws.send(json.dumps(truncate_event))

        # Clear the audio queue to stop current playback
        while not audio_queue.empty():
            audio_queue.get()

        is_assistant_speaking = False
        last_assistant_item = None
        response_start_time = None


async def openai_connection():
    """Maintain WebSocket connection to OpenAI and process messages"""
    global openai_ws, is_assistant_speaking, response_start_time, last_assistant_item

    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(uri, extra_headers=headers) as ws:
        openai_ws = ws
        print("Connected to OpenAI Realtime API")

        # Initialize session
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "echo",  # Make sure this is set to nova
                "instructions": SYSTEM_MESSAGE
                + " Speak quickly and with enthusiasm.",  # Add speed instruction
                "modalities": ["text", "audio"],
                "temperature": 0.8,
                # Remove speech_speed - it's not supported
            },
        }
        print("Sending session update:", json.dumps(session_update))
        await ws.send(json.dumps(session_update))

        try:
            # Process incoming messages from OpenAI
            async for message in ws:
                response = json.loads(message)

                if response.get("type") in [
                    "error",
                    "response.content.done",
                    "rate_limits.updated",
                    "response.done",
                    "session.created",
                ]:
                    print(f"Received event: {response['type']}", response)

                # Handle audio responses
                if (
                    response.get("type") == "response.audio.delta"
                    and "delta" in response
                ):
                    audio_data = convert_audio_from_openai(response["delta"])
                    audio_queue.put(audio_data)

                    # Track assistant speaking state
                    if not is_assistant_speaking:
                        is_assistant_speaking = True
                        response_start_time = time.time()

                    # Track the item ID for potential interruption
                    if response.get("item_id"):
                        last_assistant_item = response["item_id"]

                # Handle end of assistant speaking
                if response.get("type") == "response.done":
                    is_assistant_speaking = False
                    response_start_time = None

        except websockets.exceptions.ConnectionClosed:
            print("Connection to OpenAI closed")
        except Exception as e:
            print(f"Error in OpenAI connection: {e}")
        finally:
            openai_ws = None


def start_conversation():
    """Start the conversation with OpenAI"""
    global recording, playing

    # Start the recording and playback threads
    recording = True
    playing = True

    record_thread = threading.Thread(target=record_audio)
    play_thread = threading.Thread(target=play_audio)

    record_thread.daemon = True
    play_thread.daemon = True

    record_thread.start()
    play_thread.start()

    # Start the asyncio event loop for OpenAI connection
    asyncio.run(openai_connection())

    # When the OpenAI connection closes, stop recording and playback
    recording = False
    playing = False

    record_thread.join()
    play_thread.join()


def handle_exit(sig, frame):
    """Handle exit signals gracefully"""
    global recording, playing, audio

    print("\nShutting down...")
    recording = False
    playing = False

    # Clean up PyAudio
    audio.terminate()

    exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def main():
    """Main function"""
    print("Terminal-based OpenAI Voice Assistant")
    print("Press Ctrl+C to exit")
    print("Starting conversation...")

    start_conversation()


if __name__ == "__main__":
    main()
