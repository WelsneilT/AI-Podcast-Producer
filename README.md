# 🎙️ AI Storybook Podcast Generator

Welcome to the AI Storybook Podcast Generator! This Django-powered web application brings your creative crossover ideas to life. Simply provide a basic plotline, select two of your favorite cartoon characters, and watch as AI generates a complete, multi-chapter storybook podcast with unique illustrations, narration, and background music.

[Video demo](https://youtu.be/JFrOie6D3fw?si=WiYDfQ30spqzlN1y)


![Final Product Showcase](https://github.com/WelsneilT/AI-Podcast-Producer/blob/main/django_podcast_project/my_story_project/static/images/ui/AI%20Podcast.gif) 

---

## ✨ Features

-   **Dynamic Story Generation**: Creates a complete 5-chapter story from a simple user prompt.
-   **AI-Powered Scriptwriting**: Utilizes the Groq API with Llama 3 for incredibly fast and creative story and prompt generation.
-   **Unique AI Illustrations**: Generates a unique, watercolor-style illustration for each chapter using a locally run Stable Diffusion XL model.
-   **AI Voice Narration**: Converts chapter text into natural-sounding speech using a separate, self-hosted TTS server.
-   **Personalized Voice Cloning**: The narration voice is a custom clone of TWICE's Mina, created using a sample audio file (`mina_voice.wav`) for a unique and personal touch.
-   **Atmospheric Audio**: Mixes the generated narration with background music for a complete podcast experience.
-   **Asynchronous Task Processing**: Powered by Celery and Redis to handle the long-running AI generation tasks in the background without freezing the app.
-   **Real-time Progress Updates**: The frontend polls the server to show the user the current status of their story creation process.
-   **Robust Illustration Strategy**: Implements a "Director's Cut" logic to ensure high-quality, character-accurate images by alternating focus and using context-aware prompts.

---

## 🚀 Tech Stack

-   **Backend**: Django, Celery
-   **AI Scripting**: Groq API (Llama 3)
-   **AI Image Generation**: Diffusers, PyTorch (Stable Diffusion XL + Refiner)
-   **AI Voice Synthesis**: Self-hosted TTS Server (e.g., CoquiTTS, XTTS)
-   **Message Broker**: Redis
-   **Database**: SQLite 3 (default)
-   **Frontend**: HTML, CSS, Vanilla JavaScript

---

## 🏛️ How It Works

The application follows a sophisticated, asynchronous workflow orchestrated by Celery.

1.  **User Input**: The user submits a plotline and two characters via the Django frontend.
2.  **Celery Chain Triggered**: The Django view initiates a Celery `chain`, sending the two main tasks to the Redis broker.
3.  **Task 1: `create_full_story_task`**:
    -   Calls the Groq API to generate a compelling 5-chapter story outline (title and plot) in a structured JSON format.
    -   Loops through the outline, calling Groq again to write the full text content for each chapter.
    -   Passes the complete text package (characters, chapters, introduction) to the next task in the chain.
4.  **Task 2: `generate_all_media_task`**:
    -   Loads the Stable Diffusion XL models into VRAM.
    -   **For Chapters 1-4**: Implements the alternating character rule. It calls the Groq API *again* with a strict directive to generate an illustration prompt that is based on the chapter's content but focuses *only* on the designated character for that scene.
    -   **For Chapter 5 (Finale)**: Uses a predefined, safe prompt to generate a beautiful, character-free landscape shot of the garden.
    -   For each chapter, it:
        -   Generates the image locally using the SDXL Base + Refiner pipeline.
        -   Sends the chapter text to the separate TTS Server API to get the narration audio (`.wav`).
        -   Mixes the narration with background music using `pydub` and saves it as an `.mp3`.
5.  **Final Assembly**: After all chapters are processed, the individual `.mp3` files are concatenated into a single `full_podcast.mp3`.
6.  **Frontend Polling**: While all this happens, the user's browser periodically checks a status API endpoint. Celery updates the task status in real-time (e.g., "Directing Scene 3...").
7.  **Display Results**: Once the chain is complete, the final data package (with URLs to all images and the full podcast) is sent to the frontend and dynamically rendered on the page.

---

