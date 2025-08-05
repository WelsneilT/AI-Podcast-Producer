# creator/tasks.py (PHIÊN BẢN SỬA LỖI KEYERROR)

import json
import os
import uuid
from celery import shared_task, current_app
from django.conf import settings
from pydub import AudioSegment

from .ai_logic import (
    generate_story_outline, 
    generate_story_chapter, 
    generate_story_introduction,
    generate_illustration_prompt
)
from .ai_media import (
    load_image_gen_model,
    generate_image_locally,
    text_to_speech, 
    add_background_music,
    unload_all_ai_models
)

ART_STYLE = "charming watercolor disney storybook, masterpiece, best quality, detailed illustration"

NEGATIVE_PROMPT = (
    "deformed, blurry, ugly, malformed, mutated, disfigured, "
    "merged characters, fused characters, character hybrid, "
    "poorly drawn faces, missing features, duplicate characters, "
    "extra limbs, extra fingers, missing limbs, fused limbs, "
    "text, watermark, signature, username, low quality, worst quality, "
    "(Bart Simpson:2.0), (Lisa Simpson:2.0), (Marge Simpson:2.0), (Goofy:2.0), (Donald Duck:2.0)"
)

@shared_task
def create_full_story_task(plotline, characters):
    """TRẠM 1: Tạo toàn bộ nội dung văn bản."""
    print("--- [TRẠM 1] Bắt đầu tạo nội dung văn bản ---")
    introduction_text = generate_story_introduction(plotline, characters)
    story_outline_json = generate_story_outline(plotline, characters)
    if not story_outline_json:
        raise ValueError("Lỗi tạo dàn ý từ LLM.")
    story_data = json.loads(story_outline_json)
    outline = story_data.get('outline', [])
    story_content_list = []
    for chapter_data in outline:
        chapter_content = generate_story_chapter(chapter_data['title'], chapter_data['plot'], characters)
        story_content_list.append({
            'chapter': chapter_data.get('chapter'),
            'title': chapter_data.get('title'),
            'content': chapter_content,  # <<<--- SỬA LỖI Ở ĐÂY
        })
    return {
        'characters': characters,
        'chapters': story_content_list,
        'introduction': introduction_text
    }


@shared_task(bind=True)
def generate_all_media_task(self, text_package):
    """TRẠM 2: KẾT HỢP TRÍ TUỆ AI VÀ SỰ KIỂM SOÁT CỦA ĐẠO DIỄN"""
    print("--- [TRẠM 2] Bắt đầu quy trình tạo media - PHIÊN BẢN 'CHIẾN THẮNG' ---")
    try:
        base_pipe, refiner, _ = load_image_gen_model()
        
        story_id = str(uuid.uuid4())
        media_dir = os.path.join(settings.MEDIA_ROOT, 'stories', story_id)
        os.makedirs(media_dir, exist_ok=True)
        
        all_characters = text_package['characters']
        chapters_to_process = text_package['chapters']
        final_chapters_with_media = []
        audio_paths_for_concatenation = []
        chain_task_id = self.request.root_id

        # Xử lý audio giới thiệu (giữ nguyên)
        if text_package.get('introduction'):
            intro_narration_path = os.path.join(media_dir, "intro_narration.wav")
            intro_text = text_package.get('introduction')
            if intro_text:
                text_to_speech(intro_text, intro_narration_path)
                final_intro_audio_path = os.path.join(media_dir, "intro_audio.mp3")
                add_background_music(intro_narration_path, final_intro_audio_path)
                audio_paths_for_concatenation.append(final_intro_audio_path)
            else:
                print("    -> ⚠️ Bỏ qua audio giới thiệu vì LLM bị lỗi.")

        # Xử lý từng chương
        for i, chap in enumerate(chapters_to_process):
            chapter_number = chap['chapter']
            print(f"\n🎬 Directing Scene {chapter_number}: {chap['title']}")
            if chain_task_id:
                current_app.backend.store_result(chain_task_id, result={'status': f'Directing Scene {i+1}...'}, state='PROGRESS')

            image_path = os.path.join(media_dir, f"chapter_{chapter_number}.png")
            is_final_chapter = (i == len(chapters_to_process) - 1)

            if is_final_chapter:
                print("    -> 🌟 GRAND FINALE! Creating a beautiful garden shot.")
                action_prompt = "A stunningly beautiful, magical, and vibrant garden at sunset, flowers of all colors in full bloom, a masterpiece storybook illustration, cinematic lighting."
                final_negative_prompt = f"{NEGATIVE_PROMPT}, character, person, man, mouse, people"
            else:
                character_to_focus_index = i % 2
                focused_character = all_characters[character_to_focus_index]
                
                print(f"    -> Enforcing alternating rule. Requesting LLM to create a prompt focusing on: {focused_character['name']}")
                
                action_prompt = generate_illustration_prompt(
                    chap['content'], 
                    all_characters, 
                    character_to_focus=focused_character['name']
                )
                final_negative_prompt = NEGATIVE_PROMPT

            final_prompt = f"{action_prompt}, {ART_STYLE}"
            print(f"        -> [FINAL PROMPT] Using prompt: {final_prompt}")
            
            generate_image_locally(final_prompt, final_negative_prompt, image_path, base_pipe, refiner)
            
            if chain_task_id:
                current_app.backend.store_result(chain_task_id, result={'status': f'Chapter {i+1}: Recording narration...'}, state='PROGRESS')
            narration_path_wav = os.path.join(media_dir, f"chapter_{chapter_number}_narration.wav")
            text_to_speech(chap['content'], narration_path_wav)
            final_audio_path_mp3 = os.path.join(media_dir, f"chapter_{chapter_number}.mp3")
            add_background_music(narration_path_wav, final_audio_path_mp3)
            audio_paths_for_concatenation.append(final_audio_path_mp3)
            chap['image_url'] = os.path.join(settings.MEDIA_URL, 'stories', story_id, f"chapter_{chapter_number}.png").replace("\\", "/")
            final_chapters_with_media.append(chap)
            
        full_podcast_url = None
        if audio_paths_for_concatenation:
            full_podcast = AudioSegment.empty()
            if audio_paths_for_concatenation and 'intro' in audio_paths_for_concatenation[0]:
                full_podcast += AudioSegment.from_mp3(audio_paths_for_concatenation.pop(0))
            for audio_path in audio_paths_for_concatenation:
                full_podcast += AudioSegment.silent(duration=1500)
                full_podcast += AudioSegment.from_mp3(audio_path)
            full_podcast_path = os.path.join(media_dir, "full_podcast.mp3")
            full_podcast.export(full_podcast_path, format="mp3")
            full_podcast_url = os.path.join(settings.MEDIA_URL, 'stories', story_id, "full_podcast.mp3").replace("\\", "/")

        final_product = {
            'characters': all_characters,
            'chapters': final_chapters_with_media,
            'full_podcast_url': full_podcast_url,
            'introduction': text_package.get('introduction')
        }
        print("--- [TRẠM 2] Tất cả media đã hoàn thành. ---")
        return final_product
        
    finally:
        unload_all_ai_models()