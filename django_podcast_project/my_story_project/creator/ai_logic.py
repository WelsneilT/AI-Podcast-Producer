# creator/ai_logic.py (PHIÊN BẢN ỔN ĐỊNH - 1 PROMPT DUY NHẤT)

import os
import re
import json
from groq import Groq
from dotenv import load_dotenv

# --- Cấu hình API ---
load_dotenv()
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key: raise ValueError("GROQ_API_KEY không được tìm thấy")
    groq_client = Groq(api_key=groq_api_key)
    print("✅ Đã cấu hình Groq API Key cho logic AI.")
except Exception as e:
    print(f"🛑 LỖI cấu hình API trong ai_logic.py: {e}"); exit()


def generate_story_outline(plotline, characters):
    """Sử dụng LLM để tạo ra một dàn ý truyện gồm nhiều chương - PHIÊN BẢN CHỐNG LỖI 400."""
    print("✍️ [LLM - Biên kịch] Bắt đầu phân tích nhân vật và tạo dàn ý (Chế độ nghiêm ngặt)...")
    character_profiles = "\n".join(f"- {char['name']}: ({char.get('visual_dna', char.get('bio', ''))})" for char in characters)
    character_names_only = " and ".join([char['name'] for char in characters])
    
    # === CẢI TIẾN: PROMPT NGHIÊM NGẶT HƠN ĐỂ TRÁNH LỖI JSON ===
    prompt = f"""
    You are a JSON data generator. Your ONLY task is to create a story outline based on the provided details.
    
    **CRITICAL RULES:**
    1.  Your entire response MUST be a single, valid JSON object.
    2.  The JSON object must contain one key: "outline".
    3.  The "outline" key must be a list of 5 chapter objects.
    4.  Each chapter object must have three keys: "chapter" (number), "title" (string), and "plot" (string, a brief summary).
    5.  DO NOT write any text, explanation, or conversation before or after the JSON object. Your response must start with `{{` and end with `}}`.

    ---
    **CHARACTERS:**
    {character_profiles}

    **USER'S PLOTLINE:**
    "{plotline}"
    ---

    Generate the JSON object now.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192", 
            messages=[{"role": "user", "content": prompt}], 
            response_format={"type": "json_object"}
        )
        print("✅ [LLM] Đã nhận được dàn ý truyện (JSON).")
        return completion.choices[0].message.content
    except Exception as e:
        # Lỗi này sẽ hiển thị rõ hơn nếu Groq báo lỗi
        print(f"🛑 [LLM] Lỗi khi tạo dàn ý: {e}")
        return None


def generate_story_chapter(chapter_title, chapter_plot, characters):
    """Sử dụng LLM để viết nội dung chi tiết cho một chương."""
    print(f"✍️ [LLM] Đang viết nội dung cho chương: '{chapter_title}'...")
    character_names_only = ", ".join([f"{char['name']}" for char in characters])
    prompt = f"""
    You are a creative writer. Your task is to write the full story content for a single chapter.
    **CRUCIAL INSTRUCTIONS:**
    1.  The chapter you are writing is EXACTLY: **"{chapter_title}"**.
    2.  The plot you must follow is: "{chapter_plot}".
    3.  **You MUST ONLY write about the characters provided: {character_names_only}.**
    4.  Write approximately 150-200 words.
    5.  Output ONLY the raw text of the story. Do NOT add any extra conversational text.
    ---
    START WRITING NOW:
    """
    try:
        completion = groq_client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}])
        print(f"✅ [LLM] Đã viết xong chương '{chapter_title}'.")
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"🛑 [LLM] Lỗi khi viết chương: {e}"); return "Error generating story."


def generate_illustration_prompt(story_paragraph, all_characters, character_to_focus=None):
    """
    Tạo prompt thông minh dựa trên nội dung truyện.
    Có thể nhận lệnh 'character_to_focus' để chỉ tập trung vào một nhân vật.
    """
    print(f"🎨 [LLM-Director] Đang tạo prompt cảnh cho: '{story_paragraph[:50]}...'")
    char_details = "\n".join([f"- {char['name']}: {char.get('visual_dna', '')}" for char in all_characters])
    
    # Tạo chỉ thị động dựa trên việc có cần tập trung không
    if character_to_focus:
        print(f"    -> Nhận lệnh cứng: CHỈ tập trung vào '{character_to_focus}'.")
        focus_instruction = f"Your one and only job is to create a prompt describing the actions or emotions of **{character_to_focus}** from the paragraph. DO NOT include any other main characters."
    else:
        # Fallback (sẽ không được dùng trong logic mới nhưng vẫn giữ để an toàn)
        focus_instruction = "Decide the best composition. If one character is the clear focus, create a prompt for ONLY that character."

    prompt_template = f"""
    You are an expert Art Director creating a prompt for an illustration.

    **YOUR MISSION:**
    1.  Read the PARAGRAPH to understand the specific action and emotion.
    2.  {focus_instruction}
    3.  The prompt must be a concise, visual description (15-25 words) of the scene from the paragraph.
    4.  Your final output MUST ONLY be the raw prompt text.

    **CHARACTER VISUAL DNA:**
    {char_details}
    ---
    **PARAGRAPH FOR ILLUSTRATION:**
    "{story_paragraph}"
    ---
    **PROMPT:**
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192", 
            messages=[{"role": "user", "content": prompt_template}]
        )
        cleaned_prompt = completion.choices[0].message.content.strip().split('\n')[-1]
        print(f"    -> Prompt đã tạo (dựa trên nội dung): {cleaned_prompt}")
        return cleaned_prompt
    except Exception as e:
        print(f"🛑 [LLM] Lỗi khi tạo prompt cảnh: {e}")
        # Fallback an toàn nhất
        fallback_char = character_to_focus or all_characters[0]['name']
        return f"A beautiful watercolor painting of {fallback_char} in a garden."


def generate_story_introduction(plotline, characters):
    """Sử dụng LLM để viết một đoạn giới thiệu ngắn cho câu chuyện."""
    print("✍️ [LLM - Dẫn truyện] Đang viết lời giới thiệu...")
    character_names = " and ".join([char['name'] for char in characters])
    
    prompt = f"""
    You are a narrator for a children's storybook podcast.
    Write a short, charming introduction (40-60 words) for an upcoming story.
    It should introduce: {character_names}, and hint at the theme: "{plotline}".
    End with an inviting phrase like "Let's see what happens next!".
    Output ONLY the raw text.
    ---
    WRITE THE INTRODUCTION NOW:
    """
    try:
        completion = groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}])
        intro_text = completion.choices[0].message.content.strip()
        print(f"✅ [LLM] Lời giới thiệu đã tạo: '{intro_text[:50]}...'")
        return intro_text
    except Exception as e:
        print(f"🛑 [LLM] Lỗi khi viết lời giới thiệu: {e}"); return ""