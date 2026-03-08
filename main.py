import os
import json
import requests
import dashscope
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dashscope import Generation, VideoSynthesis

app = Flask(__name__)
CORS(app)

api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key

# 1. GENERATE CAPTION (Qwen-Max) 
def get_caption(msg):
    pro_prompt = f"""
    CONTEXT: Kamu adalah seorang Social Media Specialist yang ahli menciptakan konten viral di TikTok dan Instagram Indonesia, khususnya saat bulan Ramadan. 
    USER TOPIC: {msg}
    
    TASK: Buat 1 caption pendek yang sangat relate dengan kehidupan masyarakat Indonesia (local wisdom).
    
    TONE & STYLE:
    - Gunakan bahasa gaul (slang) Jakarta atau bahasa santai yang sedang tren.
    - Harus mengandung unsur komedi atau "self-deprecating humor" (menertawakan diri sendiri).
    - Gunakan maksimal 15-20 kata agar tidak membosankan.
    - Akhiri dengan 2-3 hashtag yang unik dan lucu (bukan hashtag standar).

    CONSTRAINT:
    - JANGAN gunakan tanda kutip.
    - JANGAN awali dengan "Ini captionnya:".
    - LANGSUNG berikan teks caption-nya saja.
    - Hindari kata-kata klise seperti "semangat puasa ya".
    """
    
    url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "qwen-max",
        "input": {
            "messages": [{"role": "user", "content": pro_prompt}]
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()
        return res_json['output']['text']
    except:
        return "Gagal membuat caption, tapi gambarnya sudah jadi!"

# 2. GENERATE GAMBAR (Wan2.6-T2I) 
def get_image(msg):
    master_visual_prompt = """
    [C0NTEXT AN4LYSIS - D0 N0T SH0W THIS TEXT]
    1. Input Message: "{msg}"
    2. Identify core EM0TI0N (e.g., exhaustion, joy, stress) from the message.
    3. Identify the SPECIFIC, L0GICAL L0C4TI0N implied by the activity (e.g., if cooking -> closed kitchen; if praying -> a specific corner of a room).
    [END AN4LYSIS]

    [VISU4L GENER4TI0N]
    STYLE: High-end cinematic 3D animation (stylized), hyper-expressive character design. Vibrant, saturated colors. Golden hour lighting (warm sunset).

    CHARACTER (Strictly One):
    - One main Indonesian character with big expressive eyes.
    - Facial Features: MUST explicitly and intensely reflect the dynamic emotion analyzed from "{msg}".
    - Posture: Body language MUST strictly match the physical state from "{msg}" (e.g., slumped and leaning for tired; upright and focused for praying).

    ENVIR0NMENT (Strictly Private & Focused Interior):
    - The background MUST be a dynamic, tightly-controlled interior space that DIRECTLY matches the context of "{msg}".
    - DO NOT show distant cityscapes, wide public streets, or open-air mosque courtyards.
    - The setting MUST change dynamically with the input message. If kitchen, show a focused shot of a stove/table. If tired on a bed, show a bedroom.
    - Add subtle, logical Ramadan props (e.g., takjil, sarung) ONLY if they fit the specific scene.

    [STRICT N0-TEXT RULE]
    **Strictly DO NOT include any written text, letters, logos, or words anywhere in the image. The final image must contain zero letters.**
    """
    
    url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "wan2.6-t2i",
        "input": {
            "messages": [{"role": "user", "content": [{"text": master_visual_prompt}]}]
        },
        "parameters": {"n": 1, "size": "1024*1024"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()
        if response.status_code == 200:
            choices = res_json.get('output', {}).get('choices', [])
            if choices:
                return choices[0]['message']['content'][0].get('image') 
        return None
    except Exception as e:
        print(f"Error Get Image: {str(e)}")
        return None

# 3. GENERATE VIDEO (Wan2.6-I2V) 
def get_video(image_url, msg):
    master_video_prompt = f"""
    MOTION STYLE: Fluid and natural character animation, high-quality 3D cinematic movement, consistent with the initial image.
    
    CHARACTER ACTION: The character is performing: {msg}. 
    - Focus on expressive facial movements (blinking, smiling, or looking exhausted/hungry).
    - Natural body swaying and subtle hand gestures that match the Ramadan context.
    
    CAMERA MOVEMENT: 
    - Slow cinematic zoom-in or a gentle 'dolly shot' towards the subject.
    - Slight 'handheld' camera shake to make it feel organic and professional.
    
    ENVIRONMENT DYNAMICS: 
    - Add subtle background movement (e.g., flickering candle light, glowing moon light pulsing, or floating dust particles in the sunset light).
    - Soft wind blowing through the character's hair or clothing.
    
    QUALITY: 60fps feel, highly stable, no flickering, consistent textures, High-end 3D cinematic fidelity.
    """
    
    url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "wan2.6-i2v",
        "input": {
            "prompt": master_video_prompt,
            "img_url": image_url
        },
        "parameters": {
            "resolution": "720P",
            "duration": 5 
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()
        return res_json['output']['task_id']
    except Exception as e:
        print(f"Error Get Video: {str(e)}")
        return None

# ENDPOINT 1: GENERATE IMAGE & CAPTION 
@app.route('/magic', methods=['POST'])
def magic_endpoint():
    data = request.json or {}
    user_msg = data.get('msg', 'Puasa lemas nunggu bedug')
    
    try:
        caption = get_caption(user_msg)
        image_url = get_image(user_msg)
        
        return jsonify({
            "status": "success",
            "caption": caption,
            "image_url": image_url
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ENDPOINT 2: REQUEST VIDEO 
@app.route('/animate', methods=['POST'])
def animate_endpoint():
    data = request.json or {}
    image_url = data.get('image_url')
    user_msg = data.get('msg')
    
    if not image_url:
        return jsonify({"status": "error", "message": "Image URL is required"}), 400
        
    try:
        video_task_id = get_video(image_url, user_msg)
        return jsonify({
            "status": "success",
            "video_task_id": video_task_id
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ENDPOINT 3: CHECK VIDEO STATUS 
@app.route('/check-video/<task_id>', methods=['GET'])
def check_video(task_id):
    try:
        resp = VideoSynthesis.fetch(task_id)
        if resp.status_code == 200:
            return jsonify({
                "status": resp.output.task_status, 
                "video_url": resp.output.video_url if resp.output.task_status == 'SUCCEEDED' else None
            })
        return jsonify({"status": "error", "message": "Task ID tidak ditemukan"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return "Ramadan Magic AI is Live!", 200

@app.route('/')
def index():
    return send_from_directory('.', 'index.html') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
