import os
import json
import dashscope
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dashscope import Generation, ImageSynthesis, VideoSynthesis
from flask import make_response

app = Flask(__name__)
CORS(app)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

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
    
    resp = Generation.call(
        model='qwen-max', 
        prompt=pro_prompt
    )
    return resp.output.text if resp.status_code == 200 else "Aduh, AI-nya lagi buka puasa. Coba lagi ya!"
# 2. GENERATE GAMBAR (Wan2.6-T2I)
def get_image(msg):
    master_visual_prompt = f"""
    STYLE: High-end 3D animation style, Disney Pixar inspired, stylized character design, cute and expressive.
    
    SUBJECT: An Indonesian character experiencing: {msg}. 
    CHARACTER DETAIL: Big expressive eyes, wearing modern modest Ramadan attire (like a stylish koko shirt or pashmina), showing a funny and relatable facial expression.
    
    SCENE & SETTING: 
    - Authentic Indonesian background (e.g., a cozy living room with a 'bedug' nearby, a traditional 'warung' at dusk, or a terrace with 'takjil' on the table).
    - Add Ramadan ornaments like 'ketupat' hanging or a glowing crescent moon lamp.
    
    LIGHTING & COLOR: 
    - Cinematic soft lighting, golden hour vibes (warm sunset orange and deep teal shadows).
    - Vibrant, saturated colors to make it pop on mobile screens.
    
    TECHNICAL: 8k resolution, Unreal Engine 5 render, ray tracing, masterpiece, extremely detailed textures.
    """
    
    resp = ImageSynthesis.call(
        model='wan2.6-t2i', 
        prompt=master_visual_prompt,
        size='1024*1024'
    )
    
    if resp.status_code == 200:
        return resp.output.results[0].url
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
    
    QUALITY: 60fps feel, highly stable, no flickering, consistent textures, 3D Pixar-style fidelity.
    """
    
    resp = VideoSynthesis.call(
        model='wan2.6-i2v',
        img_url=image_url,
        prompt=master_video_prompt
    )
    
    if resp.status_code == 200:
        return resp.output.task_id
    return None

@app.route('/magic', methods=['POST'])
def magic_endpoint():
    # Mengambil input dari frontend
    data = request.json or {}
    user_msg = data.get('msg', 'Puasa lemas nunggu bedug')
    
    try:
        caption = get_caption(user_msg)
        image_url = get_image(user_msg)
        
        video_task_id = None
        if image_url:
            video_task_id = get_video(image_url, user_msg)

        return jsonify({
            "status": "success",
            "caption": caption,
            "image_url": image_url,
            "video_task_id": video_task_id
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/check-video/<task_id>', methods=['GET'])
def check_video(task_id):
    resp = VideoSynthesis.fetch(task_id=task_id)
    if resp.status_code == 200:
        return jsonify({
            "status": resp.output.task_status, 
            "video_url": resp.output.video_url if resp.output.task_status == 'SUCCEEDED' else None
        })
    return jsonify({"status": "error", "message": "Task ID tidak ditemukan"}), 404

@app.route('/health', methods=['GET'])
def health():
    return "Ramadan Magic AI is Live!", 200

@app.route('/')
def index():
    return send_from_directory('.', 'index.html') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
