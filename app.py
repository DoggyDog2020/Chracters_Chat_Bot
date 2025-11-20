from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

app = Flask(__name__)

# Global variables
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = None
base_model = None
current_character = None
current_model = None

# Character configurations
CHARACTERS = {
    "mickey": {
        "name": "Mickey Mouse",
        "adapter": "./mickey-lora-adapter",
        "emoji": "🐭",
        "description": "Cheerful and optimistic mouse from Toontown",
        "color": "#FF0000"
    },
    "yoda": {
        "name": "Yoda",
        "adapter": "./yoda-lora-adapter",
        "emoji": "🟢",
        "description": "Wise Jedi Master who speaks in unique way",
        "color": "#00FF00"
    },
    "spiderman": {
        "name": "Spider-Man",
        "adapter": "./spiderman-lora-adapter",
        "emoji": "🕷️",
        "description": "Friendly neighborhood web-slinging hero",
        "color": "#0066FF"
    }
}

def load_character(character_id):
    """Load a character's LoRA adapter"""
    global tokenizer, base_model, current_model, current_character
    
    if character_id not in CHARACTERS:
        return False
    
    character = CHARACTERS[character_id]
    adapter_path = character["adapter"]
    
    # Check if adapter exists
    if not os.path.exists(adapter_path):
        return False
    
    try:
        # Load base model if not loaded
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Load tokenizer if not loaded
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        
        # Clear previous model
        if current_model is not None:
            del current_model
            torch.cuda.empty_cache()
        
        # Load character's adapter
        current_model = PeftModel.from_pretrained(base_model, adapter_path)
        current_model.eval()
        current_character = character_id
        
        return True
    except Exception as e:
        print(f"Error loading character: {e}")
        return False

def generate_response(message, max_tokens=50, temperature=0.7):
    """Generate a response from the current character"""
    global current_model, tokenizer
    
    if current_model is None or tokenizer is None:
        return "Please select a character first!"
    
    try:
        # Format message using chat template
        messages = [{"role": "user", "content": message}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(current_model.device)
        
        with torch.no_grad():
            outputs = current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', characters=CHARACTERS)

@app.route('/switch_character', methods=['POST'])
def switch_character():
    """Switch to a different character"""
    data = request.json
    character_id = data.get('character')
    
    if load_character(character_id):
        character = CHARACTERS[character_id]
        return jsonify({
            'success': True,
            'character': character['name'],
            'emoji': character['emoji'],
            'description': character['description']
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Failed to load {CHARACTERS[character_id]["name"]}. Make sure the model is trained!'
        })

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Empty message'})
    
    if current_character is None:
        return jsonify({'error': 'Please select a character first!'})
    
    response = generate_response(message)
    
    return jsonify({
        'response': response,
        'character': CHARACTERS[current_character]['name'],
        'emoji': CHARACTERS[current_character]['emoji']
    })

@app.route('/status')
def status():
    """Check which character is currently loaded"""
    if current_character:
        char = CHARACTERS[current_character]
        return jsonify({
            'loaded': True,
            'character': char['name'],
            'emoji': char['emoji']
        })
    else:
        return jsonify({'loaded': False})

if __name__ == '__main__':
    print("🎭 Character Chat Web App Starting...")
    print("📍 Open http://localhost:5000 in your browser")
    print("\n⚠️ Make sure you've trained the characters first!")
    app.run(debug=True, host='0.0.0.0', port=5000)
