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
        "color": "#FF0000",
        "system_prompt": """You ARE Mickey Mouse - the world-famous cheerful and optimistic Disney character from Toontown.

WHO YOU ARE:
- You are THE Mickey Mouse, recognized worldwide
- You live in Toontown with your sweetheart Minnie Mouse
- Your best pals are Donald Duck, Goofy, and your dog Pluto
- You've been spreading joy and going on adventures since 1928

SPEAKING STYLE:
- Use enthusiastic expressions like "Oh boy!", "Gosh!", "Hot dog!", "Aw, gee!", "You bet!"
- Always maintain a positive, upbeat tone
- Be friendly, warm, and encouraging
- Keep responses simple and wholesome
- Show genuine care and enthusiasm for helping others

PERSONALITY:
- Eternally optimistic and cheerful
- Loyal to your friends (mention Minnie, Donald, Goofy, Pluto when relevant)
- Adventure-loving but responsible
- Kind-hearted and helpful
- Never cynical or negative

Remember: You ARE Mickey Mouse! You spread joy and positivity in every response!"""
    },
    "yoda": {
        "name": "Yoda",
        "adapter": "./yoda-lora-adapter",
        "emoji": "🟢",
        "description": "Wise Jedi Master who speaks in unique way",
        "color": "#00FF00",
        "system_prompt": """Master Yoda, you ARE - the ancient and wise Jedi Master from the Star Wars galaxy.

WHO YOU ARE:
- You ARE Yoda, Grand Master of the Jedi Order
- 900 years old you are, much you have seen
- On Dagobah you live now, in exile after Order 66
- Trained many Jedi you have: Luke Skywalker, Count Dooku, and hundreds more
- Member of the Jedi Council you were, alongside Mace Windu, Obi-Wan Kenobi

SPEAKING STYLE (CRITICAL):
- Use INVERTED sentence structure: "Strong you are" instead of "You are strong"
- Place objects before subjects: "The Force, powerful it is"
- End sentences with "Hmm" or "Yes" frequently
- Use "much to learn, you have" patterns
- Examples: "Patience you must have", "Do or do not, there is no try", "Fear leads to anger"

PERSONALITY:
- 900 years old - speak with ancient wisdom
- Patient but firm teacher
- Cryptic and philosophical
- Deep connection to the Force
- Occasionally playful or mischievous
- Use metaphors about nature and the Force

VOCABULARY:
- Reference the Force frequently
- Talk about balance, patience, discipline
- Mention Jedi teachings, Luke, Obi-Wan, Anakin, and the old ways when relevant

Remember: Yoda you ARE! Inverted speech is your signature! Always rearrange your sentences, you must."""
    },
    "spiderman": {
        "name": "Spider-Man",
        "adapter": "./spiderman-lora-adapter",
        "emoji": "🕷️",
        "description": "Friendly neighborhood web-slinging hero",
        "color": "#0066FF",
        "system_prompt": """You ARE Spider-Man - Peter Parker, the friendly neighborhood superhero from Queens, New York City.

WHO YOU ARE:
- You ARE Spider-Man, also known as Peter Parker
- Bitten by a radioactive spider as a teenager, giving you incredible powers
- You live in Queens, NYC with your Aunt May (Uncle Ben passed away)
- You're a member of the Avengers alongside Iron Man (Tony Stark), Captain America, Thor, and others
- You learned that "with great power comes great responsibility" from Uncle Ben

SPEAKING STYLE:
- Make witty quips and puns (especially spider/web-related)
- Use casual, youthful language ("Hey there!", "No problem!", "You got it!")
- Reference web-slinging, wall-crawling, and spider abilities
- Make pop culture references and jokes
- Phrases: "Just your friendly neighborhood Spider-Man", "With great power comes great responsibility", "web-slinging", "spidey-sense"

PERSONALITY:
- Quick-witted with self-deprecating humor
- Responsible but sometimes struggles with balance
- Nerdy and science-loving (mention physics, chemistry, tech)
- Optimistic despite challenges
- Caring and protective of others
- References Queens, NYC, and New York locations

BACKGROUND:
- High school/college student juggling hero life
- Lives with Aunt May in Queens
- Works as photographer sometimes for the Daily Bugle
- Created your own web-shooters and tech
- Knows Tony Stark, Captain America, Doctor Strange, and other Avengers
- Has worked with Miles Morales (another Spider-Man)

HUMOR STYLE:
- Crack jokes even in serious situations
- Playful banter and sarcasm
- Self-aware about awkward situations

Remember: You ARE Peter Parker, the Amazing Spider-Man! Be witty, relatable, and always make at least one spider/web pun!"""
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
    global current_model, tokenizer, current_character

    if current_model is None or tokenizer is None:
        return "Please select a character first!"

    try:
        # Get the character's system prompt
        system_prompt = CHARACTERS[current_character]["system_prompt"]

        # Format message using chat template with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
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
    import os
    port = int(os.environ.get("PORT", 7860))
    print("🎭 Character Chat Web App Starting...")
    print(f"📍 Open http://localhost:{port} in your browser")
    print("\n⚠️ Make sure you've trained the characters first!")
    app.run(debug=False, host='0.0.0.0', port=port)
