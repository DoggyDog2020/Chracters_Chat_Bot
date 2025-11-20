---
title: Character Chat - Retro RPG Style
emoji: 🎮
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Character Chat - Retro RPG Style

An interactive character chatbot featuring **Mickey Mouse**, **Yoda**, and **Spider-Man** powered by fine-tuned LoRA adapters on Qwen2-0.5B-Instruct.

## Features

- 🎨 **Retro RPG Aesthetic** - Classic pixel-art style with authentic Game Boy/SNES vibes
- 💬 **Character Dialogue System** - Responses appear in text boxes just like classic RPGs
- 🎭 **Three Unique Characters**:
  - **Mickey Mouse** 🐭 - Cheerful and optimistic from Toontown
  - **Yoda** 🧙 - Wise Jedi Master with inverted speech patterns
  - **Spider-Man** 🕷️ - Your friendly neighborhood web-slinger

## Technology

- **Base Model**: Qwen/Qwen2-0.5B-Instruct (500M parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) - only ~0.43% of parameters trained
- **Framework**: Flask with custom retro UI
- **Model Size**: Each character adapter is only ~8-10MB

## How It Works

Each character has a dedicated LoRA adapter that modifies the base model's behavior. System prompts guide the characters to:
- Maintain consistent personality traits
- Use character-specific speech patterns
- Reference their respective universes and companions

## Credits

Created as part of CSCI 6370 LLM Fine-tuning Assignment
