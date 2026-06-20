import random

def sanitize_input(raw_input: str) -> str:
    """SANITIZATION: Handles case & whitespace transformation."""
    return raw_input.lower().strip()

def mock_llm_response(user_query: str) -> str:
    """
    PASS TO LLM (Flexibility): Generative fallback layer 
    triggered when deterministic rule checking yields no matches.
    """
    responses = [
        f" [LLM Generated]: I processed '{user_query}'. While I don't have a hardcoded rule for that, here is a flexible interpretation...",
        f" [LLM Generated]: Let me synthesize a response for '{user_query}' dynamically based on contextual patterns..."
    ]
    return random.choice(responses)

def run_hybrid_bot():
    # KNOWLEDGE BASE: Dictionary hash map storing 5+ distinct intents
    intent_map = {
        "hello": "Hi there! Welcome to our logic engine.",
        "hi": "Hello! Rapid-fire deterministic response activated.",
        "help": "Commands: Type a greeting, ask a general question, or type 'exit'.",
        "about": "I am a hybrid system combining rule-based efficiency with generative backup.",
        "status": "All systems nominal. Logic gate access latencies: < 1ms."
    }
    
    print("=== HYBRID CORE ACTIVE ===")
    print("Testing Rule-Based Reliability vs Generative Flexibility")
    print("Type 'exit' to terminate safely.\n")

    # INPUT LOOP: Continuous 'while' cycle
    while True:
        try:
            # 1. Capture user question
            user_raw = input("You: ")
            
            # 2. Process through normalization funnel
            clean_input = sanitize_input(user_raw)
            
            # 3. EXIT STRATEGY: Clean break command execution
            if clean_input in ['exit', 'quit']:
                print("\n[System]: Executing break statement. Closing loop.")
                break
            
            # Skip evaluation processing if input stream is completely empty
            if not clean_input:
                continue

            # 4. RULE MATCH CHECKPOINT (See: image_95301d.jpg)
            # Checking hash map keys directly for maximum algorithmic speed
            if clean_input in intent_map:
                # YES -> INSTANT RESPONSE (Speed)
                reply = intent_map[clean_input]
            else:
                # NO -> FALLBACK: PASS TO LLM (Flexibility)
                reply = mock_llm_response(user_raw)
                
            # 5. Output loop response
            print(f"Bot: {reply}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\n\n[System]: Emergency break signature caught. Terminating.")
            break

if __name__ == "__main__":
    run_hybrid_bot()