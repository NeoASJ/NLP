from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# pip install transformers
# pip install torch

def improved_ai_chatbot():
    # Configure device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model to device  
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    print("AI ChatBot: Hi! I'm an AI chatbot. Type 'quit' to exit.")
    
    chat_history_ids = None
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("AI ChatBot: Goodbye!")
            break
            
        # user input
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, 
            return_tensors='pt'
        ).to(device)
        
        #Add chat history
        bot_input_ids = new_user_input_ids if chat_history_ids is None else torch.cat(
            [chat_history_ids, new_user_input_ids], dim=-1
        )
        
        
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        # print response
        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        print(f"AI ChatBot: {response}")

# call function
improved_ai_chatbot()