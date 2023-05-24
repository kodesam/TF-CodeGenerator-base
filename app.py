import gradio as gr
from transformers import TFT5ForConditionalGeneration, RobertaTokenizer

# load saved finetuned model
model = TFT5ForConditionalGeneration.from_pretrained('ThoughtFocusAI/CodeGeneration-CodeT5-base')
# load saved tokenizer
tokenizer = RobertaTokenizer.from_pretrained('ThoughtFocusAI/CodeGeneration-CodeT5-base')

def chat(chat_history, user_input):
    query = "Generate Python: " + user_input
    encoded_text = tokenizer(query, return_tensors='tf', padding='max_length', truncation=True, max_length=48)
    
    # inference
    generated_code = model.generate(
        encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"], 
        max_length=128
    )
    
    # decode generated tokens
    decoded_code = tokenizer.decode(generated_code.numpy()[0], skip_special_tokens=True)
    
    return chat_history + [(user_input, "<pre><code>"+decoded_code+"</code></pre>")]
     
my_theme = gr.Theme.from_hub('finlaymacklon/boxy_violet')
with gr.Blocks(title="Python Code Generation",theme=my_theme) as demo:
    gr.HTML(value="<style>h1 {text-align: center;}</style><h1>Python Code Generation</h1>") 
    chatbot = gr.Chatbot([], elem_id="chatbot")
    message = gr.Textbox(label="Write a python script to..",placeholder="Eg. Check if a number is prime")
    message.submit(chat, [chatbot, message], chatbot)


demo.queue().launch(enable_queue=True)