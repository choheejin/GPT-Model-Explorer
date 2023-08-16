import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig
import fire

from utils.prompter import Prompter

import gradio as gr
import time

def respond(
        message,
        chat_history,
):
    def gen(instruction="", input_text=""):
        gc.collect()
        torch.cuda.empty_cache()
        prompt = prompter.generate_prompt(instruction, input_text)
        output = pipe(prompt, max_length=1024, temperature=0.2, num_beams=5, eos_token_id=2)
        s = output[0]["generated_text"]
        result = prompter.get_response(s)
        return result


    bot_message = gen(input_text=message)
    print(bot_message)
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    return "", chat_history

with gr.Blocks() as demo:
    # 대충 소개글
    gr.Markdown("데모입니다~")
    # 채팅 화면
    chatbot = gr.Chatbot().style(height=600)
    with gr.Row():
        with gr.Column(scale= 0.9):
            # 입력
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=0.1):
            # 버튼
            clear = gr.Button("➤")
    # 버튼 클릭
    clear.click(respond, [msg, chatbot], [msg, chatbot])
    # 엔터키
    msg.submit(respond, [msg, chatbot], [msg,chatbot])

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    MODEL = "EleutherAI/polyglot-ko-12.8b"
    LORA_WEIGHTS = "yeongsang2/polyglot-ko-12.8B-v.1.02-checkpoint-3000"

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=MODEL)
    prompter = Prompter("cbnu")

    demo.launch(server_name="0.0.0.0", server_port=5000)
