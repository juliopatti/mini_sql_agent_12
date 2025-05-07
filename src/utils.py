import os
from tiktoken import encoding_for_model
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
import json
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_prompt(prompt_filename):
    """
    Esta função le um prompt template não formatado da pasta "prompts" pelo 
    seu filename, retornando o prompt no formato string.
    """
    dirname = os.path.dirname(os.path.dirname(__file__))
    path_txt_template = os.path.join(dirname, 'prompts', prompt_filename)
    with open(path_txt_template, "r", encoding="utf-8") as file:
        prompt = file.read()
    return prompt

def format_prompt(filename, **kwargs):
    prompt = get_prompt(filename)
    formatted_prompt = prompt.format(**kwargs)
    return formatted_prompt

def count_tokens(text, model_name):
    enc = encoding_for_model(model_name)
    return len(enc.encode(text))

def get_text(question, model_name="qwen3:0.6b"):
    temperature = 0.
    llm = OllamaLLM(model=model_name, temperature=temperature) | StrOutputParser()
    result = llm.invoke(question)
    return result

def get_dynamic_fewshot(
    embeddings,
    question,
    k=5,
    key_user_input='User input',
    key_query='SQL query',
    search_type='similarity',
    path_faiss='faiss_fewshot'
):
    if not os.path.exists(path_faiss):
        with open("src/shot_ex.json", 'r', encoding='utf-8') as file:
            examples = json.load(file)
        if not len(examples):
            return ''
        questions = []
        for ex in examples:
            example_query = ex['User input']
            del ex['User input']
            questions.append(Document(example_query, metadata=ex))
        db_ex = FAISS.from_documents(questions, embedding=embeddings)
        db_ex.save_local(path_faiss)
        
    else:
        # Load FAISS examples database
        db_ex = FAISS.load_local(
            path_faiss,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
    # Dynamic selection of examples and string preparation
    ex_shots = db_ex.similarity_search(query=question, k=k, search_type=search_type)
    dynamic_fewshot = ''
    for shot in ex_shots[::-1]:
        shot = f'{key_user_input}: {shot.page_content}\n{key_query}: {shot.metadata["SQL query"].strip()}\n\n'
        dynamic_fewshot += shot
    return dynamic_fewshot

