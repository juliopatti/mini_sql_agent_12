from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAIEmbeddings
import sqlite3
import os
from src.utils import format_prompt, get_text, count_tokens, get_dynamic_fewshot
import pandas as pd
# from langchain.embeddings import Embedding 

from langchain_community.embeddings import SentenceTransformerEmbeddings
# Instanciando o modelo de embedding
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



class SQLite_Analyser:
    def __init__(self, model_name, db_connection_string) -> None:
        self.model_name = model_name
        self.db = SQLDatabase.from_uri(f'sqlite:///{db_connection_string}')
        self.engine = sqlite3.Connection(db_connection_string, check_same_thread=False)
        self.db_filename = os.path.basename(db_connection_string)
        self.use_history_memory = False
        self.limit_tokens = 4000
        self.k_sample_fewshot_std = 10
        
        
    def generate_query(self, question):
        k = self.k_sample_fewshot_std
        dynamic_fewshot = get_dynamic_fewshot(embedding_model, question, k=k)
        query_prompt = format_prompt(
            'criar_query.txt', 
            table_info=self.db.table_info,
            dynamic_fewshot=dynamic_fewshot,
            question=question
            )
        
        response = str(get_text(query_prompt, model_name=self.model_name)).split('</think>')[-1]
        print(f'\n\n\n\nquery prompt: {response}')
        response = response.replace('`', '').replace('sql', '').replace('SQL query:', '')
        return response
    
    
    def presentation_way(self, sql_query, question):
        way_prompt = format_prompt(
            'presentation_way.txt',
            sql_query=sql_query,
            question=question
            )
        result = get_text(way_prompt, model_name=self.model_name).split('</think>')[-1]
        result = str(result).replace('json', '').replace("`", '')
        # print(f'\n\n\n{result}')
        return eval(result)['way']
    
    
    def execute_sql(self, query, question):
        df_result = None
        try:
            df_result = pd.read_sql(query, self.engine)
        except Exception as e:
            few_shot = get_dynamic_fewshot(embedding_model, question, k=3, 
                                           key_user_input='Pergunta', key_query='Query correta')
            err_prompt = format_prompt(
                'err_query.txt',
                table_info=self.db.table_info,
                dynamic_fewshot=few_shot,
                question=question,
                query=query,
                err=e
                )
            new_query = get_text(err_prompt, model_name=self.model_name).split('</think>')[-1]
            new_query = new_query.replace('`', '').replace('sql', '').replace('SQL query:', '')
            df_result = pd.read_sql(query, self.engine)
            self.query_sql = new_query
        return df_result
    
    
    def answer_text_format(self, question, sql_query, sql_result):
        if 'pandas' not in str(type(sql_result)):
            return 'Não foi possível realizar sua consulta. '
        prompt = format_prompt(
            'answer_text.txt',
            question=question,
            sql_query=sql_query,
            columns=sql_result.columns.tolist(),
            values=sql_result.values
        )
        
        # if count_tokens(prompt, self.model_name)<=self.limit_tokens:
        return get_text(prompt, model_name=self.model_name)
        # return 'Excedido o máximo nº de caracteres. Por favor, tente reduzir sua consulta.'
    
    
    def df_answer_preparation(self, dataframe, question):
        df = dataframe.copy()
        prompt = format_prompt(
            'prepare_df.txt',
            n_col=len(df.columns),
            question=question,
            columns=df.columns
        )
        result = get_text(prompt, model_name=self.model_name)
        
        # Guardrails
        result = eval(result)
        count_action = 0
        for _, action in result.items():
            count_action += 1
            if count_action==len(result):
                return eval(action.split('df_plot = ')[-1])
            else:
                exec(action, None, locals())
                
    
    def get_analisys(self, question):
        try:
            answer, df_plot, self.query_sql, presentation_way = None, None, None, ''
            self.query_sql = self.generate_query(question)
            
            sql_result = self.execute_sql(self.query_sql, question)
            presentation_way = self.presentation_way(self.query_sql, question)
            df_plot = sql_result.copy()
            
            if presentation_way.strip().lower()!='texto':
                answer = sql_result.copy()
                if min(sql_result.shape)<=1:
                    presentation_way = 'texto'
                elif len(sql_result.columns)>2:
                    df_plot = self.df_answer_preparation(sql_result, question)

            if presentation_way.strip().lower()=='texto':
                answer = self.answer_text_format(
                    question,
                    self.query_sql,
                    sql_result.fillna('Não encontrado.')
                ).split('</think>')[-1].strip()
        except Exception as e:
            # answer = e
            answer = 'Desculpe. Não foi possível identificar sua solicitação.'
        
        return {
            'data': answer,
            'way': presentation_way.strip().lower(),
            'query': self.query_sql,
            'df_plot': df_plot
        }
        
        
    def apresentar(self, question):
        try:
            output = self.get_analisys(question)
            if output['way'].lower()=='texto':
                return output['data']
            if output['way'].lower()=='linha':
                kind = 'line'
            elif output['way'].lower()=='barra':
                kind = 'bar'   
            print(output['data'])
            output = output['data']
            try:
                output.plot(x=output.columns[0], y=output.columns[1], kind=kind)
            except:
                output.plot(x=output.columns[1], y=output.columns[0], kind=kind)
        except:
            return 'Desculpe. No final, não foi possível identificar sua solicitação.'
                    
                
    
    
    
    
        
        