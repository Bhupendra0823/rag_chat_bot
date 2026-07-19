from dotenv import load_dotenv
load_dotenv()
import os
from langchain_aws import ChatBedrock, BedrockEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def get_aws_embeddings():
    return BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL"),
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),   
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
def get_aws_llm():
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_LLM_MODEL"),
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  
    )
# def get_gemini_llm():
#     return ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=os.getenv("GEMINI_API_KEY")
#     )
# def get_gemini_embeddings():
#     return GoogleGenerativeAIEmbeddings(
#         model="gemini-embedding-2-preview",
#         google_api_key=os.getenv("GEMINI_API_KEY")
#     )
    
 