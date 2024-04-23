import json
import os 
import sys
import boto3
import streamlit as streamlit


from langchain_community.embeddings import BedrockEmbedding
from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.vectorstores import FAISS