from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Read demo PDF
folder_path = "input_folder"

all_documents = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        loader = PyMuPDFLoader(file_path)
        PDF_data = loader.load()
        all_documents.extend(PDF_data) 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(all_documents)

persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

model_path = "/flask-test/models/Llama-3.2-8B-Instruct/Llama-3.2-8B-Instruct.Q8_0.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> 
    You are a helpful assistant eager to assist with providing better Google search results.
    <</SYS>> 
    
    [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
            relevant, and concise:
            {question} 
    [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant eager to assist with providing better Google search results. \
        Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
        relevant, and concise: \
        {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

result = llm.invoke("How to get a girlfriend?")
print(result)

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query = "Tell me about Alison Hawk's career and age"
qa_result = qa.invoke(query)
print(qa_result)
