from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# api_key = '----------------------------------------' # get this free api key from https://makersuite.google.com/
llm = GooglePalm(google_api_key='Upload_your_Google_API', temperature=0.1)

# Checking the model
question = llm("Whats the time in Lahore, Pakistan?")
print(question)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# from langchain.vectorstores import FAISS
# Create a FAISS instance for vector database from 'data'
vector_db_filepath = 'faiss_index'


def create_vector_db():
    loader = CSVLoader(file_path='QuestionAnswers.csv', source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)
    vectordb.save_local(vector_db_filepath)


def get_qa_chain():
    vectordb = FAISS.load_local(vector_db_filepath, instructor_embeddings)
    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # rdocs = retriever.get_relevant_documents("Do you provide any virtual internship?")
    # print(rdocs)

    # If not available in the document then write "i don't know"

    from langchain.prompts import PromptTemplate

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    # chain_type_kwargs = {"prompt": PROMPT}

    # from langchain.chains import RetrievalQA

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
