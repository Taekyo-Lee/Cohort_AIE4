# 🚀 I wrote the code in a way that utilizes LangChain's LCEL as fully as possible.
import json
import uuid
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Literal

qa_prompt = """\
Given the following context, you must generate questions based on only the provided context.

You are to generate {n_questions} questions which should be provided in the following format:

["QUESTION #1", "QUESTION #2", ...]
...

Context:
{context}
"""


question_generation_chain = (
    ChatPromptTemplate([('human', qa_prompt)]) | 
    ChatOpenAI(model="gpt-4o-mini", temperature=0) | 
    RunnableLambda(lambda x: json.loads(x.content)).with_fallbacks(fallbacks=[RunnableLambda(lambda x: [])])
).with_config({'run_name':'Q-Generation-based-on-Context'}) 



# 🚀 helper function: creating question ids
def get_question_id(n_questions: int):
    return [str(uuid.uuid4()) for _ in range(n_questions)]

# 🚀 helper function: connecting questions to contexts by question ids.
def get_questions_and_relevant_docs(state: dict):
    if not state['questions']:
        return {}, {}
    
    questions = { id:question for id, question in zip(state['question_id'], state['questions']) }
    relevant_docs = { id:[state['context_id']] for id in state['question_id'] }
    return questions, relevant_docs

# 🚀 question_generation_pipeline -> input: {'n_questions':int , 'document':list[Document]} 
question_generation_pipeline = (
    RunnablePassthrough.assign(
        question_id = lambda x : get_question_id(x['n_questions']), # list[str]
        context = lambda x : x['document'].page_content,  # str
        context_id = lambda x : x['document'].metadata['id']   # str
        ) | 
    RunnablePassthrough.assign(
        questions = question_generation_chain
        ) | # list[str]
    RunnableLambda(lambda x: get_questions_and_relevant_docs(x)) # (dict, dict)
).with_config({'run_name': 'QnD-Generation'}) 



# 🚀 helper function: adaptor for batch input
def adapt_batch_input(input: dict)-> list[dict]:
    documents = input['documents']
    n_questions = input['n_questions']
    return [ {'document': document, 'n_questions':n_questions} for document in documents] 


# 🚀 helper function: parsing batched output
def parseing_batched_outputs(outputs: list[tuple[dict, dict]]):
    questions = {}
    relevant_docs = {}
    for output in outputs:
        questions.update(output[0])
        relevant_docs.update(output[1])
    return questions, relevant_docs


# 🚀 Batch processing using .map() method
question_generation_pipeline_in_batch = (
  RunnableLambda(adapt_batch_input) |  
  question_generation_pipeline.map() |
  RunnableLambda(parseing_batched_outputs)   
).with_config({'run_name': 'SentenceTransformer-Finetuning-Dataset-Generation'}) 


# 🚀 end-to-end function
def get_questions_and_contexts_for_SentenceTrnasformerFinetuning(split: Literal['training', 'validation', 'test'], documents: list[Document], n_questions: int=2)-> tuple[dict, dict]:
    assert split in ['training', 'validation', 'test'], "split must be either 'training', 'validation', or 'test'"
    assert hasattr(documents[0], 'metadata'), "metadata must be present in the documents"
    assert 'id' in documents[0].metadata, "metadata must have 'id' key"


    questions, relevant_contexts = question_generation_pipeline_in_batch.invoke({'documents':documents, 'n_questions':n_questions})
    corpus = {document.metadata["id"] : document.page_content for document in documents}

    return {
        'split': split,
        'questions': questions,
        'relevant_contexts': relevant_contexts,
        'corpus': corpus
    }


# 🚀 data loader

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sentence_transformers import InputExample

def get_data_loader(dataset: dict, batch_size: int=20):
    assert 'corpus' in dataset and 'questions' in dataset and 'relevant_contexts' in dataset, "dataset must have 'corpus', 'questions', and 'relevant_contexts' keys"

    corpus = dataset['corpus']
    queries = dataset['questions']
    relevant_docs = dataset['relevant_contexts']


    examples = []
    for query_id, query in queries.items():
        doc_id = relevant_docs[query_id][0]
        text = corpus[doc_id]
        example = InputExample(texts=[query, text])
        examples.append(example)

    loader = DataLoader(
        examples, batch_size=batch_size
    )
    return loader


# 🚀 Evaluate IR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from tqdm.notebook import tqdm

def evaluate_IR_for_langchain_embedding_model(
      dataset: dict, 
      embedding: HuggingFaceEmbeddings, 
      top_k: int=5, 
      verbose=False
      ):
  assert 'corpus' in dataset and 'questions' in dataset and 'relevant_contexts' in dataset, "dataset must have 'corpus', 'questions', and 'relevant_contexts' keys"

  
  corpus = dataset['corpus']
  questions = dataset['questions']
  relevant_docs = dataset['relevant_contexts']

  documents = [Document(page_content=content, metadata={"id": doc_id}) for doc_id, content in corpus.items()]
  vectorstore = Qdrant.from_documents(documents=documents, embedding=embedding)

  retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

  eval_results = []
  for id, question in tqdm(questions.items()):
    retrieved_nodes = retriever.invoke(question)
    retrieved_ids = [node.metadata["id"] for node in retrieved_nodes]
    expected_id = relevant_docs[id][0]
    is_hit = expected_id in retrieved_ids
    eval_results.append({"id": id, "question": question, "expected_id": expected_id, "is_hit": is_hit})

  return eval_results



# 🚀 Dataset for RAGAS e2e evaluation
from datasets import Dataset

def generate_RAGAS_e2e_evalation_dataset(chain, testset):
  answers = []
  contexts = []
  questions = [x['question'] for x in testset]
  ground_truths = [x['ground_truth'] for x in testset]

  for question in tqdm(questions):
    answer = chain.invoke({"question" : question})
    answers.append(answer["prediction"])
    contexts.append([answer["contexts"]])

  return Dataset.from_dict({
      "question" : questions,
      "answer" : answers,
      "contexts" : contexts,
      "ground_truth" : ground_truths
  })