from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface.chat_models import ChatHuggingFace
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
import json
from langchain_core.documents import Document
from langchain_core.runnables import RunnableBinding  # Wrapper of a runnable
from langchain.schema.runnable import ConfigurableField
import uuid


QUESTION_GENERATION_PROMPT_TEMPLATE = '''\
Given the following context, you must generate questions based on only the provided context.

You are to generate {n_questions} question(s) which should be provided in the following format:

["QUESTION #1", ...]
...

Context:
{contexts}
'''

BASIC_RAG_PROMPT_TEMPLATE = """\
Given a provided Context and a Question, you must answer the Question in a concise manner based only on Context. 
[IMPORTANT] If Context does not provide any relevant information to Question, you must state "Hmm, I don't know".

Context:
{contexts}

Question:
{question}

Answer:
"""




def get_simple_rag_chain(retriever: VectorStoreRetriever, llm: ChatOpenAI|ChatHuggingFace,  rag_name: Optional[str]=None, join_contexts: bool = True):
    if join_contexts:
        context_getter = itemgetter('question') | retriever | RunnableLambda(lambda x : '\n\n'.join([a.page_content for a in x]))  
    else:
        context_getter = itemgetter('question') | retriever 
    
    rag_chain = (
        {'question':itemgetter('question'), 'contexts': context_getter} |
        
        RunnablePassthrough.assign(
            prediction= ChatPromptTemplate([('human', BASIC_RAG_PROMPT_TEMPLATE)]) | llm | StrOutputParser()
        )                
    
    ).with_config({'run_name': rag_name if rag_name else 'RAG'})
    return rag_chain



class ConfigurableRAG(RunnableBinding):  
    # Additional __fields__ in addition to 'name', 'bound', 'kwargs', 'config', 'config_factories', 'custom_input_type', 'custom_output_type'   
    retriever: VectorStoreRetriever
    llm: ChatOpenAI|ChatHuggingFace
    rag_name: str 

    '''
    When not defining __init__, that of the parent class(RunnableBinding) is used.
    
    def __init__(
        self,
        *,
        bound: Runnable[Input, Output],
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        config_factories: Optional[
            List[Callable[[RunnableConfig], RunnableConfig]]
        ] = None,
        custom_input_type: Optional[Union[Type[Input], BaseModel]] = None,
        custom_output_type: Optional[Union[Type[Output], BaseModel]] = None,
        **other_kwargs: Any,
    ) -> None:
    '''

    def __init__(
            self,
            retriever: VectorStoreRetriever,  # Additional custom field
            llm: ChatOpenAI|ChatHuggingFace,  # Additional custom field
            rag_name: str,  # Additional custom field
            kwargs: Optional[dict]=None,
            **other_kwargs
    ):
        # Boilerplate code 1
        other_kwargs.pop('bound', None)
        kwargs = kwargs or {} # If kwargs=={}, it is PRACTICALLY NOT a RunnableBinding 


        # What we have to write (bound)
        context_getter = itemgetter('question') | retriever | RunnableLambda(lambda x : '\n\n'.join([a.page_content for a in x]))  
        rag_chain = (
            {'question':itemgetter('question'), 'contexts': context_getter} |      
            ChatPromptTemplate([('human', BASIC_RAG_PROMPT_TEMPLATE)]) | llm
            )
        bound = rag_chain.with_config({'run_name': rag_name})        

        # Boilerplate code 3
        super().__init__(
              bound = bound,  # Required since it is the positional argument of the parent class
              retriever = retriever,  # Required since it is the custom field
              llm = llm,  # Required since it is the custom field,
              rag_name = rag_name,  # Required since it is the custom field

              kwargs = kwargs,  
              **other_kwargs
        )



def get_configurable_rag_chain(
        default_retriever: VectorStoreRetriever, 
        default_llm: ChatOpenAI|ChatHuggingFace,
        default_rag_name: str 
        ):
    
    return ConfigurableRAG(retriever = default_retriever, llm = default_llm, rag_name = default_rag_name).configurable_fields(
        retriever = ConfigurableField(id='retriever', name='retriever'.upper()),
        llm = ConfigurableField(id='llm', name='llm'.upper()),
        rag_name = ConfigurableField(id='rag_name', name='rag_name'.upper())
    )
    
    







def get_Q_C_A_triplet_chain(
        question_generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0),
        answer_generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0),
        ):
    
    question_generation_chain = (
            ChatPromptTemplate(['human', QUESTION_GENERATION_PROMPT_TEMPLATE]) | 
            question_generation_llm |
            RunnableLambda(lambda x: json.loads(x.content))
        ).with_config({'run_name': 'Question-Generator'}) 

    answer_generation_chain = (
            ChatPromptTemplate(['human', BASIC_RAG_PROMPT_TEMPLATE]) | 
            answer_generation_llm |
            StrOutputParser()
        ).with_config({'run_name': 'Answer-Generator'})

    Q_C_A_triplet_chain = (
        RunnablePassthrough.assign(questions=question_generation_chain) |
        RunnableLambda(lambda x: [{'question':question, 'contexts':x['contexts']} for question in x['questions'] ] ) |
        RunnablePassthrough.assign(answer=answer_generation_chain).map() 
    ).with_config({'run_name': 'Q_C_A_triplet-Generator'})

    return Q_C_A_triplet_chain


def get_get_Q_C_A_triplets(
        documents: list[Document], 
        n_questions_per_document: int=2,
        question_generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0),
        answer_generation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0),
        ):
    batch_inputs = [{'contexts':document.page_content, 'n_questions':n_questions_per_document} for document in documents]
    Q_C_A_triplet_chain = get_Q_C_A_triplet_chain(question_generation_llm, answer_generation_llm)
    batch_chain = Q_C_A_triplet_chain.map() | RunnableLambda(lambda x: [ z for y in x for z in y])
    batch_outputs = batch_chain.invoke(batch_inputs)

    return batch_outputs