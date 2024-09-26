from typing import Literal, Optional
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader
import uuid
import random
from tqdm.notebook import tqdm

class Chunker:
    def __init__(
            self, 
            strategy: Literal['recursive', 'semantic'], 
            recursive_chunking_config: Optional[dict] = None,
            embeddings: Optional[OpenAIEmbeddings|HuggingFaceEmbeddings|str] = None,
            semantic_chunking_config: Optional[dict] = None,
            verbose: Optional[bool] = True
            )-> None:

        """
        Example:
            .. code-block:: python

                from modules.chunker import Chunker

                recursive_chunker_openai = Chunker(
                    strategy='recursive', 
                    recursive_chunking_config=dict(chunk_size=1000, chunk_overlap=100)
                    )     
                
                semantic_chunker_snowflake = Chunker(
                    strategy='semantic', 
                    embeddings='Snowflake/snowflake-arctic-embed-m', 
                    semantic_chunking_config=dict(buffer_size=1, breakpoint_threshold_amount=90)
                    )
        """


        ### Validation
        if strategy not in ['recursive', 'semantic']:
            raise ValueError("[ERROR], strategy must be one of the following: 'recursive', 'semantic'")

        self.verbose = verbose
        
        if strategy == 'recursive':
            ### Configurations for recursive chunking
            if recursive_chunking_config == None:
               self.recursive_chunking_config = dict(chunk_size=1000, chunk_overlap=100, length_function=len)
            else:
                self.recursive_chunking_config = recursive_chunking_config
                if 'chunk_size' not in recursive_chunking_config:
                    self.recursive_chunking_config['chunk_size'] = 1000
                if 'chunk_overlap' not in recursive_chunking_config:
                    self.recursive_chunking_config['chunk_overlap'] = 100
                if 'length_function' not in recursive_chunking_config:
                    self.recursive_chunking_config['length_function'] = len

            ### Chunker
            self.chunker = RecursiveCharacterTextSplitter(**self.recursive_chunking_config)
            if self.verbose:
                print(f"[INFO] RecursiveCharacterTextSplitter loaded successfully: chunk_size={self.recursive_chunking_config['chunk_size']}, chunk_overlap={self.recursive_chunking_config['chunk_overlap']}")

        if strategy == 'semantic':
            ### Embeddings for semantic chunking
            if embeddings == None:
                self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')        
            elif isinstance(embeddings, str):
                if embeddings in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
                    self.embeddings = OpenAIEmbeddings(model=embeddings)
                else:
                    self.embeddings = HuggingFaceEmbeddings(model_name=embeddings)
            elif isinstance(embeddings, (OpenAIEmbeddings, HuggingFaceEmbeddings)):
                self.embeddings = embeddings    
            
            ### Embeddings name
            if hasattr(self.embeddings, 'model_name'): 
                self.embeddings_model_name = self.embeddings.model_name
            else: 
                self.embeddings_model_name = self.embeddings.model

            ### Configuration for semantic chunking
            if semantic_chunking_config == None:
                self.semantic_chunking_config = dict(
                    buffer_size=1, 
                    breakpoint_threshold_type="percentile", 
                    breakpoint_threshold_amount=95
                    )
            else:
                self.semantic_chunking_config = semantic_chunking_config


            ### Chunker
            self.chunker = SemanticChunker(embeddings = self.embeddings, **self.semantic_chunking_config)
            if self.verbose:
                print(f"[INFO] SemanticChunker loaded successfully: embeddings={self.embeddings_model_name}")
        

    
    def split_documents(
            self, 
            document_paths: Optional[str|list[str]]=None, 
            documents: Optional[list[Document]]=None, 
            verbose: Optional[bool]=None
            )-> list[Document]:
        
        verbose = verbose or self.verbose

        ### Load documents
        if documents != None:
            self.documents = documents
        elif document_paths != None:
            self.document_paths = document_paths

            if isinstance(self.document_paths, str):
                self.document_paths = [self.document_paths]        
            
            list_of_documents = [PyMuPDFLoader(file_path = document_path).load() for document_path in self.document_paths]
            self.documents = [document for list_of_document in list_of_documents for document in list_of_document] 
            if verbose:
                print(f"[INFO] Documents loaded successfully: {len(self.documents)} documents in total before chunking.")                
        else:
            ValueError("[ERROR] Either document_path or documents must be provided.")

        if verbose:
            print('Splitting documents...')
        self.documents = self.chunker.split_documents(documents=self.documents)

        if verbose:
            print(f"[INFO] Documents chunked up successfully: {len(self.documents)} documents in total after chunking.")

        return self.documents
    

    def get_dataset_splits_for_embedding_finetuning(
            self, 
            ratio: Optional[dict[str, int]]={'train': 0.7, 'val': 0.15, 'test': 0.15},
            document_paths: Optional[str|list[str]]=None, 
            documents: Optional[list[Document]]=None, 
            already_chunked_up: Optional[bool]=False,
            verbose: Optional[bool]=None
            )-> tuple[list[Document], list[Document], list[Document]]:
        verbose = verbose or self.verbose

        if documents:
            if already_chunked_up:
                training_documents = documents
            else:
                training_documents = self.split_documents(documents=documents, verbose=verbose)
        elif document_paths:
            training_documents = self.split_documents(document_paths=document_paths, verbose=verbose)
        else:
            ValueError("[ERROR] Either document_path or documents must be provided.")

        random.shuffle(training_documents)

        id_set = set()
        for document in training_documents:
            id = str(uuid.uuid4())
            while id in id_set:
                id = uuid.uuid4()
            id_set.add(id)
            document.metadata["id"] = id

        training_documents_size = len(training_documents)
        training_split_size = int(training_documents_size * ratio['train'])
        val_split_size = int(training_documents_size * ratio['val'])
        test_split_size = int(training_documents_size * ratio['test'])

        training_split_documents = training_documents[:training_split_size]
        val_split_documents = training_documents[training_split_size:training_split_size+val_split_size]
        test_split_documents = training_documents[training_split_size+val_split_size:]

        if verbose:
            print(f"[INFO] Dataset splits created successfully: training={len(training_split_documents)}, val={len(val_split_documents)}, test={len(test_split_documents)}")
        
        return training_split_documents, val_split_documents, test_split_documents


        