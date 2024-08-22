import os
from typing import List
import PyPDF2  
from langchain_text_splitters import RecursiveCharacterTextSplitter
 

class FileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        else:
            self.load_file()

    def load_file(self):
        if self.path.endswith(".txt"):
            with open(self.path, "r", encoding=self.encoding) as f:
                self.documents.append(f.read())
        
        elif self.path.endswith(".pdf"):
            with open(self.path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for i in range(len(reader.pages) ):
                    pageObj = reader.pages[i]
                    self.documents.append(pageObj.extract_text())
            
            
        else:
            raise ValueError("Provided file is not a .txt file or a .pdf file.")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt") or file.endswith(".pdf"):
                    self.load_file()                    

    def load_documents(self):
        self.load()
        return self.documents
