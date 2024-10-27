import os
import pickle
from dotenv import load_dotenv
os.environ["LANGSMITH_PROJECT"] = "LangGraph"
load_dotenv('/home/jetlee/workspace/.env')


from typing import Optional, Literal, List, Dict
from langchain_openai import ChatOpenAI
from typing import Literal, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import ArxivQueryRun
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain_core.tools import tool, BaseTool
from typing import Annotated
from typing import TypedDict, Literal
from langchain.agents import AgentExecutor
import functools
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph.state import StateGraph
from langgraph.constants import END
from pathlib import Path
import uuid
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Literal
from langchain_core.runnables import RunnableSequence


### 🚀 Helper) create_random_subdirectory
def create_random_subdirectory():
    random_id = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID
    subdirectory_path = os.path.join('data', random_id)
    os.makedirs(subdirectory_path, exist_ok=True)
    return subdirectory_path

### 🚀 Helper) Dummy tool for routing
def get_router_dummy_tool(options: list[str]):  # 😉 This is dummy tool!!!
  nexts = ['FINISH'] + options

  @tool
  def routing_tool(
    next: Annotated[Literal[*nexts], "next member to select for getting the job done."]
    )-> Annotated[str, "response from the selected member"]: 
    
    """Select the next member for getting the job done. If you think the task is already done, you can choose 'FINISH'."""
    return "Job done"

  return routing_tool


### 🚀 Helper) 'TEAM' Leader Agent
def get_team_leader_agent(
    llm: ChatOpenAI,
    departments: list[str],
    system_prompt: Optional[str],
):   
    options = ["FINISH"] + departments

    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", '{messages}'),
            ("system", "Given the conversation above, who should act next?  Or should we FINISH? Select one of: {options}"),  
            ("placeholder", '{agent_scratchpad}'),  
        ]
    ).partial(departments=departments, options=options) 


    agent = create_openai_tools_agent(llm=llm.with_config({'run_name':f'Team_Leader_LLM'}) , tools=[get_router_dummy_tool(options=departments)], prompt=prompt).with_config({'run_name':f'Team_Leader_Agent'})  
    return agent






### 🚀 Helper) 'PART' Leader Agent
def get_part_leader_agent(
    llm: ChatOpenAI,
    team: Literal['ResearchTeam', 'DocumentTeam'],
    tasks: list[str],
    system_prompt: Optional[str]=None,
):    
    if not system_prompt:
        system_prompt = (
    "You are a supervisor of {team} tasked with managing a conversation between the"
    " following workers: {tasks}. Given the following user request,"
    " determine the the member to act next. Each member will perform a"
    " task and respond with their results and status. "
    "When finished, respond with FINISH."
    "\n[REMEMBER 1.] You should ask your team members ONE BY ONE."
    "\n[REMEMBER 2.] You should ask all of your team members at least once before FINISH."
    )
    else:
        system_prompt = system_prompt

    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", '{messages_for_research_team}' if team=='ResearchTeam' else '{messages_for_document_team}'),
            ("placeholder", '{agent_scratchpad}'),  
        ]
    ).partial(tasks=tasks, team=team) 


    agent = create_openai_tools_agent(llm=llm.with_config({'run_name':f'{team}_Leader_LLM'}) , tools=[get_router_dummy_tool(options=tasks)], prompt=prompt).with_config({'run_name':f'{team}_Leader_Agent'})  
    return agent


### 🚀 Helper) Task AgentExecutor Helper
def get_task_node_agent_executor(
    llm: ChatOpenAI,    
    task: str,
    tools: list[BaseTool],
    team: Literal['ResearchTeam', 'DocumentTeam'],
    system_prompt: Optional[str]=None,
):    
    if not system_prompt:
        system_prompt = ("\nYou, {task}Agent, are a member of {team}. Work autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members will collaborate with you with their own specialties."
        " You are chosen for a reason!")
    else:
        system_prompt = system_prompt

    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", '{messages_for_research_team}' if team=='ResearchTeam' else '{messages_for_document_team}'),
            ("placeholder", '{agent_scratchpad}'),  # 😀 REQUIRED (This is SPECIAL key for AgentExecutor)
        ]
    ).partial(task=task, team=team) 

    agent = create_openai_tools_agent(llm=llm.with_config({'run_name':f"{task}_llm"}), tools=tools, prompt=prompt)    # 😀  Agent: llm + 'tool' + prompt
    executor = AgentExecutor(agent=agent, tools=tools).with_config({'run_name':f"{task}_Member_AgentExecutor"})  # 😀  AgentExecutor : agent + 'tool' -> LECL agent + Looping mechanism
    return executor


### 🚀 Helper) Node adaptors
def member_node_adpator(state: dict, team: Literal['ResearchTeam', 'DocumentTeam'], task: str, agentexecutor: AgentExecutor):  
    result = agentexecutor.invoke(state)
    key_to_update = "messages_for_research_team" if team == 'ResearchTeam' else "messages_for_document_team"
    
    return {key_to_update: [('human', f"Work from {task} Agent: {result['output']}")] }
    
def leader_node_adpator(state: dict, agent: AgentExecutor):
    state['intermediate_steps'] = []
    next = agent.invoke(state)[0].tool_input['next']    
    return { 'next': next}


def team_leader_node_adpator(state: dict, agent):
    state['intermediate_steps'] = []  # Since this is dummy
    next = agent.invoke(state)[0].tool_input['next']    
    return {'next': next}  # update

### 🚀 Helper) Enter & Exit subgraphs
def enter_sub_graph(global_state: TypedDict, team: Literal['ResearchTeam', 'DocumentTeam'], tasks: list[str]):
    team_messge = "messages_for_research_team" if team == 'ResearchTeam' else "messages_for_document_team"
    local_state = {}
    local_state[team_messge] = global_state['messages'][-1]
    local_state['tasks'] = tasks
    local_state['next'] = ''
    if team_messge == 'messages_for_document_team':
        local_state['current_files'] = ''
        local_state['per_member_messages_for_document_team'] = []
    
    return local_state

def exit_sub_graph(local_state: TypedDict, team: Literal['ResearchTeam', 'DocumentTeam']):
    team_messge = "messages_for_research_team" if team == 'ResearchTeam' else "messages_for_document_team"
    work =  local_state[team_messge][-1]    
    return {'messages': [work], 'team_message': local_state[team_messge]} 


"""
Team
"""
# 😉 Define Team Leader
team_leader_agent = get_team_leader_agent(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    departments= ['ResearchTeam', 'DocumentTeam'],
    system_prompt="You are a supervisor tasked with managing a conversation between the"
    " following departments: {departments}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When all workers are finished,"
    " you must respond with FINISH.",
)
adapted_team_leader_agent_callable = functools.partial(team_leader_node_adpator, agent=team_leader_agent)

## Sanity check 
# state = {
#     'messages': [('human', "Write a LinkedIn post on the paper 'Extending Llama-3’s Context Ten-Fold Overnight'.")],
#     'next': '',
#     'departments': ['ResearchTeam', 'DocumentTeam']
#     }
# response = adapted_team_leader_agent_callable(state)  
# print(response)


# 😉 Entire Graph!!
# Step 1) Define State and instantiate Graph
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] 
    next: str
    departments :List[str] 
super_graph = StateGraph(State)



# Step 2) Define nodes (llm & tools) and add them
if 'team_leader' not in super_graph.nodes.keys():
    super_graph.add_node(
        node = 'team_leader',
        action = adapted_team_leader_agent_callable
    )








"""
Research Part
"""
# 😉 Research Team Leader
research_team_leader_agent = get_part_leader_agent(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    team="ResearchTeam",
    tasks = ['WebSearch', 'AcademicPaperRetrieval(RAG)']
)
adapted_research_team_leader_agent_callable = functools.partial(leader_node_adpator, agent=research_team_leader_agent)
## Sanity check
# response = adapted_research_team_leader_agent_callable({'messages_for_research_team':[('human', "What is the paper 'Extending Llama-3’s Context Ten-Fold Overnight' about?")], 'intermediate_steps': []   })
# print(response)




# 😉 RAG Member
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text,)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50, length_function = tiktoken_len,)
# docs = PyMuPDFLoader("https://arxiv.org/pdf/2404.19553").load()
with open('extending_context_window_llama_3_docs.pkl', 'rb') as f:
    docs = pickle.load(f)

qdrant_client = QdrantClient(location="http://localhost:6333")
collection_name = "extending_context_window_llama_3"
embedding_model = "text-embedding-3-small"
if collection_name not in [x.name for x in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    qdrant_vectorstore = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=OpenAIEmbeddings(model=embedding_model))
    qdrant_vectorstore.add_documents(docs)
else:
    qdrant_vectorstore = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=OpenAIEmbeddings(model=embedding_model))

qdrant_retriever = qdrant_vectorstore.as_retriever().with_config({'run_name':"Retriever"})


RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""

rag_prompt = ChatPromptTemplate([('human', RAG_PROMPT)]) 
llm = ChatOpenAI(model="gpt-4o-mini", name="RAG_LLM")
output_parser = StrOutputParser()  
rag_chain = (
    {'question': itemgetter('question'), 'context': itemgetter('question') | qdrant_retriever} |
    rag_prompt | 
    llm |
    output_parser
).with_config({'run_name':'[RAGChain]'})


@tool
def academic_paper_retrieval(
    question: Annotated[str, "query to ask the retrieve information tool"]) -> Annotated[str, "answer to the query"]:
    '''Use Retrieval Augmented Generation (RAG) to retrieve information about the 'Extending Llama-3’s Context Ten-Fold Overnight' paper.'''
    result = rag_chain.invoke({"question" : question})
    return result 

RAG_member_AgentExecutor = get_task_node_agent_executor(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    task = 'AcademicPaperRetrieval(RAG)',
    tools = [academic_paper_retrieval],
    team='ResearchTeam',
)
adapted_RAG_member_AgentExecutor_callable = functools.partial(member_node_adpator, team='ResearchTeam', task='AcademicPaperRetrieval(RAG)', agentexecutor=RAG_member_AgentExecutor)
## Sanity check
# response = adapted_RAG_member_AgentExecutor_callable({'messages_for_research_team':[('human', "What is the paper 'Extending Llama-3’s Context Ten-Fold Overnight' about?")]   })
# print(response)



# 😉 WebSearch Agent Executor
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_tool = TavilySearchResults(max_results=5)

WebSearch_member_AgentExecutor = get_task_node_agent_executor(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    task = 'WebSearch',
    tools = [tavily_tool],
    team='ResearchTeam',
)
adpated_WebSearch_member_AgentExecutor_callable = functools.partial(member_node_adpator, team='ResearchTeam', task='WebSearch', agentexecutor=WebSearch_member_AgentExecutor)
## Sanity check
# response = adpated_WebSearch_member_AgentExecutor_callable({'messages_for_research_team':[('human', "What is the paper 'Extending Llama-3’s Context Ten-Fold Overnight' about?")]   })
# print(response)



# 😉 ResearchTeam Graph!!
# Step 1) Define State and instantiate Graph
class ResearchTeamState(TypedDict):
    messages_for_research_team: Annotated[list[BaseMessage], add_messages] 
    tasks: list[str]
    next: str

research_graph = StateGraph(ResearchTeamState)


# Step 2) Define nodes (llm & tools) and add them
if 'leader' not in research_graph.nodes.keys():
    research_graph.add_node(
        node = 'leader',
        action = adapted_research_team_leader_agent_callable
    )


if 'WebSearchAgent' not in research_graph.nodes.keys():
    research_graph.add_node(
        node = 'WebSearch',
        action = adpated_WebSearch_member_AgentExecutor_callable
    )

if 'AcademicPaperRetrieval(RAG)' not in research_graph.nodes.keys():
    research_graph.add_node(
        node = 'AcademicPaperRetrieval(RAG)',
        action = adapted_RAG_member_AgentExecutor_callable
    )


# Step 3) Add Entrypoint 
research_graph.set_entry_point('leader')   


# Step 4) Add Branch (conditional edge)
research_graph.add_edge("WebSearch", "leader")
research_graph.add_edge("AcademicPaperRetrieval(RAG)", "leader")
if 'leader' not in research_graph.branches:
    research_graph.add_conditional_edges(
        source = "leader",
        path = lambda x: x["next"],
        path_map = {"WebSearch": "WebSearch", "AcademicPaperRetrieval(RAG)": "AcademicPaperRetrieval(RAG)", "FINISH": END},
    )


# Step 5) Compile Graph (Make it a LangChain Runnable)
research_state_machine = research_graph.compile().with_config({'run_name':'ResearchTeamStateMachine'}) 


# Step 6 - Extra) Additional Adaptors before and after the state machine
before_adaptor = RunnableLambda(functools.partial(enter_sub_graph, team='ResearchTeam', team_members=['WebSearch', 'AcademicPaperRetrieval(RAG)'])) 
after_adaptor = RunnableLambda( functools.partial(exit_sub_graph, team='ResearchTeam') ) 
research_state_machine_e2e =  before_adaptor | research_state_machine | after_adaptor


## Sanity check
research_team_state = {
    'messages_for_research_team': [('human', "What is the paper 'Extending Llama-3’s Context Ten-Fold Overnight' about?")],
    'tasks': ['WebSearch', 'AcademicPaperRetrieval(RAG)'],
    'next': ''
}

# response = research_state_machine.invoke(research_team_state)
# for i, message in enumerate(response['messages_for_research_team']):
#     content = message.content
#     print(f"Message {i}: {content}")
#     print('-'*50)



"""
Document Part
"""
# 😉 Define tools to be used by DocumentTeam members
WORKING_DIRECTORY = Path(create_random_subdirectory())

@tool
def save_outline(points: Annotated[List[str], "List of main points or sections."], file_name: Annotated[str, "File path to save the outline."]) -> Annotated[str, "Path of the saved outline file."]:
    """Save a given outline as file."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(file_name: Annotated[str, "File path to load the document."],  start: Annotated[Optional[int], "The start line. Default is 0"] = None, end: Annotated[Optional[int], "The end line. Default is None"] = None) -> str:
    """Read the specified document from a file."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(content: Annotated[str, "Text content to be written into the document."], file_name: Annotated[str, "File path to save the document."]) -> Annotated[str, "Path of the saved document file."]:
    """Write text document into a file."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(file_name: Annotated[str, "Path of the document to be edited."], inserts: Annotated[Dict[int, str], "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line."] = {}) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# 😉 DocumentTeam Leader
document_team_leader_agent = get_part_leader_agent(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    team="DocumentTeam",
    tasks = ['NoteTaking', 'DocWriting', 'CopyWriting', 'DopenessEditing'],
    system_prompt=("You are a supervisor tasked with managing a conversation between the"
    " following workers: {tasks}. You should always verify the technical"
    " contents after any edits are made. "
    "Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When each team is finished,"
    " you must respond with FINISH."
    "[REMEMBER] You should ask for all of your team members to do their job before finishing in the following order [NoteTaking-> DocWriting-> CopyWriting-> DopenessEditing]."
    )
)
adapted_document_team_leader_agent_callable = functools.partial(leader_node_adpator, agent=document_team_leader_agent)


# 😉 Prelude for DocumentTeam members (모든 멤버들이 다른 멤버가 한 일을 알아햐 함)
def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir()
    try:
        written_files = [f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")]
    except:
        pass
    state['per_member_messages_for_document_team'] = [state['messages_for_document_team'][0].copy(deep=True)]
    if not written_files: 
        return {**state, "current_files": "No files written.\n\n"}   
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files])+ '\n\n', 
    }


# 😉 Note Tracker (Write outline) member
NoteTaker_member_AgentExecutor = get_task_node_agent_executor(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    task = 'NoteTaking',
    tools = [save_outline, read_document],
    team='DocumentTeam',
    system_prompt = ("You are an expert senior researcher tasked with writing a LinkedIn post outline and taking notes to craft a LinkedIn post based on given outline. "
    "First read the document if exists, then do your job.\n\n{current_files}\n" 
    "After writing the outline, save it as a new file. File name must include 'outline'.\n[REMEMBER] Never overwrite an existing file.\n")
)
NoteTaker_member_AgentExecutor = prelude | NoteTaker_member_AgentExecutor
adapted_NoteTaker_member_AgentExecutor_callable = functools.partial(member_node_adpator, team='DocumentTeam', task="NoteTaking", agentexecutor=NoteTaker_member_AgentExecutor)


# 😉 Writer member
Writer_member_AgentExecutor = get_task_node_agent_executor(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    task = 'DocWriting',
    tools = [write_document, edit_document, read_document],
    team='DocumentTeam',
    system_prompt =  ("You are an expert writing technical LinkedIn posts."
    "\n\n{current_files}Your job consists of three steps.\n\n Step1) Read the *outline* file.\n"
    "Step2) Write a draft for the LinkedIn post based on the outline.\n"
    "Step3) After writing the post, save it as a new file.  File name must include 'draft_1'.\n[REMEMBER] Never overwrite an existing file.\n")
)
Writer_member_AgentExecutor = prelude | Writer_member_AgentExecutor
adapted_Writer_member_AgentExecutor_callable = functools.partial(member_node_adpator, team='DocumentTeam', task="DocWriting", agentexecutor=Writer_member_AgentExecutor)



# 😉 Copy Writer member
CopyWriter_member_AgentExecutor = get_task_node_agent_executor(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    task = 'CopyWriting',
    tools = [write_document, edit_document, read_document],
    team='DocumentTeam',
    system_prompt = ("You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues."
    "\n\n{current_files}Your job consists of three steps.\n\n Step1) Read the first draft (draft_1)\n"
    "Step2) Edit it.\n\nAfter editing the document\nStep3) After editing the post, save it as a new file. File name must include 'draft_2'.\n\n"
    "[REMEMBER] Never overwrite an existing file.\n")
)
CopyWriter_member_AgentExecutor = prelude | CopyWriter_member_AgentExecutor
adapted_CopyWriter_member_AgentExecutor_callable = functools.partial(member_node_adpator, team='DocumentTeam', task="CopyWriting", agentexecutor=CopyWriter_member_AgentExecutor)


# 😉 Dope Editor member
DopeEditor_member_AgentExecutor = get_task_node_agent_executor(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    task = 'DopenessEditing',
    tools = [write_document, edit_document, read_document],
    team='DocumentTeam',
    system_prompt = ("You are an expert in dopeness, litness, coolness, etc - you edit the document to make sure it's dope."
    "\n\n{current_files}Your job consists of three steps.\n\n Step1) Read the second draft (draft_2)\n"
    "Step2) Edit it. After editing the document.\nStep3) After writing the post, save it as a new file. File name must include 'final_LinkIn_post'.\n"
    "[REMEMBER] Never overwrite an existing file.\n")
)
DopeEditor_member_AgentExecutor = prelude | DopeEditor_member_AgentExecutor
adapted_DopeEditor_member_AgentExecutor_callable = functools.partial(member_node_adpator, team='DocumentTeam', task="DopenessEditing", agentexecutor=DopeEditor_member_AgentExecutor)



# 😉 DocumentTeam Graph!!
# Step 1) Define State and instantiate Graph
class DocumentTeamState(TypedDict):
    messages_for_document_team: Annotated[List[BaseMessage], add_messages] 
    per_member_messages_for_document_team: Annotated[List[BaseMessage], add_messages] 
    tasks: List[str] 
    next: str 
    current_files: str 

documenting_graph = StateGraph(DocumentTeamState)


# Step 2) Define nodes (llm & tools) and add them
if 'leader' not in documenting_graph.nodes.keys():
    documenting_graph.add_node(node = 'leader', action = adapted_document_team_leader_agent_callable)

if 'NoteTaking' not in documenting_graph.nodes.keys():
    documenting_graph.add_node(node = 'NoteTaking', action = adapted_NoteTaker_member_AgentExecutor_callable)

if 'DocWriting' not in documenting_graph.nodes.keys():
    documenting_graph.add_node(node = 'DocWriting', action = adapted_Writer_member_AgentExecutor_callable)

if 'CopyWriting' not in documenting_graph.nodes.keys():
    documenting_graph.add_node(node = 'CopyWriting', action = adapted_CopyWriter_member_AgentExecutor_callable)

if 'DopenessEditing' not in documenting_graph.nodes.keys():
    documenting_graph.add_node(node = 'DopenessEditing', action = adapted_DopeEditor_member_AgentExecutor_callable)

# Step 3) Add Entrypoint 
documenting_graph.set_entry_point("leader")


# Step 4) Add Branch (conditional edge)
documenting_graph.add_edge("NoteTaking", "leader")
documenting_graph.add_edge("DocWriting", "leader")
documenting_graph.add_edge("CopyWriting", "leader")
documenting_graph.add_edge("DopenessEditing", "leader")

if 'leader' not in research_graph.branches:
    research_graph.add_conditional_edges(
        source = "leader",
        path = lambda x: x["next"],
        path_map =  {
            "NoteTaking": "NoteTaking",
            "DocWriting": "DocWriting",
            "CopyWriting" : "CopyWriting",
            "DopenessEditing" : "DopenessEditing",
            "FINISH": END,
        }
    )

# response = adapted_NoteTaker_member_AgentExecutor_callable({'current_files':"There is no current file."})

# print(response)
# Step 5) Compile Graph (Make it a LangChain Runnable)
# document_team_state_machine = documenting_graph.compile().with_config({'run_name':'DocumentTeamStateMachine'}) 

