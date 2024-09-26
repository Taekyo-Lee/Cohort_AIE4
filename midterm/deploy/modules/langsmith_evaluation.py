# 🤗 Make use of 'labeled_score_string' for evaluation of chunking strategies

from langsmith.evaluation import evaluate  
from langsmith.evaluation import LangChainStringEvaluator  
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain_core.runnables import RunnableLambda 

# 🤗 prepare_datas
def labeled_score_string_prepare_data(run, example):
  return {
      "prediction" : run.outputs["prediction"],
      "reference" : example.outputs["answer"],
      "input" : example.inputs["question"]
  } 
PREPARE_DATA = {}
PREPARE_DATA['labeled_score_string'] = labeled_score_string_prepare_data

# 🤗 Evaluator configurations 
labeled_score_string_evaluator_config = {
  'llm': ChatOpenAI(model='gpt-4o-mini'),
  "accuracy": "Is the generated answer the same as the reference answer?",
}
EVALUATOR_CONFIG = {}
EVALUATOR_CONFIG['labeled_score_string'] = labeled_score_string_evaluator_config


# 🤗 Evaluators
labeled_score_string_evaluator = LangChainStringEvaluator(
    evaluator = "labeled_score_string", 
    config = EVALUATOR_CONFIG["labeled_score_string"],
    prepare_data = PREPARE_DATA["labeled_score_string"]
  )

EVALUATORS = {}
EVALUATORS['labeled_score_string'] = labeled_score_string_evaluator


# 🤗 Run LangSmith evaluation!!
def run_langsmith_evaluation(
      dataset_name: str, 
      evaluator_kinds: list[str]|str, 
      langchain_rag_chain,
      config: Optional[dict]=None, 
      experiment_prefix: Optional[str]=None , 
      metadata: Optional[dict]=None
      ):
    if isinstance(evaluator_kinds, str):
        evaluator_kinds = [evaluator_kinds]
    evaluators = [EVALUATORS[x] for x in evaluator_kinds]
    config = config or {}


    results = evaluate(
        RunnableLambda(lambda x: langchain_rag_chain.invoke(x, config=config)).invoke,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata=metadata,
        )
    return results


