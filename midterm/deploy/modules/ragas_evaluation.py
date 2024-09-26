from typing import Optional
from datasets import Dataset
from tqdm.notebook import tqdm
from langchain_core.runnables import RunnableLambda, Runnable
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI


def RAGAS_e2e_evalate(
    target: Runnable,
    testset: list[dict],
    config: Optional[dict]=None,
    metrics: Optional[list[str]]= [faithfulness, answer_relevancy, context_recall, context_precision],
):
    config = config or {}

    answers = []
    contexts = []
    questions = [x['question'] for x in testset]
    ground_truths = [x['ground_truth'] for x in testset]

    for question in tqdm(questions):
        answer = target.invoke({'question' : question}, config=config)
        answers.append(answer["prediction"])
        contexts.append([answer["contexts"]])

    data_to_evaluate = Dataset.from_dict({
        "question" : questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : ground_truths
    })

    result = evaluate(
                dataset = data_to_evaluate, 
                metrics = metrics,
                llm = ChatOpenAI(model="gpt-4o-mini")
                )
    return result
