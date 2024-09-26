import pandas as pd
from typing import Optional


def viaulaize_results(
        RAG_RAGAS_results: dict[str, dict[str, float]],
        orders: Optional[list[str]]=None
        ):
    import matplotlib.pyplot as plt

    if not orders:
        faithfulness = { x:RAG_RAGAS_results[x]['faithfulness'] for x in RAG_RAGAS_results}
        answer_relevancy = { x:RAG_RAGAS_results[x]['answer_relevancy'] for x in RAG_RAGAS_results }
        context_recall = { x:RAG_RAGAS_results[x]['context_recall'] for x in RAG_RAGAS_results  }
        context_precision = { x:RAG_RAGAS_results[x]['context_precision'] for x in RAG_RAGAS_results }
    else:
        faithfulness = { x:RAG_RAGAS_results[x]['faithfulness'] for x in orders}
        answer_relevancy = { x:RAG_RAGAS_results[x]['answer_relevancy'] for x in orders }
        context_recall = { x:RAG_RAGAS_results[x]['context_recall'] for x in orders  }
        context_precision = { x:RAG_RAGAS_results[x]['context_precision'] for x in orders }



    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
   

    data = [
        (faithfulness, 'Faithfulness'),
        (answer_relevancy, 'Answer Relevancy'),
        (context_recall, 'Context Recall'),
        (context_precision, 'Context Precision')
    ]

    for i, (metric, title) in enumerate(data):
        row, col = divmod(i, 2)
        axs[row, col].bar(metric.keys(), metric.values())
        axs[row, col].set_xticklabels(metric.keys(), rotation=45, ha='right') 
        axs[row, col].set_title(title)
        axs[row, col].set_ylim(0, 1)

        for j in range(len(metric)):
            axs[row, col].text(j, list(metric.values())[j], round(list(metric.values())[j], 2), ha = 'center')

    plt.show()    
    return pd.DataFrame(RAG_RAGAS_results, index=['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision'])


def get_bar_plot(
        data: dict, *, 
        orders: Optional[list[str]]=None,
        title: Optional[str]=None, 
        xlabel: Optional[str]=None,
        ylabel: Optional[str]=None
         ):
    import matplotlib.pyplot as plt
    
    if orders:
        data = { x:data[x] for x in orders}


    plt.bar(data.keys(), data.values())
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if title:
        plt.title(title)
    plt.xticks(rotation=45)
    for i in range(len(data)):
        plt.text(i, list(data.values())[i], round(list(data.values())[i], 2), ha = 'center')

    plt.show()
