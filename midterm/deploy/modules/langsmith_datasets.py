from typing import Optional, Iterable
from langsmith import Client
from langsmith.schemas import Dataset
import pandas as pd
from tqdm.notebook import tqdm
import random

def create_dataset_on_langsmith(client: Client, dataset_name: str, description: Optional[str]=None)-> Dataset:
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset {dataset.name} ({dataset.id}) already exists.")
    except:
        dataset = client.create_dataset(
            dataset_name=dataset_name, description=description if description else ''
        )
        print(f"Dataset {dataset.name} ({dataset.id}) created successfully.")
    
    return dataset


def push_datapoints(
        data: list[dict],
        input_keys: Iterable,
        output_keys: Iterable,
        dataset_name: str, 
        shuffle: bool=True,
        description: Optional[str]=None, 
        create_new: bool=False
        ):
    
    """
    Example:
        .. code-block:: python

            from modules.langsmith_datasets import push_datapoints

            data = RAGAS_test_dataset_df.to_dict(orient='records')
            input_keys=['question', 'contexts', 'evolution_type', 'metadata']
            output_keys=['ground_truth']
            dataset_name="AIE4-midterm-Golden_Test_Data_Set"
            description="100 RAGAS-generated questions on 'Artificial Intelligence Risk Management Framework.pdf' and 'Blueprint-for-an-AI-Bill-of-Rights.pdf'."

            push_datapoints(
                data=data,
                input_keys=input_keys,
                output_keys=output_keys,
                dataset_name=dataset_name,
                description=description,
                create_new=True,
            )    
    """

    client = Client()
    if shuffle:        
        random.shuffle(data)

    if create_new:
        try:
            client.delete_dataset(dataset_name=dataset_name)
        except:
            pass
        finally:
            dataset = create_dataset_on_langsmith(client=client, dataset_name=dataset_name, description=description)
    else:
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
            return dataset
        except:
            dataset = create_dataset_on_langsmith(client=client, dataset_name=dataset_name, description=description)

    
    dataset_id = dataset.id
    data_df = pd.DataFrame(data)

    for _, row in tqdm(data_df.iterrows()):
        inputs = {key:row[key] for key in input_keys}
        outputs = {key:row[key] for key in output_keys}

        client.create_example(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset_id
        )
    

    datapoint_size = client.read_dataset(dataset_id=dataset_id).example_count
    print(f"{len(data_df)} data points successfully pushed. There are {datapoint_size} data points in dataset.")

    return dataset