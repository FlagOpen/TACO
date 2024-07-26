from datasets import load_dataset, load_from_disk


TACO_PATH="~/datasets/raw/BAAI-TACO/"

def get_taco_dataset(
    path=TACO_PATH,
    split='test', 
    difficulties=None, 
    skills=None,
    index = None,
    standard_only=False,
    shuffle=False, 
    extra_filter = None
):
    if path.endswith('.jsonl'):
        from datasets import load_dataset
        taco = load_dataset("json", data_files={'train': path})
    else:
        from datasets import load_from_disk
        taco = load_from_disk(path)
    if split is not None:
        taco = taco[split]
    if difficulties is not None:
        taco = taco.filter(lambda entry: entry['difficulty'] in difficulties)
    if skills is not None:
        taco = taco.filter(lambda entry: entry['skill'] in skills)
    if standard_only:
        taco = taco.filter(lambda entry: not (len(entry['starter_code'])>0))
    if extra_filter is not None:
        taco = extra_filter(taco)
    if shuffle:
        taco = taco.shuffle(seed=42)
    if index is not None:
        if isinstance(index, int):
            taco = taco.select(range(index))
        # elif isinstance(index, slice):
        #     taco = taco.select(index)
        elif isinstance(index, list):
            taco = taco.select(index)
    print(taco)
    return taco