# Inference Retrieval-augmengated Generation
---
We separate the empirical expeimrent pipeline into two parts:
(1) Evaluation Pipelines 
(2) Data Generation Pipeline: transform multidocument summarization data into long-form RAG data.

### Evaluation Pipeline
This repo modularizes the following phase into standalone modules.

- Indexing
    * `bash src/bm25_index.sh`
    * `bash src/contriever.sh` 
    * `bash splade_encode.sh && bash splade_index.sh`
- [Retrieval](src/retrieve)
- [Augment](src/augment)
- [Generate](src/generate)

We include the entire RAG pipeline with evaluation on `crux-*.py`

> We will add the [Wrapper]() class to integrate all the modules and take the query as input for generate RAG results.

### Scripts
The bathc slurm scripts are in [slurm_scripts](slurm_scripts)

---
### Data generation
The code and data will be release soon.
- Codes: [Github](https://anonymous.4open.science/r/crux-data-generation/)
- Data: [Huggingface](https://huggingface.co/datasets/#/crux) ... release soon
- Human annotation pipeline: [Github](https://anonymous.4open.science/r/crux-demo-interface/)
