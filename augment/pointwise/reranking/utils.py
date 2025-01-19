from augment.pointwise.reranking.crossencoders import monoT5, monoBERT

def load_reranker(
    reranker_class, 
    reranker_name_or_path, 
    tokenizer_name=None, 
    device='cuda',
    fp16=False,
    **kwargs
):
    model_cls_map = {"monot5": monoT5, "monobert": monoBERT}

    if reranker_class is not None:
        model_cls = model_cls_map[reranker_class.lower()]
    else:
        for model_cls_key in model_cls_map:
            if model_cls_key.lower() in reranker_name_or_path.lower():
                model_cls = model_cls_map[model_cls_key]
                break
    
    crossencoder = model_cls(
        model_name_or_dir=reranker_name_or_path,
        tokenizer_name=(tokenizer_name or reranker_name_or_path),
        device=device,
        fp16=fp16
    )

    return crossencoder

