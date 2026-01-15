import torch.cuda.nvtx as nvtx

@nvtx.range("Transformer LM Forward Pass")
def annotated_TransformerLM_forward(model, x):
    """
    Input: 
        - x: A batched sequence of integer token IDs, (batch_size, sequence_length)

    Output:
        - A Tensor of (batch_size, sequence_length, vocab_size) a raw logits distribution over 
        the vocabulary.
    """
    # MatMul: (batch_size, sequence_length) . (vocab_size, embedding_dim) 
    with nvtx.range("Token Embedding"):
        x = model.in_embedding.forward(x)
    
    batch_size, seq_len = x.shape[0], x.shape[1]
    positions = model.token_positions[:seq_len].unsqueeze(0).expand(batch_size, -1)
    positions = positions.to(x.device)

    for i, tf_block in enumerate(model.tf_layers):
        with nvtx.range("TF layer {i}"):
            x = tf_block.forward(x, token_positions=positions)
    
    with nvtx.range("Pre-Norm Transformer Forward Pass"):
        x = model.norm.forward(x)

    with nvtx.range("Logit Head Ouput"):
        x = model.head.forward(x)
    
    # softmax(x, -1)  # Softmax muted, Return raw logit
    return x
