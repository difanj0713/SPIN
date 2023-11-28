from imports import *

# Validated Forward Pass of DistilBert, RoBERTa, GPT2, GPT2-medium, GPT2-large, GPT2-xl

# Tokenizers: 
def roberta_tokenizer(input_lines):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=f'../Sparsify-then-Classify/model/roberta')
    #tokenizer.pad_token = '<|endoftext|>'
    tokens_batch = tokenizer(input_lines, padding='longest', truncation=True, 
        #return_overflowing_tokens=True, 
        return_tensors='pt')
    return tokens_batch['input_ids'], tokens_batch['attention_mask']

def GPT2_base_tokenizer(input_lines):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=f'../Sparsify-then-Classify/model/gpt2-base')
    tokenizer.pad_token = '<|endoftext|>'
    tokens_batch = tokenizer(input_lines, padding='longest', truncation=True, 
        #return_overflowing_tokens=True, 
        return_tensors='pt')
    return tokens_batch['input_ids'], tokens_batch['attention_mask']

def GPT2_medium_tokenizer(input_lines):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', cache_dir=f'../Sparsify-then-Classify/model/gpt2-medium')
    tokenizer.pad_token = '<|endoftext|>'
    tokens_batch = tokenizer(input_lines, padding='longest', truncation=True, 
        #return_overflowing_tokens=True, 
        return_tensors='pt')
    return tokens_batch['input_ids'], tokens_batch['attention_mask']

def GPT2_large_tokenizer(input_lines):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', cache_dir=f'../Sparsify-then-Classify/model/gpt2-large')
    tokenizer.pad_token = '<|endoftext|>'
    tokens_batch = tokenizer(input_lines, padding='longest', truncation=True, 
        #return_overflowing_tokens=True, 
        return_tensors='pt')
    return tokens_batch['input_ids'], tokens_batch['attention_mask']

def GPT2_tokenizer(input_lines):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', cache_dir=f'../Sparsify-then-Classify/model/gpt2-xl')
    tokenizer.pad_token = '<|endoftext|>'
    tokens_batch = tokenizer(input_lines, padding='longest', truncation=True, 
        #return_overflowing_tokens=True, 
        return_tensors='pt')
    return tokens_batch['input_ids'], tokens_batch['attention_mask']

def GPT2_base_imdb_tokenizer(input_lines):
    tokenizer = GPT2Tokenizer.from_pretrained('mnoukhov/gpt2-imdb-sentiment-classifier', cache_dir=f'../Sparsify-then-Classify/model/gpt2-base-imdb-finetuned')
    tokenizer.pad_token = '<|endoftext|>'
    tokens_batch = tokenizer(input_lines, padding='longest', truncation=True, 
        #return_overflowing_tokens=True, 
        return_tensors='pt')
    return tokens_batch['input_ids'], tokens_batch['attention_mask']

# BERT family
class Bert:
    def __init__(self, tokenizer, model, batch_size=500):
        self.tokenizer = tokenizer  # input: list of strings, return: (input_ids, attention_mask)
        self.model = model
        self.batch_size = batch_size
        self.batch_embedding = {}
    
    def embedding(self, input_lines, is_sorted=True):
        input_ids, mask = self.tokenizer(input_lines)
        n_tokens = torch.tensor([sum(seq) for seq in mask])
        if is_sorted:
            sort_index = torch.argsort(n_tokens, descending=True)

            input_ids = input_ids[sort_index]
            mask = mask[sort_index]
            n_tokens = n_tokens[sort_index]
        else:
            sort_index = torch.arange(len(input_lines))
        
        n_lines = input_ids.shape[0]
        self.n_lines = n_lines
        device = input_ids.device

        batch_embedding = {}

        batch_id = 0
        while batch_id * self.batch_size < n_lines:
            batch_start = batch_id * self.batch_size
            batch_end   = batch_start + self.batch_size if batch_start + self.batch_size < n_lines else n_lines
            batch_n_lines = batch_end - batch_start
            if is_sorted:
                max_n_tokens = n_tokens[batch_start]
            else:
                max_n_tokens = torch.max(n_tokens[batch_start:batch_end])

            batch_index = sort_index[batch_start:batch_end]
            batch_input_ids = input_ids[batch_start:batch_end, :max_n_tokens]
            batch_mask = mask[batch_start:batch_end, :max_n_tokens]
            batch_n_tokens = n_tokens[batch_start:batch_end]

            with torch.no_grad():
                batch_inputs_embeds = self.model.embeddings.word_embeddings(batch_input_ids)

                seq_length = batch_inputs_embeds.size(1)
                batch_position_ids = torch.arange(seq_length, dtype=torch.long, device=batch_input_ids.device)
                batch_position_ids = batch_position_ids.unsqueeze(0).expand_as(batch_input_ids)

                batch_position_embeds = self.model.embeddings.position_embeddings(batch_position_ids)
                batch_hidden_states = batch_inputs_embeds + batch_position_embeds # + batch_token_type_embeds
                batch_hidden_states = self.model.embeddings.LayerNorm(batch_hidden_states)
                batch_hidden_states = self.model.embeddings.dropout(batch_hidden_states)
            
            batch_embedding[batch_id] = {
                'index': batch_index,
                'hidden_states': batch_hidden_states,
                'mask': batch_mask,
                'n_tokens': batch_n_tokens
            }
            batch_id += 1
        self.batch_embedding = batch_embedding
        return batch_id

    def forward(self, batch_id, layer_limit, verbose=0,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        batch = self.batch_embedding[batch_id]
        hidden_states = batch['hidden_states']
        mask = batch['mask']
        n_tokens = batch['n_tokens']
        
        batch_size, seq_length = mask.shape
        head_mask = [None] * layer_limit

        input_shape = hidden_states.size()
        attention_mask = mask#: torch.Tensor = self.model.get_extended_attention_mask(mask, input_shape)

        all_hidden_states = () if output_all_hidden_states else None
        all_activations = () if output_all_activations else None
        all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
        all_pooled_activations = () if output_all_pooled_activations else None
        
        with torch.no_grad():
            h = hidden_states

        for layer in range(layer_limit):            
            with torch.no_grad():
                tmp_block = self.model.transformer.layer[layer]

                if output_all_hidden_states:
                    all_hidden_states += (h,)
                if output_all_pooled_hidden_states:
                    first_hs = h[:,0]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([first_hs, max_hs, avg_hs], dim=-1)
                    all_pooled_hidden_states += (pooled_hs,)

                attn = tmp_block.attention
                sa_output = attn(query=h, key=h, value=h, mask=attention_mask)
                sa_output = tmp_block.sa_layer_norm(sa_output[0] + h)
                
                tmp_ffn = tmp_block.ffn
                act = tmp_ffn.activation(tmp_ffn.lin1(sa_output))
                ffn_output = tmp_ffn.dropout(tmp_ffn.lin2(act))
                ffn_output: torch.Tensor = tmp_block.output_layer_norm(ffn_output + sa_output)
                
                h = ffn_output
            
                if output_all_activations:
                    all_activation += (act,)
                if output_all_pooled_activations:
                    first_act = act[:,0]
                    act = act * mask.unsqueeze(-1)
                    sum_masked = act.sum(dim=1)
                    avg_act = sum_masked / n_tokens.unsqueeze(-1)
                    act = act.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_act = act.max(dim=1)[0]
                    pooled_act = torch.stack([first_act, max_act, avg_act], dim=-1)
                    all_pooled_activations += (pooled_act,)

            if verbose>1:
                print('Layer ', layer + 1, ' / ', layer_limit, ' Processed.')
        
        with torch.no_grad():
            last_hidden_states = h if output_last_hidden_states else None

            if output_all_hidden_states:
                all_hidden_states += (h,)
            if output_all_pooled_hidden_states:
                first_hs = h[:,0]
                hs = h * mask.unsqueeze(-1)
                sum_masked = hs.sum(dim=1)
                avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                max_hs = hs.max(dim=1)[0]
                pooled_hs = torch.stack([first_hs, max_hs, avg_hs], dim=-1)
                all_pooled_hidden_states += (pooled_hs,)

            if all_pooled_hidden_states is not None:
                all_pooled_hidden_states = torch.stack(all_pooled_hidden_states)            
            if all_pooled_activations is not None:
                all_pooled_activations = torch.stack(all_pooled_activations)
                
        return (last_hidden_states, \
            all_hidden_states, all_activations, \
            all_pooled_hidden_states, all_pooled_activations,)
    
    def get_result(self, input_lines, layer_limit=6, verbose=0,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        if layer_limit > self.n_layer:
            print('DistilBERT layer limit ', self.n_layer)
            return
        n_batch = self.embedding(input_lines, is_sorted=True)
        
        index = ()       

        last_hidden_states = () if output_last_hidden_states else None
        all_hidden_states = () if output_all_hidden_states else None
        all_activations = () if output_all_activations else None
        all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
        all_pooled_activations = () if output_all_pooled_activations else None
        
        for batch_id in range(n_batch):
            if verbose:
                print('Batch ', batch_id + 1, ' / ', n_batch)
            
            batch = self.batch_embedding[batch_id]
            index += tuple(batch['index'].numpy())
            
            batch_last_hidden_states, \
                batch_all_hidden_states, batch_all_activations, \
                batch_all_pooled_hidden_states, batch_all_pooled_activations = \
                self.forward(batch_id, layer_limit, verbose,
                    output_last_hidden_states,
                    output_all_hidden_states, output_all_activations, 
                    output_all_pooled_hidden_states, output_all_pooled_activations
                )
            
            if output_last_hidden_states:
                last_hidden_states += (batch_last_hidden_states,)
            if output_all_hidden_states:
                all_hidden_states += (batch_all_hidden_states,)
            if output_all_activations:
                all_activations += (batch_all_activations,)
            if output_all_pooled_hidden_states:
                all_pooled_hidden_states += (batch_all_pooled_hidden_states,)
            if output_all_pooled_activations:
                all_pooled_activations += (batch_all_pooled_activations,)
            
        output = (index,)
        #index = torch.cat(index, dim=0)
        #if output_last_hidden_states:
        #    output += (last_hidden_states,)
        if output_all_hidden_states:
            output += (all_hidden_states,)
        if output_all_activations:
            output += (all_activations,)
        if output_all_pooled_hidden_states:
            output += (torch.cat(all_pooled_hidden_states, dim=1),)
        if output_all_pooled_activations:
            output += (torch.cat(all_pooled_activations, dim=1),)
        
        return output

class DistilBert(Bert):
    def __init__(self, tokenizer, model, batch_size=1000):
        self.n_positions = 512
        self.n_layer = 6
        self.batch_size = batch_size
        super().__init__(tokenizer, model, batch_size)

class RoBERTa(Bert):
    def __init__(self, tokenizer, model, batch_size=1000):
        self.n_positions = 512
        self.n_layer = 12
        self.batch_size = batch_size
        super().__init__(tokenizer, model, batch_size)

# Class GPT2-family
class GPT2:
    def __init__(self, tokenizer, local_model_dir, batch_size):
        self.tokenizer = tokenizer  # input: list of strings, return: (input_ids, attention_mask)
        self.local_model_dir = local_model_dir
        self.batch_size = batch_size
        self.batch_embedding = {}
        with open(f'{self.local_model_dir}/model.wte.pkl', 'rb') as f:
            self.wte = pickle.load(f)
        with open(f'{self.local_model_dir}/model.wpe.pkl', 'rb') as f:
            self.wpe = pickle.load(f)
        with open(f'{self.local_model_dir}/model.ln_f.pkl', 'rb') as f:
            self.ln_f = pickle.load(f)

    def embedding(self, input_lines, is_sorted=True):
        input_ids, mask = self.tokenizer(input_lines)
        n_tokens = torch.tensor([sum(seq) for seq in mask])
        if is_sorted:
            sort_index = torch.argsort(n_tokens, descending=True)

            input_ids = input_ids[sort_index]
            mask = mask[sort_index]
            n_tokens = n_tokens[sort_index]
        else:
            sort_index = torch.arange(len(input_lines))
        
        n_lines = input_ids.shape[0]
        self.n_lines = n_lines
        device = input_ids.device

        batch_embedding = {}

        batch_id = 0
        while batch_id * self.batch_size < n_lines:
            batch_start = batch_id * self.batch_size
            batch_end   = batch_start + self.batch_size if batch_start + self.batch_size < n_lines else n_lines
            batch_n_lines = batch_end - batch_start
            if is_sorted:
                max_n_tokens = n_tokens[batch_start]
            else:
                max_n_tokens = torch.max(n_tokens[batch_start:batch_end])

            batch_position_ids = torch.arange(0, max_n_tokens, dtype=torch.long, device=device)\
                .unsqueeze(0).view(-1, max_n_tokens)
            
            batch_index = sort_index[batch_start:batch_end]
            batch_input_ids = input_ids[batch_start:batch_end, :max_n_tokens]
            batch_mask = mask[batch_start:batch_end, :max_n_tokens]
            batch_n_tokens = n_tokens[batch_start:batch_end]
            
            batch_inputs_embeds = self.wte(batch_input_ids)
            batch_position_embeds = self.wpe(batch_position_ids)
            batch_hidden_states = batch_inputs_embeds + batch_position_embeds
            
            batch_embedding[batch_id] = {
                'index': batch_index,
                'hidden_states': batch_hidden_states,
                'mask': batch_mask,
                'n_tokens': batch_n_tokens
            }
            batch_id += 1
        self.batch_embedding = batch_embedding
        return batch_id

    def forward(self, batch_id, layer_limit, verbose=0,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        batch = self.batch_embedding[batch_id]
        hidden_states = batch['hidden_states']
        mask = batch['mask']
        n_tokens = batch['n_tokens']
        attention_mask = mask
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.long)
        attention_mask = (1.0 - attention_mask) * -10000.0

        all_hidden_states = () if output_all_hidden_states else None
        all_activations = () if output_all_activations else None
        all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
        all_pooled_activations = () if output_all_pooled_activations else None
        
        with torch.no_grad():
            h = hidden_states

        for layer in range(layer_limit):            
            with torch.no_grad():
                with open(self.local_model_dir + '/model.h.'+str(layer)+'.pkl', 'rb') as f:
                    tmp_block = pickle.load(f)

                if output_all_hidden_states:
                    all_hidden_states += (h,)                
                if output_all_pooled_hidden_states:
                    last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([last_hs, max_hs, avg_hs], dim=-1)
                    all_pooled_hidden_states += (pooled_hs,)

                attn_output = tmp_block.attn(tmp_block.ln_1(h), attention_mask=attention_mask)
                h = h + attn_output[0]

                act = tmp_block.mlp.act(tmp_block.mlp.c_fc(tmp_block.ln_2(h)))
                m = tmp_block.mlp.dropout(tmp_block.mlp.c_proj(act))
                h = h + m
            
                if output_all_activations:
                    all_activations += (act,)
                if output_all_pooled_activations:
                    last_act = act[torch.arange(act.shape[0]),n_tokens-1]
                    act = act * mask.unsqueeze(-1)
                    sum_masked = act.sum(dim=1)
                    avg_act = sum_masked / n_tokens.unsqueeze(-1)
                    act = act.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_act = act.max(dim=1)[0]
                    pooled_act = torch.stack([last_act, max_act, avg_act], dim=-1)
                    all_pooled_activations += (pooled_act,)

            if verbose>1:
                print('Layer ', layer + 1, ' / ', layer_limit, ' Processed.')
        
        with torch.no_grad():
            last_hidden_states = h if output_last_hidden_states else None

            if output_all_hidden_states:
                all_hidden_states += (h,)
            if output_all_pooled_hidden_states:
                last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                hs = h * mask.unsqueeze(-1)
                sum_masked = hs.sum(dim=1)
                avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                max_hs = hs.max(dim=1)[0]
                pooled_hs = torch.stack([last_hs, max_hs, avg_hs], dim=-1)
                all_pooled_hidden_states += (pooled_hs,)

            if all_pooled_hidden_states is not None:
                all_pooled_hidden_states = torch.stack(all_pooled_hidden_states)            
            if all_pooled_activations is not None:
                all_pooled_activations = torch.stack(all_pooled_activations)
                
        return (last_hidden_states, \
            all_hidden_states, all_activations, \
            all_pooled_hidden_states, all_pooled_activations,)
        
    def get_result(self, input_lines, layer_limit=48, verbose=0, \
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        if layer_limit > self.n_layer:
            print('GPT2-Base layer limit ', self.n_layer)
            return
        n_batch = self.embedding(input_lines, is_sorted=True)
        
        index = ()       

        last_hidden_states = () if output_last_hidden_states else None
        all_hidden_states = () if output_all_hidden_states else None
        all_activations = () if output_all_activations else None
        all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
        all_pooled_activations = () if output_all_pooled_activations else None
        
        for batch_id in range(n_batch):
            if verbose:
                print('Batch ', batch_id + 1, ' / ', n_batch)
            
            batch = self.batch_embedding[batch_id]
            index += tuple(batch['index'].numpy())
            
            batch_last_hidden_states, \
                batch_all_hidden_states, batch_all_activations, \
                batch_all_pooled_hidden_states, batch_all_pooled_activations = \
                self.forward(batch_id, layer_limit, verbose,
                    output_last_hidden_states,
                    output_all_hidden_states, output_all_activations, 
                    output_all_pooled_hidden_states, output_all_pooled_activations
                )
            
            if layer_limit == self.n_layer:
                batch_last_hidden_states = self.ln_f(batch_last_hidden_states)
                if output_all_hidden_states:
                    batch_all_hidden_states += (batch_last_hidden_states,)
                if output_all_pooled_hidden_states:
                    mask = batch['mask']
                    n_tokens = batch['n_tokens']
                    h = batch_last_hidden_states
                    
                    last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([last_hs, max_hs, avg_hs], dim=-1)
                    batch_all_pooled_hidden_states = torch.cat((batch_all_pooled_hidden_states, pooled_hs.unsqueeze(0)), dim=0)
            
            if output_last_hidden_states:
                last_hidden_states += (batch_last_hidden_states,)
            if output_all_hidden_states:
                all_hidden_states += (batch_all_hidden_states,)
            if output_all_activations:
                all_activations += (batch_all_activations,)
            if output_all_pooled_hidden_states:
                all_pooled_hidden_states += (batch_all_pooled_hidden_states,)
            if output_all_pooled_activations:
                all_pooled_activations += (batch_all_pooled_activations,)
            
        output = (index,)
        #index = torch.cat(index, dim=0)
        #if output_last_hidden_states:
        #    output += (last_hidden_states,)
        if output_all_hidden_states:
            output += (all_hidden_states,)
        if output_all_activations:
            output += (all_activations,)
        if output_all_pooled_hidden_states:
            output += (torch.cat(all_pooled_hidden_states, dim=1),)
        if output_all_pooled_activations:
            output += (torch.cat(all_pooled_activations, dim=1),)
        
        return output

class GPT2(GPT2):
    def __init__(self, tokenizer, local_model_dir=f'../Sparsify-then-Classify/model/gpt2', batch_size=1000):
        self.n_positions = 1024
        self.n_layer = 48
        self.batch_size = batch_size
        super().__init__(tokenizer, local_model_dir)

class GPT2Medium(GPT2):
    def __init__(self, tokenizer, local_model_dir=f'../Sparsify-then-Classify/model/gpt2-medium', batch_size=1000):
        self.n_positions = 1024
        self.n_layer = 24
        self.batch_size = batch_size
        super().__init__(tokenizer, local_model_dir, batch_size)

class GPT2Large(GPT2):
    def __init__(self, tokenizer, local_model_dir=f'../Sparsify-then-Classify/model/gpt2-large', batch_size=1000):
        self.n_positions = 1024
        self.n_layer = 36
        self.batch_size = batch_size
        super().__init__(tokenizer, local_model_dir, batch_size)

class GPT2XL(GPT2):
    def __init__(self, tokenizer, local_model_dir=f'../Sparsify-then-Classify/model/gpt2-xl', batch_size=1000):
        self.n_positions = 1024
        self.n_layer = 48
        self.batch_size = batch_size
        super().__init__(tokenizer, local_model_dir, batch_size)