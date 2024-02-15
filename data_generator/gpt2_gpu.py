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
            self.ln_f = pickle.load(f).cuda()
        self._load_model()

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
    
    def _load_model(self):
        
        self.blocks = []
        for layer in tqdm.tqdm(range(self.n_layer)):
            with open(self.local_model_dir + '/model.h.'+str(layer)+'.pkl', 'rb') as f:
                tmp_block = pickle.load(f)
                tmp_block.requires_grad_(False)
                tmp_block = tmp_block.cuda()
                self.blocks.append(tmp_block)

    def forward(self, batch_id, layer_limit, verbose=0,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        batch = self.batch_embedding[batch_id]
        hidden_states = batch['hidden_states'].cuda()
        mask = batch['mask'].cuda()
        n_tokens = batch['n_tokens'].cuda()
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
                # with open(self.local_model_dir + '/model.h.'+str(layer)+'.pkl', 'rb') as f:
                #     tmp_block = pickle.load(f)
                tmp_block = self.blocks[layer]

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
        
        for batch_id in tqdm.tqdm(range(n_batch)):
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
                    mask = batch['mask'].cuda()
                    n_tokens = batch['n_tokens'].cuda()
                    h = batch_last_hidden_states.cuda()
                    
                    last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([last_hs, max_hs, avg_hs], dim=-1)
                    batch_all_pooled_hidden_states = torch.cat((batch_all_pooled_hidden_states, pooled_hs.unsqueeze(0)), dim=0)
            
            if output_last_hidden_states:
                last_hidden_states += (batch_last_hidden_states.cpu(),)
            if output_all_hidden_states:
                all_hidden_states += (batch_all_hidden_states.cpu(),)
            if output_all_activations:
                all_activations += (batch_all_activations.cpu(),)
            if output_all_pooled_hidden_states:
                all_pooled_hidden_states += (batch_all_pooled_hidden_states.cpu(),)
            if output_all_pooled_activations:
                all_pooled_activations += (batch_all_pooled_activations.cpu(),)

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

