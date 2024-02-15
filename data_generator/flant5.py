from imports import *
from transformers import T5Tokenizer, T5Model

class FlanT5_Base:
    def __init__(self, tokenizer, model, batch_size=64):
        self.n_positions = 512
        self.tokenizer = tokenizer  # input: list of strings, return: (input_ids, attention_mask)
        self.model = model
        self.n_layer = len(model.encoder.block)
        self.batch_size = batch_size
        #self.model.eval()
        self._load_model()

    def _load_model(self):
        self.model.encoder = self.model.encoder.cuda()
        self.model.decoder = self.model.decoder.cuda()

    def embedding(self, input_lines, is_decoder=False):
        tokenizer_output = self.tokenizer(input_lines, padding='longest', truncation=True, return_tensors='pt')
        input_ids = tokenizer_output['input_ids']
        mask = tokenizer_output['attention_mask']
        n_tokens = torch.tensor([sum(seq) for seq in mask])
        
        n_lines = input_ids.shape[0]
        self.n_lines = n_lines

        input_ids = input_ids.cuda()
        mask = mask.cuda()
        n_tokens = n_tokens.cuda()

        with torch.no_grad():
            if is_decoder:
                inputs_embeds = self.model.decoder.embed_tokens(input_ids)
                hidden_states = self.model.decoder.dropout(inputs_embeds)
            else:                    
                inputs_embeds = self.model.encoder.embed_tokens(input_ids)
                hidden_states = self.model.encoder.dropout(inputs_embeds)
        batch = {
            'hidden_states': hidden_states,
            'mask': mask,
            'n_tokens': n_tokens
        }
        return batch
    
    def encoder_forward(self, batch, layer_limit, verbose=0,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        hidden_states = batch['hidden_states'].cuda()
        mask = batch['mask'].cuda()
        n_tokens = batch['n_tokens'].cuda()

        extended_attention_mask = self.model.encoder.get_extended_attention_mask(mask, mask.shape)

        all_hidden_states = () if output_all_hidden_states else None
        all_activations = () if output_all_activations else None
        all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
        all_pooled_activations = () if output_all_pooled_activations else None
        
        with torch.no_grad():
            h = hidden_states

        position_bias = None

        for layer in range(layer_limit):            
            with torch.no_grad():
                tmp_block = self.model.encoder.block[layer]
                tmp_block.requires_grad_(False)

                if output_all_hidden_states:
                    all_hidden_states += (h,)
                if output_all_pooled_hidden_states:
                    first_hs = h[:,0]
                    last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([first_hs, last_hs, max_hs, avg_hs], dim=-1)
                    all_pooled_hidden_states += (pooled_hs,)
                
                attn = tmp_block.layer[0]
                attn_output = attn(h, attention_mask=extended_attention_mask, position_bias=position_bias)
                h = attn_output[0]
                position_bias = attn_output[2]
                
                ffn = tmp_block.layer[1]
                ffn_h = ffn.layer_norm(h)
                
                act = ffn.DenseReluDense.act(ffn.DenseReluDense.wi_0(ffn_h))
                ffn_linear = ffn.DenseReluDense.wi_1(ffn_h)
                ffn_h = act * ffn_linear
                ffn_h = ffn.DenseReluDense.wo(ffn.DenseReluDense.dropout(ffn_h))
                ffn_h = ffn.dropout(ffn_h)
                
                h += ffn_h
            
                if output_all_activations:
                    all_activations += (act,)
                if output_all_pooled_activations:
                    first_act = act[:,0]
                    last_act = act[torch.arange(act.shape[0]),n_tokens-1]
                    act = act * mask.unsqueeze(-1)
                    sum_masked = act.sum(dim=1)
                    avg_act = sum_masked / n_tokens.unsqueeze(-1)
                    act = act.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_act = act.max(dim=1)[0]
                    pooled_act = torch.stack([first_act, last_act, max_act, avg_act], dim=-1)
                    all_pooled_activations += (pooled_act,)

            if verbose>1:
                print('Encoder Layer ', layer + 1, ' / ', layer_limit, ' Processed.')
                
        with torch.no_grad():
            if output_all_hidden_states:
                all_hidden_states += (h,)
            if output_all_pooled_hidden_states:
                first_hs = h[:,0]
                last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                hs = h * mask.unsqueeze(-1)
                sum_masked = hs.sum(dim=1)
                avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                max_hs = hs.max(dim=1)[0]
                pooled_hs = torch.stack([first_hs, last_hs, max_hs, avg_hs], dim=-1)
                all_pooled_hidden_states += (pooled_hs,)
                
            if layer_limit == self.n_layer:
                h = self.model.encoder.final_layer_norm(h)
                h = self.model.encoder.dropout(h)
                
                if output_all_hidden_states:
                    all_hidden_states += (h,)
                if output_all_pooled_hidden_states:
                    first_hs = h[:,0]
                    last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([first_hs, last_hs, max_hs, avg_hs], dim=-1)
                    all_pooled_hidden_states += (pooled_hs,)
            
            last_hidden_states = h if output_last_hidden_states else None

            if all_pooled_hidden_states is not None:
                all_pooled_hidden_states = torch.stack(all_pooled_hidden_states)           
            if all_pooled_activations is not None:
                all_pooled_activations = torch.stack(all_pooled_activations)
                
        return (last_hidden_states, \
            all_hidden_states, all_activations, \
            all_pooled_hidden_states, all_pooled_activations)
    
    def decoder_forward(self, batch, encoder_batch, encoder_hidden_states, layer_limit, verbose=0,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        hidden_states = batch['hidden_states'].cuda()
        mask = batch['mask'].cuda()
        n_tokens = batch['n_tokens'].cuda()

        encoder_mask = encoder_batch['mask']
        encoder_n_tokens = encoder_batch['n_tokens']
        
        batch_size, seq_length = mask.shape
        encoder_batch_size, encoder_seq_length = encoder_mask.shape

        if (batch_size==1) and (batch_size<encoder_batch_size):
            hidden_states = torch.stack([hidden_states[0,:]]*encoder_batch_size)
            mask = torch.stack([mask[0,:]]*encoder_batch_size)
            n_tokens = torch.tensor([n_tokens]*encoder_batch_size)
        
        extended_attention_mask = self.model.decoder.get_extended_attention_mask(mask, mask.shape)
        encoder_extended_attention_mask = self.model.decoder.invert_attention_mask(encoder_mask)

        all_hidden_states = () if output_all_hidden_states else None
        all_activations = () if output_all_activations else None
        all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
        all_pooled_activations = () if output_all_pooled_activations else None
        
        with torch.no_grad():
            h = hidden_states

        position_bias = None
        encoder_decoder_position_bias = None

        for layer in range(layer_limit):            
            with torch.no_grad():
                tmp_block = self.model.decoder.block[layer]
                tmp_block.requires_grad_(False)

                if output_all_hidden_states:
                    all_hidden_states += (h,)
                if output_all_pooled_hidden_states:
                    first_hs = h[:,0]
                    last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([first_hs, last_hs, max_hs, avg_hs], dim=-1)
                    all_pooled_hidden_states += (pooled_hs,)
                
                self_attn = tmp_block.layer[0]
                self_attn_output = self_attn(
                    h, 
                    attention_mask=extended_attention_mask.cuda(), 
                    position_bias=position_bias, 
                    output_attentions=True
                )
                h = self_attn_output[0]
                position_bias = self_attn_output[2]

                cross_attn = tmp_block.layer[1]
                cross_attn_output = cross_attn(
                    h, 
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_extended_attention_mask.cuda(),
                    position_bias=encoder_decoder_position_bias, 
                    output_attentions=True
                )
                h = cross_attn_output[0]
                encoder_decoder_position_bias = cross_attn_output[2]
                
                ffn = tmp_block.layer[2]
                ffn_h = ffn.layer_norm(h)
                
                act = ffn.DenseReluDense.act(ffn.DenseReluDense.wi_0(ffn_h))
                ffn_linear = ffn.DenseReluDense.wi_1(ffn_h)
                ffn_h = act * ffn_linear
                ffn_h = ffn.DenseReluDense.wo(ffn.DenseReluDense.dropout(ffn_h))
                ffn_h = ffn.dropout(ffn_h)
                
                h += ffn_h
            
                if output_all_activations:
                    all_activations += (act,)
                if output_all_pooled_activations:
                    first_act = act[:,0]
                    last_act = act[torch.arange(act.shape[0]),n_tokens-1]
                    act = act * mask.unsqueeze(-1)
                    sum_masked = act.sum(dim=1)
                    avg_act = sum_masked / n_tokens.unsqueeze(-1)
                    act = act.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_act = act.max(dim=1)[0]
                    pooled_act = torch.stack([first_act, last_act, max_act, avg_act], dim=-1)
                    all_pooled_activations += (pooled_act,)

            if verbose>1:
                print('Decoder Layer ', layer + 1, ' / ', layer_limit, ' Processed.')
                
        with torch.no_grad():
            if output_all_hidden_states:
                all_hidden_states += (h,)
            if output_all_pooled_hidden_states:
                first_hs = h[:,0]
                last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                hs = h * mask.unsqueeze(-1)
                sum_masked = hs.sum(dim=1)
                avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                max_hs = hs.max(dim=1)[0]
                pooled_hs = torch.stack([first_hs, last_hs, max_hs, avg_hs], dim=-1)
                all_pooled_hidden_states += (pooled_hs,)
                
            if layer_limit == self.n_layer:
                h = self.model.decoder.final_layer_norm(h)
                h = self.model.decoder.dropout(h)
                
                if output_all_hidden_states:
                    all_hidden_states += (h,)
                if output_all_pooled_hidden_states:
                    first_hs = h[:,0]
                    last_hs = h[torch.arange(h.shape[0]),n_tokens-1]
                    hs = h * mask.unsqueeze(-1)
                    sum_masked = hs.sum(dim=1)
                    avg_hs = sum_masked / n_tokens.unsqueeze(-1)
                    hs = hs.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    max_hs = hs.max(dim=1)[0]
                    pooled_hs = torch.stack([first_hs, last_hs, max_hs, avg_hs], dim=-1)
                    all_pooled_hidden_states += (pooled_hs,)
            
            last_hidden_states = h if output_last_hidden_states else None

            if all_pooled_hidden_states is not None:
                all_pooled_hidden_states = torch.stack(all_pooled_hidden_states)            
            if all_pooled_activations is not None:
                all_pooled_activations = torch.stack(all_pooled_activations)
                
        return (last_hidden_states, 
            all_hidden_states, all_activations, 
            all_pooled_hidden_states, all_pooled_activations)
    
    def get_result(self, input_lines, #decoder_input_lines=None, 
                layer_limit=None, verbose=0, 
                batch_output_dir=None,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        if layer_limit:
            if layer_limit > self.n_layer:
                print('Flan-T5 encoder layer limit ', self.n_layer)
                return
        else:
            layer_limit = self.n_layer
        
        if batch_output_dir is None:
            last_hidden_states = () if output_last_hidden_states else None
            all_hidden_states = () if output_all_hidden_states else None
            all_activations = () if output_all_activations else None
            all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
            all_pooled_activations = () if output_all_pooled_activations else None
        n_lines = len(input_lines)
        n_batch = math.ceil(n_lines / self.batch_size)
        pbar = tqdm(total = n_batch)
        batch_id = 0
        while batch_id * self.batch_size < n_lines:
            if verbose:
                print('Batch ', batch_id+1, ' / ', n_batch, '\tMem:', torch.cuda.mem_get_info())
            batch_start = batch_id * self.batch_size
            batch_end   = batch_start + self.batch_size if batch_start + self.batch_size < n_lines else n_lines
            batch_input = input_lines[batch_start:batch_end]
            encoder_batch = self.embedding(batch_input)
            
            encoder_batch_last_hidden_states, \
                encoder_batch_all_hidden_states, encoder_batch_all_activations, \
                encoder_batch_all_pooled_hidden_states, encoder_batch_all_pooled_activations = \
                self.encoder_forward(encoder_batch, layer_limit, verbose,
                    output_last_hidden_states,
                    output_all_hidden_states, output_all_activations, 
                    output_all_pooled_hidden_states, output_all_pooled_activations
                )
            
            # a simulation of model._shift_right() 
            decoder_batch = copy.deepcopy(encoder_batch)
            hidden_states = decoder_batch['hidden_states']
            hidden_states = torch.roll(hidden_states, shifts=1, dims=1)
            hidden_states[:,0,:] = self.model.decoder.embed_tokens(torch.tensor(0).cuda())
            decoder_batch['hidden_states'] = hidden_states

            decoder_batch_last_hidden_states, \
                decoder_batch_all_hidden_states, decoder_batch_all_activations, \
                decoder_batch_all_pooled_hidden_states, decoder_batch_all_pooled_activations = \
                self.decoder_forward(decoder_batch, encoder_batch, encoder_batch_last_hidden_states, layer_limit, verbose,
                    output_last_hidden_states,
                    output_all_hidden_states, output_all_activations, 
                    output_all_pooled_hidden_states, output_all_pooled_activations
                )

            if output_last_hidden_states:
                batch_last_hidden_states = decoder_batch_last_hidden_states.cpu()
            if output_all_hidden_states:
                batch_all_hidden_states = (encoder_batch_all_hidden_states.cpu(), decoder_batch_all_hidden_states.cpu())
            if output_all_activations:
                batch_all_activations = (encoder_batch_all_activations.cpu(), decoder_batch_all_activations.cpu())
            if output_all_pooled_hidden_states:
                batch_all_pooled_hidden_states = torch.cat((encoder_batch_all_pooled_hidden_states, decoder_batch_all_pooled_hidden_states),dim=0).cpu()
            if output_all_pooled_activations:
                batch_all_pooled_activations = torch.cat((encoder_batch_all_pooled_activations, decoder_batch_all_pooled_activations),dim=0).cpu()
            

            if batch_output_dir:
                if output_last_hidden_states:
                    with open(batch_output_dir+'/last_hs_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_last_hidden_states, f)
                if output_all_hidden_states:
                    with open(batch_output_dir+'/all_hs_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_hidden_states, f)
                if output_all_activations:
                    with open(batch_output_dir+'/all_act_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_activations, f)
                if output_all_pooled_hidden_states:
                    with open(batch_output_dir+'/all_pooled_hs_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_pooled_hidden_states, f)
                if output_all_pooled_activations:
                    with open(batch_output_dir+'/all_pooled_act_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_pooled_activations, f)
            else:
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
            batch_id += 1
            pbar.update(1)
            torch.cuda.empty_cache()
        pbar.close()
        if batch_output_dir is None:
            output = ()
            #index = torch.cat(index, dim=0)
            if output_last_hidden_states:
                output += (last_hidden_states,)
            if output_all_hidden_states:
                output += (all_hidden_states,)
            if output_all_activations:
                output += (all_activations,)
            if output_all_pooled_hidden_states:
                output += (torch.cat(all_pooled_hidden_states, dim=1),)
            if output_all_pooled_activations:
                output += (torch.cat(all_pooled_activations, dim=1),)        
            return output
        else:
            return 0

