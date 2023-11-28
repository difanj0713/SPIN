from imports import *

def main():
    parser = argparse.ArgumentParser(description='Initialize analysis for specified model and dataset.')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='The name of the model. Options: distilbert, roberta, gpt2-base, gpt2-medium, gpt2-large, gpt2-xl')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='The name of the dataset. Options: imdb, edos, sst-2')
    args = parser.parse_args()

    finetuned_model_dict = {
        "distilbert_imdb": "lvwerra/distilbert-imdb",
        "roberta_imdb": "wrmurray/roberta-base-finetuned-imdb",
        "gpt2_imdb": "lvwerra/gpt2-imdb",
        "gpt2-medium_imdb": "edbeeching/gpt2-medium-imdb",
        "gpt2-large_imdb": "edbeeching/gpt2-large-imdb",
        "gpt2-xl_imdb": "edbeeching/gpt2-xl-imdb",
        "distilbert_edos": "lct-rug-2022/edos-2023-baseline-distilbert-base-uncased-label_sexist",
        "roberta_edos": "lct-rug-2022/edos-2023-baseline-roberta-base-label_sexist",
        "distilbert_sst-2": "distilbert-base-uncased-finetuned-sst-2-english",
        "roberta_sst-2": "textattack/roberta-base-SST-2",
        "gpt2_sst-2": "PavanNeerudu/gpt2-finetuned-sst2",
        "gpt2-medium_sst-2": "michelecafagna26/gpt2-medium-finetuned-sst2-sentiment",
    }

    dataset = args.dataset
    model_name = args.model_name
    frozen_model_dir = f"../Sparsify-then-Classify/model/{args.model_name}"
    if not os.path.exists(frozen_model_dir): # download the frozen model
        if args.model_name.startswith('gpt2'):
            local_model_dir = f"../Sparsify-then-Classify/model/{args.model_name}"
            tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, cache_dir=local_model_dir)
            model = GPT2Model.from_pretrained(args.model_name, cache_dir=local_model_dir)
            
            wte = model.wte
            wpe = model.wpe
            h = model.h
            ln_f = model.ln_f

            with open(local_model_dir + '/model.wte.pkl', 'wb') as f:
                pickle.dump(wte, f)
            with open(local_model_dir + '/model.wpe.pkl', 'wb') as f:
                pickle.dump(wpe, f)
            with open(local_model_dir + '/model.ln_f.pkl', 'wb') as f:
                pickle.dump(ln_f, f)

            for i in range(len(h)):
                with open(local_model_dir + '/model.h.'+str(i)+'.pkl', 'wb') as f:
                    pickle.dump(h[i], f)

        elif args.model_name == "distilbert":
            local_model_dir = f"../Sparsify-then-Classify/model/{args.model_name}"
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=local_model_dir)
            dm = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=local_model_dir)
        
        elif args.model_name == "roberta":
            local_model_dir = f"../Sparsify-then-Classify/model/{args.model_name}"
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=local_model_dir)
            dm = RobertaModel.from_pretrained('roberta-base', cache_dir=local_model_dir)
        
        else:
            pass

    model_key = f"{args.model_name}_{args.dataset}"
    if model_key in finetuned_model_dict:
        finetuned_model_dir = f"../Sparsify-then-Classify/model/{model_key}"
        if not os.path.exists(finetuned_model_dir):
            if args.model_name.startswith('gpt2'):
                local_model_dir = f"../Sparsify-then-Classify/model/{model_key}"
                tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_dict[model_key], cache_dir=local_model_dir)
                model = GPT2Model.from_pretrained(finetuned_model_dict[model_key], cache_dir=local_model_dir)
                
                wte = model.wte
                wpe = model.wpe
                h = model.h
                ln_f = model.ln_f

                with open(local_model_dir + '/model.wte.pkl', 'wb') as f:
                    pickle.dump(wte, f)
                with open(local_model_dir + '/model.wpe.pkl', 'wb') as f:
                    pickle.dump(wpe, f)
                with open(local_model_dir + '/model.ln_f.pkl', 'wb') as f:
                    pickle.dump(ln_f, f)

                for i in range(len(h)):
                    with open(local_model_dir + '/model.h.'+str(i)+'.pkl', 'wb') as f:
                        pickle.dump(h[i], f)
            
            elif args.model_name == "distilbert":
                local_model_dir = f"../Sparsify-then-Classify/model/{model_key}"
                tokenizer = DistilBertTokenizer.from_pretrained(finetuned_model_dict[model_key], cache_dir=local_model_dir)
                dm = DistilBertModel.from_pretrained(finetuned_model_dict[model_key], cache_dir=local_model_dir)
            
            elif args.model_name == "roberta":
                local_model_dir = f"../Sparsify-then-Classify/model/{args.model_name}"
                tokenizer = RobertaTokenizer.from_pretrained(finetuned_model_dict[model_key], cache_dir=local_model_dir)
                dm = RobertaModel.from_pretrained(finetuned_model_dict[model_key], cache_dir=local_model_dir)
            
            else:
                pass

if __name__ == "__main__":
    main()
