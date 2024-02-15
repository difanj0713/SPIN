from imports import *
from models import *

layer_dict = {"distilbert": 6, "roberta": 12, "gpt2-xl": 48, "gpt2": 12, "gpt2-medium": 24, "gpt2-large": 36}

def main():
    parser = argparse.ArgumentParser(description='Initialize analysis for specified model and dataset.')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='The name of the model. Options: distilbert, roberta, gpt2-base, gpt2-medium, gpt2-large, gpt2-xl')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='The name of the dataset. Options: imdb, edos, sst-2')
    parser.add_argument('--is_finetuned', type=int, required=True,
                        help='Flag for finetuned models. Options: 1 for finetuned, 0 for frozen')
    args = parser.parse_args()

    text_train, text_val, text_test, label_train, label_val, label_test = my_train_test_split(args.dataset)
    
    finetuned_model_key = f"{args.model_name}_{args.dataset}"
    frozen_model_key = f"{args.model_name}"
    model_dict = {"distilbert": DistilBert,
                  "roberta": Roberta,
                  "gpt2": GPT2Base,
                  "gpt2-medium": GPT2Medium,
                  "gpt2-large": GPT2Large,
                  "gpt2-xl": GPT2XL,
                }
    finetuned_model_dict = {
        "distilbert_imdb": "lvwerra/distilbert-imdb",
        "roberta_imdb": "wrmurray/roberta-base-finetuned-imdb",
        "gpt2_imdb": "mnoukhov/gpt2-imdb-sentiment-classifier",
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
    tokenizer_dict = {  "distilbert": distilbert_tokenizer,
                        "roberta": roberta_tokenizer,
                        "gpt2": GPT2_tokenizer,
                        "gpt2-medium": GPT2_medium_tokenizer,
                        "gpt2-large": GPT2_large_tokenizer,
                        "gpt2-xl": GPT2_xl_tokenizer,
                    }
    finetuned_tokenizer_dict = { "distilbert_imdb": distilbert_imdb_tokenizer,
                                 "distilbert_edos": distilbert_edos_tokenizer,
                                 "distilbert_sst-2": distilbert_sst2_tokenizer,
                                 "roberta_imdb": roberta_imdb_tokenizer,
                                 "roberta_edos": roberta_edos_tokenizer,
                                 "roberta_sst-2": roberta_sst2_tokenizer,
                                 "gpt2_imdb": GPT2_imdb_tokenizer,
                                 "gpt2_sst-2": GPT2_sst2_tokenizer,
                                 "gpt2-medium_imdb": GPT2_medium_imdb_tokenizer,
                                 "gpt2-medium_sst-2": GPT2_medium_sst2_tokenizer,
                                 "gpt2-large_imdb": GPT2_large_imdb_tokenizer,
                                 "gpt2-xl_imdb": GPT2_xl_imdb_tokenizer,
                    }
    #model = model_dict[frozen_model_key]

    batch_size = 1000 # the number of sentences in one batch for forward pass
    if args.is_finetuned == 1:
        if finetuned_model_key in finetuned_tokenizer_dict:
            finetuned_tokenizer = finetuned_tokenizer_dict[finetuned_model_key]
        else:
            sys.exit("Finetuned models not available for model {0}, dataset {1}".format(args.model_name, args.dataset))

        if args.model_name == "distilbert":
            dm = DistilBertModel.from_pretrained(finetuned_model_dict[finetuned_model_key], cache_dir=f'../model/${finetuned_model_key}')
            model = model_dict[frozen_model_key](tokenizer=finetuned_tokenizer, model=dm, batch_size=batch_size)
        elif args.model_name == "roberta":
            rm = RobertaModel.from_pretrained(finetuned_model_dict[finetuned_model_key], cache_dir=f'../model/${finetuned_model_key}')
            model = model_dict[frozen_model_key](tokenizer=finetuned_tokenizer, model=rm, batch_size=batch_size)
        else:
            local_model_dir = f"../model/{finetuned_model_key}"
            model = model_dict[frozen_model_key](tokenizer=finetuned_tokenizer, local_model_dir=local_model_dir, batch_size=batch_size)
        
        # train, val, test
        input_lines = text_train
        res = model.get_result(input_lines, layer_limit=layer_dict[args.model_name], verbose=2, \
                    output_last_hidden_states=True,
                    output_all_hidden_states=False, output_all_activations=False, 
                    output_all_pooled_hidden_states=True, output_all_pooled_activations=True)
        output_dir = f'../dataset_acts/{args.dataset}/train_all_{args.model_name}_finetuned_res.pkl'
        with open(output_dir, 'wb') as f:
            pickle.dump(res, f)

        input_lines = text_val
        res = model.get_result(input_lines, layer_limit=layer_dict[args.model_name], verbose=2, \
                    output_last_hidden_states=True,
                    output_all_hidden_states=False, output_all_activations=False, 
                    output_all_pooled_hidden_states=True, output_all_pooled_activations=True)
        output_dir = f'../dataset_acts/{args.dataset}/val_all_{args.model_name}_finetuned_res.pkl'
        with open(output_dir, 'wb') as f:
            pickle.dump(res, f)

        input_lines = text_test
        res = model.get_result(input_lines, layer_limit=layer_dict[args.model_name], verbose=2, \
                    output_last_hidden_states=True,
                    output_all_hidden_states=False, output_all_activations=False, 
                    output_all_pooled_hidden_states=True, output_all_pooled_activations=True)
        output_dir = f'../dataset_acts/{args.dataset}/test_all_{args.model_name}_finetuned_res.pkl'
        with open(output_dir, 'wb') as f:
            pickle.dump(res, f)
        

    else:
        frozen_tokenizer = tokenizer_dict[frozen_model_key]

        if args.model_name == "distilbert":
            dm = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=f'../model/distilbert')
            model = model_dict[frozen_model_key](tokenizer=frozen_tokenizer, model=dm, batch_size=batch_size)
        elif args.model_name == "roberta":
            rm = RobertaModel.from_pretrained('roberta-base', cache_dir=f'../model/roberta')
            model = model_dict[frozen_model_key](tokenizer=frozen_tokenizer, model=rm, batch_size=batch_size)
        else:
            local_model_dir = f"../model/{frozen_model_key}"
            model = model_dict[frozen_model_key](tokenizer=frozen_tokenizer, local_model_dir=local_model_dir, batch_size=batch_size)

        # train, val, test
        input_lines = text_train
        res = model.get_result(input_lines, layer_limit=layer_dict[args.model_name], verbose=2, \
                    output_last_hidden_states=True,
                    output_all_hidden_states=False, output_all_activations=False, 
                    output_all_pooled_hidden_states=True, output_all_pooled_activations=True)
        output_dir = f'../dataset_acts/{args.dataset}/train_all_{args.model_name}_res.pkl'
        with open(output_dir, 'wb') as f:
            pickle.dump(res, f)

        input_lines = text_val
        res = model.get_result(input_lines, layer_limit=layer_dict[args.model_name], verbose=2, \
                    output_last_hidden_states=True,
                    output_all_hidden_states=False, output_all_activations=False, 
                    output_all_pooled_hidden_states=True, output_all_pooled_activations=True)
        output_dir = f'../dataset_acts/{args.dataset}/val_all_{args.model_name}_res.pkl'
        with open(output_dir, 'wb') as f:
            pickle.dump(res, f)

        input_lines = text_test
        res = model.get_result(input_lines, layer_limit=layer_dict[args.model_name], verbose=2, \
                    output_last_hidden_states=True,
                    output_all_hidden_states=False, output_all_activations=False, 
                    output_all_pooled_hidden_states=True, output_all_pooled_activations=True)
        output_dir = f'../dataset_acts/{args.dataset}/test_all_{args.model_name}_res.pkl'
        with open(output_dir, 'wb') as f:
            pickle.dump(res, f)

    print("Activations and hidden states for each layer are successfully stored.")

if __name__ == "__main__":
    main()