from imports import *

warnings.filterwarnings("ignore")
hs_layer_dict = {"distilbert": 7, "roberta": 13, "gpt2-xl": 50, "gpt2": 14, "gpt2-medium": 26, "gpt2-large": 38}
act_layer_dict = {"distilbert": 6, "roberta": 12, "gpt2-xl": 48, "gpt2": 12, "gpt2-medium": 24, "gpt2-large": 36}
pooling_dict = {0: "first_token", 1: "max_pooling", 2: "avg_pooling"}
f1_score_macro = partial(f1_score, average='macro')
metrics_dict = {"imdb": accuracy_score, "edos": f1_score_macro, "sst-2": accuracy_score}

def linear_probe(model_name, is_finetuned, dataset, iter_interval=2, max_iter_increment=16):
    if model_name.lower().startswith("flan"):
        pooling_choices = 4
    else:
        pooling_choices = 3

    if is_finetuned:
        train_pkl = f'../dataset_acts/{dataset}/train_all_{model_name}_finetuned_res.pkl'
        test_pkl = f'../dataset_acts/{dataset}/test_all_{model_name}_finetuned_res.pkl'
        val_pkl = f'../dataset_acts/{dataset}/val_all_{model_name}_finetuned_res.pkl'
        clf_output_dir = f'../dataset_acts/{dataset}/{model_name}_trained_lr_finetuned.pkl'
        score_output_dir = f'../dataset_acts/{dataset}/performance/{model_name}_lr_finetuned.pkl'
    else:
        train_pkl = f'../dataset_acts/{dataset}/train_all_{model_name}_res.pkl'
        test_pkl = f'../dataset_acts/{dataset}/test_all_{model_name}_res.pkl'
        val_pkl = f'../dataset_acts/{dataset}/val_all_{model_name}_res.pkl'
        clf_output_dir = f'../dataset_acts/{dataset}/{model_name}_trained_lr.pkl'
        score_output_dir = f'../dataset_acts/{dataset}/performance/{model_name}_lr.pkl'

    with open(train_pkl, 'rb') as f:
        train_res = pickle.load(f)
    with open(test_pkl, 'rb') as f:
        test_res = pickle.load(f)
    with open(val_pkl, 'rb') as f:
        val_res = pickle.load(f)
    
    text_train, text_val, text_test, label_train, label_val, label_test = my_train_test_split(dataset)
    y_train_full = [label_train[i] for i in train_res[0]]
    y_test_full = [label_test[i] for i in test_res[0]]
    y_val_full = [label_val[i] for i in val_res[0]]
    accuracy_scores = defaultdict(float)
    val_acc_hs = defaultdict(float)
    val_acc_act = defaultdict(float)
    clf_dict = defaultdict()

    default_C_values = [0.2, 0.5, 1, 5, 10, 100] # for L1-regularization
    for layer in range(hs_layer_dict[model_name]):
        best_clf_all=0
        #best_layer=-1
        best_iter_all=0
        best_pooling_choice=-1
        best_val_acc_all=0
        #best_C_all=-1
        for pooling_choice in range(0, 3, 1):
            X_train_layer = train_res[1][layer,:,:,pooling_choice].detach()
            X_val_layer = val_res[1][layer,:,:,pooling_choice].detach()
            X_test_layer = test_res[1][layer,:,:,pooling_choice].detach()
            best_clf=0
            best_iter=0
            best_val_acc=0
            #best_C=0
            for C in default_C_values:
                clf = LogisticRegression(max_iter=iter_interval, C=C, penalty='elasticnet', l1_ratio=1, solver='saga', multi_class='ovr', n_jobs=-1, random_state=42, warm_start=True)
            
                for i in range(max_iter_increment):
                    clf.fit(X_train_layer, y_train_full)
                    tmp_val_acc = metrics_dict[dataset](y_val_full, clf.predict(X_val_layer))
                    if best_val_acc < tmp_val_acc:
                        best_clf = clf
                        best_iter = int((i+1)*iter_interval)
                        best_val_acc = tmp_val_acc
                        best_C = C
                    if best_val_acc_all < tmp_val_acc:
                        best_clf_all = clf
                        best_val_acc_all = tmp_val_acc
                        best_iter_all = int((i+1)*iter_interval)
                        best_pooling_choice = pooling_choice
                        best_C = C
                        #best_layer = layer
                        #best_train_acc = accuracy_score(y_train, clf.predict(X_train))
            
            clf_dict[("hs", pooling_choice, layer)] = best_clf
        pooling = pooling_dict[best_pooling_choice]
        print("Hidden States Best Combination of Hyperparams: Layer {0}, Pooling choice {1}, Best iter {2}, Best validation performance {3}".format(layer, pooling, best_iter_all, best_val_acc_all))
        #X_test_layer = val_res[1][layer, :, :, best_pooling_choice].detach()
        #y_pred = best_clf.predict(X_test_layer)
        #test_acc = accuracy_score(y_val_full, y_pred)
        val_acc_hs[("hs", best_pooling_choice, layer)] = best_val_acc
        
    best_val_key = max(val_acc_hs, key=val_acc_hs.get)
    _rep, _pool, best_layer = best_val_key
    best_clf = clf_dict[best_val_key]
    X_test_layer = test_res[1][best_layer, :, :, _pool].detach()
    y_pred = best_clf.predict(X_test_layer)
    best_layer_test_acc = metrics_dict[dataset](y_test_full, y_pred)
    print("Hidden States Best Layer-wise Probing Performance = {0} on layer {1}, pooling choice {2}.".format(best_layer_test_acc, best_layer, pooling_dict[_pool]))

    for layer in range(act_layer_dict[model_name]):
        best_clf_all=0
        #best_layer=-1
        best_iter_all=0
        best_pooling_choice=-1
        best_val_acc_all=0
        best_C_all=-1
        for pooling_choice in range(0, 3, 1):
            X_train_layer = train_res[2][layer,:,:,pooling_choice].detach()
            X_val_layer = val_res[2][layer,:,:,pooling_choice].detach()
            X_test_layer = test_res[2][layer,:,:,pooling_choice].detach()
            best_clf=0
            best_iter=0
            best_val_acc=0

            for C in default_C_values:
                clf = LogisticRegression(max_iter=iter_interval, C=C, penalty='elasticnet', l1_ratio=1, solver='saga', multi_class='ovr', n_jobs=-1, random_state=42, warm_start=True)
            
                for i in range(max_iter_increment):
                    clf.fit(X_train_layer, y_train_full)
                    tmp_val_acc = accuracy_score(y_val_full, clf.predict(X_val_layer))
                    if best_val_acc < tmp_val_acc:
                        best_clf = clf
                        best_iter = int((i+1)*iter_interval)
                        best_val_acc = tmp_val_acc
                    if best_val_acc_all < tmp_val_acc:
                        best_clf_all = clf
                        best_val_acc_all = tmp_val_acc
                        best_iter_all = int((i+1)*iter_interval)
                        best_pooling_choice = pooling_choice
                        #best_layer = layer
                        #best_train_acc = accuracy_score(y_train, clf.predict(X_train))
            
            clf_dict[("act", pooling_choice, layer)] = best_clf
        pooling = pooling_dict[best_pooling_choice]
        print("Activations Best Combination of Hyperparams: Layer {0}, Pooling choice {1}, Best iter {2}, Best validation performance {3}".format(layer, pooling, best_iter, best_val_acc))
        #X_test_layer = val_res[2][layer, :, :, pooling_choice].detach()
        #y_pred = best_clf.predict(X_test_layer)
        #test_acc = accuracy_score(y_val_full, y_pred)
        val_acc_act[("act", best_pooling_choice, layer)] = best_val_acc
        
    best_val_key = max(val_acc_act, key=val_acc_act.get)
    _rep, _pool, best_layer = best_val_key
    best_clf = clf_dict[best_val_key]
    X_test_layer = test_res[2][best_layer, :, :, _pool].detach()
    y_pred = best_clf.predict(X_test_layer)
    best_layer_test_acc = accuracy_score(y_test_full, y_pred)
    print("Activations Best Layer-wise Probing Performance = {0} on layer {1}, pooling choice {2}.".format(best_layer_test_acc, best_layer, pooling_dict[_pool]))

    for key, value in val_acc_hs.items():
        accuracy_scores[key] = value
    for key, value in val_acc_act.items():
        accuracy_scores[key] = value
    with open(clf_output_dir, 'wb') as f:
        pickle.dump(clf_dict, f)
    with open(score_output_dir, 'wb') as f:
        pickle.dump(accuracy_scores, f)

def main() -> None:
    parser = argparse.ArgumentParser(description='Initialize analysis for specified model and dataset.')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='The name of the model. Options: distilbert, roberta, gpt2-base, gpt2-medium, gpt2-large, gpt2-xl')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='The name of the dataset. Options: imdb, edos, sst-2')
    parser.add_argument('--is_finetuned', type=int, required=True,
                        help='Flag for finetuned models. Options: 1 for finetuned, 0 for frozen')
    args = parser.parse_args()
    linear_probe(args.model_name, args.is_finetuned, args.dataset, iter_interval=2, max_iter_increment=16)

if __name__ == "__main__":
    main()