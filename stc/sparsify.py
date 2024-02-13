from imports import *
from concurrent import futures
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import pdb

warnings.filterwarnings("ignore")
hs_layer_dict = {"distilbert": 7, "roberta": 13, "gpt2-xl": 50, "gpt2": 13, "gpt2-medium": 26, "gpt2-large": 38}
act_layer_dict = {"distilbert": 6, "roberta": 12, "gpt2-xl": 48, "gpt2": 12, "gpt2-medium": 24, "gpt2-large": 36}
pooling_dict = {0: "first_token", 1: "max_pooling", 2: "avg_pooling"}
f1_score_macro = partial(f1_score, average='macro')
metrics_dict = {"imdb": accuracy_score, "edos": f1_score_macro, "sst-2": accuracy_score}

def linear_probe_worker(dataset, layer, pooling_choice, C, y_train_full, y_val_full, train_res, val_res, rep, iter_interval, max_iter_increment):
    if rep == "hs":
        X_train_layer = train_res[1][layer,:,:,pooling_choice].detach()
        X_val_layer = val_res[1][layer,:,:,pooling_choice].detach()
    else:
        X_train_layer = train_res[2][layer,:,:,pooling_choice].detach()
        X_val_layer = val_res[2][layer,:,:,pooling_choice].detach()
    clf = LogisticRegression(max_iter=iter_interval, C=C, penalty='elasticnet', l1_ratio=1, solver='saga', multi_class='ovr', n_jobs=-1, random_state=42, warm_start=True)
    best_val_acc = 0
    for i in range(max_iter_increment):
        clf.fit(X_train_layer, y_train_full)
        tmp_val_acc = metrics_dict[dataset](y_val_full, clf.predict(X_val_layer))
        if best_val_acc < tmp_val_acc:
            best_val_acc = tmp_val_acc   
    return (layer, pooling_choice, C, best_val_acc, clf)

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
    else:
        train_pkl = f'../dataset_acts/{dataset}/train_all_{model_name}_res.pkl'
        test_pkl = f'../dataset_acts/{dataset}/test_all_{model_name}_res.pkl'
        val_pkl = f'../dataset_acts/{dataset}/val_all_{model_name}_res.pkl'
        clf_output_dir = f'../dataset_acts/{dataset}/{model_name}_trained_lr.pkl'

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
    default_C_values = [0.2, 1, 10, 100] # for L1-regularization
    layer_range = range(hs_layer_dict[model_name])
    pooling_choices_range = range(pooling_choices)
    combinations = list(itertools.product(layer_range, pooling_choices_range, default_C_values))
    all_clf_dict = defaultdict()

    #pdb.set_trace()
    rep = "hs"
    best_val_acc_per_layer = {}
    best_clf_per_layer = {}
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(linear_probe_worker, dataset, layer, pooling_choice, C, y_train_full, y_val_full, train_res, val_res, rep, iter_interval, max_iter_increment) for layer, pooling_choice, C in combinations]
        #results = [future.result() for future in futures]
        for future in concurrent.futures.as_completed(futures):
            layer, pooling_choice, C, val_acc, clf = future.result()
            hs_key = (rep, pooling_choice, layer)
            all_clf_dict[hs_key] = clf
            if layer not in best_val_acc_per_layer or best_val_acc_per_layer[layer] < val_acc:
                best_val_acc_per_layer[layer] = val_acc
                best_clf_per_layer[layer] = (clf, pooling_choice)
    best_layer = max(best_val_acc_per_layer, key=best_val_acc_per_layer.get)
    best_clf, best_pooling_choice = best_clf_per_layer[best_layer]

    X_test_layer = test_res[1][best_layer, :, :, best_pooling_choice].detach()
    y_pred = best_clf.predict(X_test_layer)
    test_performance = metrics_dict[dataset](y_test_full, y_pred)
    print(f"Hidden States Best Layer-wise Probing Performance = {test_performance} on layer {best_layer}, pooling choice {pooling_dict[best_pooling_choice]}.")
    #pdb.set_trace()

    rep = "act"
    best_val_acc_per_layer = {}
    best_clf_per_layer = {}
    layer_range = range(act_layer_dict[model_name])
    combinations = list(itertools.product(layer_range, pooling_choices_range, default_C_values))
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(linear_probe_worker, dataset, layer, pooling_choice, C, y_train_full, y_val_full, train_res, val_res, rep, iter_interval, max_iter_increment) for layer, pooling_choice, C in combinations]
        #results = [future.result() for future in futures]
        for future in concurrent.futures.as_completed(futures):
            layer, pooling_choice, C, val_acc, clf = future.result()
            hs_key = (rep, pooling_choice, layer)
            all_clf_dict[hs_key] = clf
            if layer not in best_val_acc_per_layer or best_val_acc_per_layer[layer] < val_acc:
                best_val_acc_per_layer[layer] = val_acc
                best_clf_per_layer[layer] = (clf, pooling_choice)
    best_layer = max(best_val_acc_per_layer, key=best_val_acc_per_layer.get)
    best_clf, best_pooling_choice = best_clf_per_layer[best_layer]

    X_test_layer = test_res[2][best_layer, :, :, best_pooling_choice].detach()
    y_pred = best_clf.predict(X_test_layer)
    test_performance = metrics_dict[dataset](y_test_full, y_pred)
    print(f"Activations Best Layer-wise Probing Performance = {test_performance} on layer {best_layer}, pooling choice {pooling_dict[best_pooling_choice]}.")

    #pdb.set_trace()
    with open(clf_output_dir, 'wb') as f:
        pickle.dump(clf_dict, f)

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