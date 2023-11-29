from imports import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go

hs_layer_dict = {"distilbert": 7, "roberta": 13, "gpt2-xl": 50, "gpt2": 14, "gpt2-medium": 26, "gpt2-large": 38}
act_layer_dict = {"distilbert": 6, "roberta": 12, "gpt2-xl": 48, "gpt2": 12, "gpt2-medium": 24, "gpt2-large": 36}

def plot_what_which_where(model_name, dataset, is_finetuned):
    if is_finetuned:
        with open(f'../gpt-activations/dataset_acts/{dataset}/new_agg/{model_name}_lr_agg_neurons_finetuned.pkl', 'rb') as f:
            data_dict = pickle.load(f)
    else:
        with open(f'../gpt-activations/dataset_acts/{dataset}/new_agg/{model_name}_lr_agg_neurons.pkl', 'rb') as f:
            data_dict = pickle.load(f)

    data_list = []
    for key, value in data_dict.items():
        if len(key) == 6:
            data_list.append({'representation': key[0], 
                            'layer': key[1], 
                            'pooling_choice': key[2], 
                            'threshold': key[3], 
                            'num_selected_features': key[4], 
                            'split': key[5], 
                            'accuracy': value})

    df = pd.DataFrame(data_list)


    test_df = df[df['split'] == 'test']
    #test_df['log_threshold'] = np.log1p(test_df['threshold'])

    test_df_hs = test_df[test_df['representation'] == 'hs']
    test_df_act = test_df[test_df['representation'] == 'act']

    num_layers = act_layer_dict[model_name]
    baseline_solid = 0.9013 # corresponding frozen head performance
    baseline_solid_name = 'frozen baseline'
    baseline_dash = 0.9274 # corresponding finetuned head performance 
    baseline_dash_name = 'fine-tuned baseline'

    fig = make_subplots(rows=2, cols=4, 
        shared_xaxes=True, shared_yaxes=True,
        column_widths=[0.08,0.42,0.08,0.42], row_heights=[0.72,0.28],
        horizontal_spacing=0.01, vertical_spacing=0
    )

    fig.add_trace(go.Scatter(
            x=test_df_hs['pooling_choice'],
            y=test_df_hs['accuracy'], 
            mode='markers',
            marker=dict(
                color=test_df_hs['layer'], coloraxis='coloraxis',
                size=(test_df_hs['threshold']*10000)**0.32,
                symbol=test_df_hs['pooling_choice'],
            ),
            showlegend=False
        ),
        row=1, col=1)
    fig.add_trace(go.Scatter(
            x=test_df_hs['num_selected_features'],
            y=test_df_hs['accuracy'], 
            mode='markers',
            marker=dict(
                color=test_df_hs['layer'], coloraxis='coloraxis',
                size=(test_df_hs['threshold']*10000)**0.32,
                symbol=test_df_hs['pooling_choice'],
            ),
            showlegend=False
        ),
        row=1, col=2)
    fig.add_trace(go.Scatter(
            x=test_df_act['pooling_choice'],
            y=test_df_act['accuracy'], 
            mode='markers',
            marker=dict(
                color=test_df_act['layer'], coloraxis='coloraxis',
                size=(test_df_act['threshold']*10000)**0.32,
                symbol=test_df_act['pooling_choice'],
            ),
            showlegend=False
        ),
        row=1, col=3)
    fig.add_trace(go.Scatter(
            x=test_df_act[test_df_act['pooling_choice']==0]['num_selected_features'], 
            y=test_df_act[test_df_act['pooling_choice']==0]['accuracy'], 
            mode='markers',
            marker=dict(
                color=test_df_act[test_df_act['pooling_choice']==0]['layer'], coloraxis='coloraxis',
                size=(test_df_act[test_df_act['pooling_choice']==0]['threshold']*10000)**0.32,
                symbol=test_df_act[test_df_act['pooling_choice']==0]['pooling_choice'],
            ),
            name='single token',
            showlegend=True
        ),
        row=1, col=4)
    fig.add_trace(go.Scatter(
            x=test_df_act[test_df_act['pooling_choice']==1]['num_selected_features'], 
            y=test_df_act[test_df_act['pooling_choice']==1]['accuracy'], 
            mode='markers',
            marker=dict(
                color=test_df_act[test_df_act['pooling_choice']==1]['layer'], coloraxis='coloraxis',
                size=(test_df_act[test_df_act['pooling_choice']==1]['threshold']*10000)**0.32,
                symbol=test_df_act[test_df_act['pooling_choice']==1]['pooling_choice'],
                #opacity=0.8
            ),
            name='max pooling',
            showlegend=True
        ),
        row=1, col=4)
    fig.add_trace(go.Scatter(
            x=test_df_act[test_df_act['pooling_choice']==2]['num_selected_features'], 
            y=test_df_act[test_df_act['pooling_choice']==2]['accuracy'], 
            mode='markers',
            marker=dict(
                color=test_df_act[test_df_act['pooling_choice']==2]['layer'], coloraxis='coloraxis',
                size=(test_df_act[test_df_act['pooling_choice']==2]['threshold']*10000)**0.32,
                symbol=test_df_act[test_df_act['pooling_choice']==2]['pooling_choice'],
                #opacity=0.8
            ),
            name='avg pooling',
            showlegend=True
        ),
        row=1, col=4)

    fig.add_trace(go.Scatter(
            x=test_df_hs['num_selected_features'],
            y=test_df_hs['layer'], 
            mode='markers',
            marker=dict(
                color=test_df_hs['layer'], coloraxis='coloraxis',
                size=(test_df_hs['threshold']*10000)**0.32,
                symbol=test_df_hs['pooling_choice'],
            ),
            showlegend=False
        ),
        row=2, col=2)
    fig.add_trace(go.Scatter(
            x=test_df_act['num_selected_features'],
            y=test_df_act['layer'], 
            mode='markers',
            marker=dict(
                color=test_df_act['layer'], coloraxis='coloraxis',
                size=(test_df_act['threshold']*10000)**0.32,
                symbol=test_df_act['pooling_choice'],
            ),
            showlegend=False
        ),
        row=2, col=4)



    fig.add_hline(y=baseline_solid, line_dash='0', line_color="rgba(0,0,0,0.2)", row=1, col='all')
    fig.add_hline(y=baseline_dash, line_dash='6', line_color="rgba(0,0,0,0.2)", row=1, col='all')
    if baseline_dash < baseline_solid:
        baseline_dash_position = 'bottom left'
        baseline_solid_position = 'top left'
    else:
        baseline_dash_position = 'top left'
        baseline_solid_position = 'bottom left'
    fig.add_hline(y=baseline_solid, line_dash='0', line_color="rgba(0,0,0,0)", row=1, col=2,
        annotation_text=baseline_solid_name, annotation_position=baseline_solid_position)
    fig.add_hline(y=baseline_dash, line_dash='6', line_color="rgba(0,0,0,0)", row=1, col=2,
        annotation_text=baseline_dash_name, annotation_position=baseline_dash_position)
    fig.add_hline(y=baseline_solid, line_dash='0', line_color="rgba(0,0,0,0)", row=1, col=4,
        annotation_text=baseline_solid_name, annotation_position=baseline_solid_position)
    fig.add_hline(y=baseline_dash, line_dash='6', line_color="rgba(0,0,0,0)", row=1, col=4,
        annotation_text=baseline_dash_name, annotation_position=baseline_dash_position)


    fig.update_yaxes(range=[0.45,1], row=1, col=1, title_text='Accuracy', automargin=True)
    fig.update_xaxes(type='log', row=1, col=2)
    fig.update_xaxes(type='log', row=1, col=4)

    fig.update_yaxes(autorange=True, row=2, col=2, ticklabelstep=1, showticklabels=True, title_text='aggregated layers', automargin=True)
    fig.update_yaxes(row=2, col=2, ticklabelstep=1, showticklabels=True)
    fig.update_xaxes(type='log', row=2, col=2, title_text='Number of aggregated hidden state features', automargin=True)
    fig.update_xaxes(type='log', row=2, col=4, title_text='Number of aggregated activation features', automargin=True)


    fig.update_coloraxes(
        colorbar=dict(orientation='h', len=0.5,
                    thickness=16, dtick=max(num_layers/12,1), 
                    title_text='Number of aggregated layers', #title_side='top',
                    y=1.02))

    fig.update_layout(
        template='plotly_white',
        width=1200, height=720,
        coloraxis=dict(colorscale=px.colors.diverging.Portland),
        legend=dict(title="", orientation="v",
            xanchor="right", yanchor="bottom", x=1, y=0.32),    
        font=dict(size=12)
    )

    fig.show()
    if is_finetuned:
        fig.write_image(f'../gpt-activations/dataset_acts/{dataset}/what-which-where/{model_name}_finetuned.pdf')
    else:
        fig.write_image(f'../gpt-activations/dataset_acts/{dataset}/what-which-where/{model_name}_frozen.pdf')