import matplotlib.pyplot as plt
import numpy as np

def plot_rmse_comparison(rmse_summary):
    methods = ['PCA', 'Autoencoder']
    colors = ['#7dc4fa', '#99e6ff']

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    all_vals = [v for model_vals in rmse_summary.values() for pair in model_vals.values() for v in pair]
    ymin = min(all_vals) - 0.05
    ymax = max(all_vals) + 0.3

    bars_legend = []

    for idx, (model, embeddings) in enumerate(rmse_summary.items()):
        ax = axs[idx]
        labels = list(embeddings.keys())
        pca_vals = [v[0] for v in embeddings.values()]
        ae_vals = [v[1] for v in embeddings.values()]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, pca_vals, width, label='PCA', color=colors[0])
        bars2 = ax.bar(x + width/2, ae_vals, width, label='Autoencoder', color=colors[1])

        if idx == 0:
            bars_legend = [bars1[0], bars2[0]]

        ax.set_title(model.upper(), fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=14)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=13)

    fig.suptitle('Сравнение RMSE по методам агрегации для каждой модели', fontsize=20)
    fig.legend(bars_legend, ['PCA', 'Autoencoder'], loc='lower center', ncol=2, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def plot_matching_comparison(number_of_matching_summary):
    models = ['rf', 'catboost', 'xgb', 'cnn']
    model_names = {
        'rf': 'RandomForest',
        'catboost': 'CatBoost',
        'xgb': 'XGBoost',
        'cnn': 'CNN'
    }
    colors = ['#7dc4fa', '#99e6ff']

    global_vals = []
    for model in models:
        for emb in number_of_matching_summary[model]:
            match_pca = number_of_matching_summary[model][emb][0]
            match_ae = number_of_matching_summary[model][emb][1]

            total_pca = sum(match_pca.values())
            total_ae = sum(match_ae.values())

            if total_pca > 0:
                global_vals.append(100 * sum(v for k, v in match_pca.items() if int(k) >= 1) / total_pca)
                global_vals.append(100 * sum(v for k, v in match_pca.items() if int(k) >= 2) / total_pca)
            if total_ae > 0:
                global_vals.append(100 * sum(v for k, v in match_ae.items() if int(k) >= 1) / total_ae)
                global_vals.append(100 * sum(v for k, v in match_ae.items() if int(k) >= 2) / total_ae)

    global_min = min(global_vals) - 5
    global_max = max(global_vals) + 5

    fig, axs = plt.subplots(2, 2, figsize=(30, 18))
    axs = axs.flatten()

    for idx, model in enumerate(models):
        ax = axs[idx]
        name = model_names[model]

        embeddings = list(number_of_matching_summary[model].keys())
        pca_vals_1, pca_vals_2 = [], []
        ae_vals_1, ae_vals_2 = [], []

        for emb in embeddings:
            match_pca = number_of_matching_summary[model][emb][0]
            match_ae = number_of_matching_summary[model][emb][1]
            total_pca = sum(match_pca.values())
            total_ae = sum(match_ae.values())

            at_least_1_pca = sum(v for k, v in match_pca.items() if int(k) >= 1)
            at_least_2_pca = sum(v for k, v in match_pca.items() if int(k) >= 2)
            at_least_1_ae = sum(v for k, v in match_ae.items() if int(k) >= 1)
            at_least_2_ae = sum(v for k, v in match_ae.items() if int(k) >= 2)

            pca_vals_1.append(round(100 * at_least_1_pca / total_pca, 2) if total_pca > 0 else 0)
            pca_vals_2.append(round(100 * at_least_2_pca / total_pca, 2) if total_pca > 0 else 0)
            ae_vals_1.append(round(100 * at_least_1_ae / total_ae, 2) if total_ae > 0 else 0)
            ae_vals_2.append(round(100 * at_least_2_ae / total_ae, 2) if total_ae > 0 else 0)

        x = np.arange(len(embeddings))
        width = 0.2

        bars1 = ax.bar(x - 1.6*width, pca_vals_1, width, label='PCA ≥1', color=colors[0])
        bars2 = ax.bar(x - 0.5*width, pca_vals_2, width, label='PCA ≥2', color=colors[0], hatch='//')
        bars3 = ax.bar(x + 0.5*width, ae_vals_1, width, label='AE ≥1', color=colors[1])
        bars4 = ax.bar(x + 1.6*width, ae_vals_2, width, label='AE ≥2', color=colors[1], hatch='//')

        ax.set_title(name, fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(embeddings, rotation=45, ha='right', fontsize=14)
        ax.set_ylabel('Процент', fontsize=14)
        ax.set_ylim(global_min, global_max)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 7),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=12)

    fig.suptitle('Раздельный процент совпадений ≥1 и ≥2 по методам агрегации и эмбеддингам', fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
