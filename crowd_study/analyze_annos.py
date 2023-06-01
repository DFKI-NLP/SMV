import imgkit
import numpy as np
import pandas as pd
import seaborn as sns
import thermostat

from collections import defaultdict
from tqdm import tqdm
from typing import List


def calculate_simulation_accuracy(df):
    q = [list(df["simulation"])[i] == list(df["predicted"])[i] for i in range(len(list(df["predicted"])))]
    acc = 100 * (sum(q)/len(list(df["predicted"])))
    return acc


def average_rating(df, rating_column):
    try:
        return sum([int(r) for r in df[rating_column]])/len(df)
    except ValueError:
        print()


col_names = ["idx", "type", "text", "explanation", "simulation", "helpful", "easy"]

print("Loading datasets...")
imdb_all = []
ag_all = []

def load_anno_csv(filename, task_df_list):
    anno_df = pd.read_csv(filename, names=col_names, header=0)
    if '_G' in filename:
        pass
    else:
        anno_df['type'] = anno_df.apply(lambda row: 'HM' if pd.isnull(row['explanation']) else 'SMV', axis=1)
    task_df_list.append(anno_df)

# IMDb
for anno_file in [
        r"IMDb_G0_Anno1.csv", r"IMDb_G0_Anno0.csv", r"IMDb_G0_Anno2.csv", r"IMDb_G0_Anno3.csv",
        r"IMDb_G1_Anno4.csv", r"IMDb_G1_Anno5.csv", r"IMDb_G1_Anno6.csv",
        r"IMDb_G2_Anno7.csv", r"IMDb_G2_Anno8.csv", r"IMDb_G2_Anno9.csv",
        r"IMDb_G3_Anno2.csv", r"IMDb_G3_Anno0.csv",
    ]:
    load_anno_csv(anno_file, imdb_all)
imdb_cov = pd.read_csv("IMDb_cov_sum_dict.csv", names=['idx', 'cov_value'], header=0)

# AG News
for anno_file in [
        r"AGNews_G0_Anno1.csv", r"AGNews_G0_Anno0.csv", r"AGNews_G0_Anno2.csv", r"AGNews_G0_Anno3.csv",
        r"AGNews_G1_Anno4.csv", r"AGNews_G1_Anno5.csv", r"AGNews_G1_Anno6.csv",
        r"AGNews_G2_Anno7.csv", r"AGNews_G2_Anno8.csv", r"AGNews_G2_Anno9.csv",
        r"AGNews_G3_Anno2.csv", r"AGNews_G3_Anno0.csv",
    ]:
    load_anno_csv(anno_file, ag_all)
ag_cov = pd.read_csv("AGNews_cov_sum_dict.csv", names=['idx', 'cov_value'], header=0)

print("Loading Thermostat configs...")
imdb_config = thermostat.load('imdb-bert-lig', cache_dir="../data")
ag_config = thermostat.load('ag_news-bert-lig', cache_dir="../data")

# Subsets and settings
calculate_correlations = False
false_model_predictions = False
incorrectly_simulated = False
high_coverage = True

explanation_types = ["HM", "SMV", "ER", "GPT"]

def analyze_dataset(csv_dfs: List, thermo_config):
    print(thermo_config.dataset_name)
    dc = {}
    for etype in explanation_types:
        dc[f"{etype}_Accuracy"] = []
        dc[f"{etype}_helpful"] = []
        dc[f"{etype}_easy"] = []

    lens_input, lens_smv = None, None

    #sorted_hm_dfs, sorted_smv_dfs, sorted_er_dfs, sorted_gpt_dfs = [], [], [], []
    sorted_etype_dfs = []
    smv_advs = defaultdict(int)
    for anno_df in tqdm(csv_dfs):
        # Retrieve label predicted by model from Thermostat.
        pred_label = [thermo_config[i].predicted_label for i in list(anno_df['idx'])]
        true_label = [thermo_config[i].true_label for i in list(anno_df['idx'])]
        if 'imdb' in thermo_config.config_name:
            pred_label = ['Positive sentiment' if i == 'pos' else 'Negative sentiment' for i in pred_label]
            true_label = ['Positive sentiment' if i == 'pos' else 'Negative sentiment' for i in true_label]
            cov_info = imdb_cov
        else:
            cov_info = ag_cov

        coverage = list(cov_info["cov_value"])
        higher_cov_smvs = [cov > np.mean(coverage) for cov in coverage]
        cov_info['high'] = higher_cov_smvs

        attributions = [thermo_config[i].attributions for i in list(anno_df['idx'])]

        anno_df['predicted'] = pred_label
        anno_df['true'] = true_label
        anno_df['correct_pred'] = [True if list(anno_df["true"])[i] == list(anno_df["predicted"])[i] else False
                                   for i in range(len(list(anno_df["predicted"])))]
        anno_df['correct_sim'] = [True if list(anno_df["predicted"])[i] == list(anno_df["simulation"])[i] else False
                                  for i in range(len(list(anno_df["predicted"])))]
        anno_df['attributions'] = attributions

        # Distinguish between types of explanations
        for etype in explanation_types:
            #hm = anno_df[anno_df['explanation'].isnull()]
            #smv = anno_df[anno_df['explanation'].notnull()]
            etype_df = anno_df[anno_df['type'] == etype]
            if etype_df.empty or pd.isnull(etype_df['simulation']).all():
                continue

            #hm = hm.sort_values('idx')
            #smv = smv.sort_values('idx')
            etype_df = etype_df.sort_values('idx')

            #sorted_hm_dfs.append(hm)
            #sorted_smv_dfs.append(smv)
            sorted_etype_dfs.append(etype_df)

            #lens_smv = [len(v.replace('»', '').replace('«', '').split(' ')) for v in smv.explanation]
            #lens_input = [len(text.split(' ')) for text in smv.text]
            #abstraction_factor = [i/s for i, s in zip(lens_input, lens_smv)]

            """ [Cov] Coverage """
            etype_df = etype_df.merge(cov_info, on='idx', how='left')
            if high_coverage:
                etype_df = etype_df[etype_df["high"] == True]

            """ False model predictions """
            if false_model_predictions:
                etype_df = etype_df[etype_df["correct_pred"] == False]

            """ [|W|] Longer input length """ # TODO
            #longer_than_average_texts = [len(text.split(' ')) > np.mean(lens_input) for text in smv['text']]
            #hm = hm[longer_than_average_texts]
            #smv = smv[longer_than_average_texts]

            """ [Abs] More concise verbalizations """ # TODO
            #more_concise_smvs = [abstr > np.mean(abstraction_factor) for abstr in abstraction_factor]
            #hm = hm[more_concise_smvs]
            #smv = smv[more_concise_smvs]

            """ [Delta] Complexity of saliency map """ # TODO
            #sum_of_abs_diffs = [sum([abs(v) for v in np.diff(atts)]) for atts in list(smv.attributions)]
            #complex_sms = [soad > np.mean(sum_of_abs_diffs) for soad in sum_of_abs_diffs]
            #hm = hm[complex_sms]
            #smv = smv[complex_sms]

            """ Correctly simulated """
            if incorrectly_simulated:
                etype_df = etype_df.loc[etype_df["simulation"] != etype_df["predicted"]]

            dc[f"{etype}_Accuracy"].append(calculate_simulation_accuracy(etype_df))
            #dc["SMV_Accuracy"].append(calculate_simulation_accuracy(smv))

            dc[f"{etype}_helpful"].append(average_rating(etype_df, 'helpful'))
            #dc["SMV_helpful"].append(average_rating(smv, 'helpful'))

            dc[f"{etype}_easy"].append(average_rating(etype_df, 'easy'))
            #dc["SMV_easy"].append(average_rating(smv, 'easy'))

    if lens_input and lens_smv:
        pass
        #print(f"Average length of inputs: {np.mean(lens_input)}")
        #print(f"Average length of SMVs: {np.mean(lens_smv)}")
        #print(f"Average abstraction level: {np.mean(abstraction_factor)}")

    for key in dc.keys():
        print(f"{key}: {np.mean(dc[key])}")

    if smv_advs:
        smv_adv_df = pd.DataFrame(smv_advs.items(), columns=['idx', 'accum_rating'])

    """ Correlations """
    def map_boolean_to_int(li):
        return [1 if i else 0 for i in li]

    if calculate_correlations:
        corr_arrays = {'|S_V|': lens_smv,
                       '|W|': lens_input,
                       #'Abs': abstraction_factor,
                       'Cov': coverage,
                       #'δ': sum_of_abs_diffs,
                       'y = ŷ': map_boolean_to_int(list(etype_df['correct_pred']))}
        cc_hm = pd.concat(sorted_etype_dfs[0])
        cc_smv = pd.concat(sorted_etype_dfs[1])
        for expl_prefix, cc_df in zip(["HM", "SMV"], [cc_hm, cc_smv]):
            for taf, anno_field in zip(['A', 'B1', 'B2'], ['correct_sim', 'helpful', 'easy']):
                mean_agg = cc_df.groupby('idx', as_index=False).agg({anno_field: 'mean'})
                corr_arrays[expl_prefix + ' ' + taf] = list(mean_agg[anno_field])

        corr_df = pd.DataFrame(corr_arrays)
        correlations = corr_df.corr()
        if 'imdb' in thermo_config.config_name:
            cm = sns.light_palette("gold", as_cmap=True)
        else:
            cm = sns.light_palette("seagreen", as_cmap=True)
        table = correlations.style.background_gradient(cmap=cm).set_precision(2)
        html = table.render()
        imgkit.from_string(html, f'correlations_{thermo_config.config_name}.png')
    print()


print("Analyzing annotations...")
analyze_dataset(imdb_all, imdb_config)
analyze_dataset(ag_all, ag_config)

""" Inter-annotator agreement """
annotators_coll = defaultdict(list)


def collect_ratings(dataset):
    for i, annotator in enumerate(dataset):
        annotators_coll[i] += list(annotator['helpful'])
        annotators_coll[i] += list(annotator['easy'])


collect_ratings(imdb_all)
collect_ratings(ag_all)
annotators_coll_list = [ac for ac in annotators_coll.values()]
print()
