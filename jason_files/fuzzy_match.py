# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
from Levenshtein import ratio as similarity_score


# A function returns list of near-matched predictive terms for each term in the data
# Args:
#   data: Dataframe containing REPORTED_TERM which is the input text that we are trying to map to an AE.
#   target: String, name of target variable, i.e. the MedDRA term level that we are trying to predict - e.g. 'PREFERRED_TERM'.
#   Meddra_dict: Dataframe containing complete Meddra dictionary
#   similarity_threshold: Real number from 0 to 1, higher means more similar.
#                         It limits the output to near-matched predictive terms above the similarity threshold.
#   excl_stopwords: Boolean; if true, delete stopwords from reported term and target before matching.
#
# Output:
#   Return a list of lists of potential matched predictive terms with the corresponding similarity score
def fuzzy_matching_potential(data, target, Meddra_dict, similarity_threshold=0.5, excl_stopwords=False):
    # Select all values from target column in the dataframe Meddra_dict
    # Convert to all-caps - Levenshtein distance is case sensitive and will fail to match 'WORD' to 'Word', for example
    all_target_val = [str(x) for x in Meddra_dict[target].str.upper().values if str(x) != 'nan']

    # Select all values from 'REPORTED_TERM' column in the dataframe data
    # Convert to all-caps - Levenshtein distance is case sensitive and will fail to match 'WORD' to 'Word', for example
    all_data_val = data['REPORTED_TERM'].str.upper().values

    # Delete stopwords, if requested
    if excl_stopwords:
        all_target_val = delete_stopwords(all_target_val, STOPWORDS)
        all_data_val = delete_stopwords(all_data_val, STOPWORDS)

    # potential_matched_target is a list of tuples being (nearest-matched target term, similarity measure)
    # it gets one predictive term that's most similar to a reported term
    potential_matched_target = [[] for i in all_data_val]

    # looping through all the data and potential matched predicted terms
    for data_itr, each_data_val in enumerate(all_data_val):
        for each_target_val in all_target_val:
            # find a similarity measure between each reported_term and each potential predicted term
            # token_sort_ratio works for alphanumeric tokens in a string and also take unordered string elements into consideration
            similarity = similarity_score(str(each_data_val), str(each_target_val))

            # if similarity measure is above the threshold, save potential matched terms and similarity into the output list
            if similarity > similarity_threshold:
                try:
                    potential_matched_target[data_itr].append((each_target_val, similarity))
                except:
                    potential_matched_target[data_itr] = [(each_target_val, similarity)]
            else:
                continue

    return potential_matched_target


# select_most_similar_one takes the output from near_matching / near_matching_baseline as input,
# and returns the nearest-matched term for each reported term

# Args:
#   output_ls is the output from near_matching, aka a list of lists
#
# Output:
#   a list of tuples with the most similar predicted term and the corresponding similarity measure,
#   emtpy tuple can be returned if output_ls contains no potential matched term for a prefered term
def select_most_similar_one(output_ls):
    n_data = len(output_ls)
    output_df = pd.DataFrame(columns=['term', 'conf'], index=range(n_data))

    for i in range(n_data):
        sims = [t[1] for t in
                output_ls[i]]  # similarities between this term and all other terms that meet the similarity threshold
        if len(sims) > 0:  # a may be empty due to similarity threshold argument applied in another fucntion
            output_df.iloc[i] = list(output_ls[i][sims.index(max(sims))])
        # else, this row remains N/A.

    return output_df


# Takes the results from fuzzy_matching and evaluates accuracy

# Args:
#   matching_results: Results from running the function fuzzy_matching
#   data: Dataframe containing REPORTED_TERM which is the input text that we are trying to map to an AE.
#   target: String, name of target variable, i.e. the MedDRA term level that we are trying to predict - e.g. 'PREFERRED_TERM'.
#
# Output:
#   Return a tuple of two elements: the accuracy (a string, e.g. '75%'), and a dataframe with complete results
def calc_fuzzy_matching_accuracy(matching_results, data, target):
    single_output = select_most_similar_one(matching_results)
    output_df = single_output
    output_df.columns = ['Predicted_term', 'confidence']
    return output_df


# Runs fuzzy matching algorithm and returns a dataframe with the best matched MedDRA terms
# Args:
#   data: Dataframe containing REPORTED_TERM which is the input text that we are trying to map to an AE.
#   target: String, name of target variable, i.e. the MedDRA term level that we are trying to predict - e.g. 'PREFERRED_TERM'.
#   Meddra_dict: Dataframe containing complete Meddra dictionary
#   similarity_threshold: Real number from 0 to 1, higher means more similar.
#                         It limits the output to near-matched predictive terms above the similarity threshold.
#   excl_stopwords: Boolean; if true, delete stopwords from reported term and target before matching.
#
# Output:
#   Return a dataframe with complete results
def fuzzy_matching(data, target, Meddra_dict, similarity_threshold=0.5, excl_stopwords=False):
    matching_results = fuzzy_matching_potential(data, target, Meddra_dict, similarity_threshold=0.5,
                                                excl_stopwords=False)
    results_llt = calc_fuzzy_matching_accuracy(matching_results, data, target)
    results_llt.columns = [target + '_PRED', 'CONF']
    results_llt['REPORTED_TERM'] = data['REPORTED_TERM'].values
    return results_llt

