import utils.scaffold as usc
import utils.chem as uc
from collections import namedtuple
import pandas as pd

batch_size =1 


def _cleanup_decoration(dec_smi):
    dec_mol = uc.to_mol(dec_smi)
    if not dec_mol:
        return None
    return usc.to_smiles(usc.remove_attachment_point_numbers(dec_mol))


def _join_scaffold(scaff, decs):
    try:
        mol = usc.join_joined_attachments(scaff, decs)
        if mol:
            return usc.to_smiles(mol)
        else:
            return "Failed"
    except:
        return "Failed"


def _format_attachment_point(smi, num):
    smi = usc.add_first_attachment_point_number(smi, num)
    return usc.to_smiles(uc.to_mol(smi))  # canonicalize


def _create_decorations_map(decorations_smi, attachment_points):
            decorations = decorations_smi.split(usc.ATTACHMENT_SEPARATOR_TOKEN)
            return str({idx: _cleanup_decoration(dec) for dec, idx in zip(decorations, attachment_points)})



def initialize_results(scaffs):
    cols = ["smiles", "scaffold", "decorations", "count"]
    results_df = pd.DataFrame({"smiles" : scaffs, "scaffold" : scaffs, "decorations" :[{} for i in range(len(scaffs))]})
    return results_df


### main ###
def fast_join(input_scaffolds, sample_model):
    RETRY = 10
    # print(RETRY)
    FAILED = False

    results_df = initialize_results(input_scaffolds)
    scaffold_df = results_df[["smiles", "scaffold", "decorations"]]

    if len(scaffold_df)>0:
        scaffold_df["attachment_points"] = scaffold_df.apply(lambda x: usc.get_attachment_points(x["smiles"]), axis=1)
        scaffold_df["randomized_scaffold"] = scaffold_df.apply(lambda x: usc.remove_attachment_point_numbers(x["smiles"]), axis=1)
        scaffold_df['idx'] = scaffold_df.index

    scaffolds = scaffold_df[['idx', 'randomized_scaffold']]
    scaffold_buffer = []
    idx_buffer = []
    out_file = []
    while True:
        for idx, row in scaffolds.iterrows():
            num_decorations_per_scaffold = str(row['randomized_scaffold']).count("*")*2
            scaffold_buffer += [row['randomized_scaffold']]*num_decorations_per_scaffold
            idx_buffer +=[row['idx']]*num_decorations_per_scaffold
            if len(scaffold_buffer) > batch_size:
                for idx, (scaff, dec, nll) in zip(idx_buffer, sample_model.run(scaffold_buffer)):
                    out_file.append((idx, scaff, dec, nll))
                    idx_buffer = []
                    scaffold_buffer=[]
        sample_df = pd.DataFrame({'id': [_[0] for _ in out_file], 'decoration_smi': [_[2] for _ in out_file], 'nll':[_[3] for _ in out_file]})
        final_df = sample_df.join(scaffold_df, on='id')


        ### multi ###
        # final_df['decorations'] = final_df.apply(lambda x:_format_attachment_point (x['decoration_smi'], x['attachment_points'][0]), axis=1)
        # print(final_df['decorations'])
        # final_df['smiles'] = final_df.apply(lambda x: _join_scaffold(x['smiles'], x['decorations']), axis=1)
        # print(final_df['smiles'])

        ### single ###   
        final_df['smiles'] = final_df.apply(lambda x: _join_scaffold(x['randomized_scaffold'], x['decoration_smi']), axis=1)
        final_df = final_df[final_df['smiles'] != 'Failed']
        if len(list(final_df['smiles']))>0:
            break
        elif RETRY ==0:
            FAILED = True
            break
        else:
            RETRY = RETRY - 1
            continue
    if FAILED:
        final_df["smiles"] = "FAILED"
    final_df['clean_scaffold'] = final_df.apply(lambda x:  _cleanup_decoration(x['scaffold']), axis=1)
    return final_df[["smiles", "clean_scaffold", "decoration_smi"]]


