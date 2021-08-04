import torch
import torch.utils.data as tud
import argparse
import utils.chem as uc
import utils.scaffold as usc
import models.model as mm
import models.actions as ma
import models.dataset as mda
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import seaborn as sns
from join import fast_join
from tqdm import tqdm, trange
import pdb
import pickle
import csv
import json
import random
random.seed(42)

from chemprop.train.predict import predict
from chemprop.data import MoleculeDataLoader, get_data_from_smiles
from chemprop.utils import load_checkpoint, load_scalers
from chemprop.data import StandardScaler
from typing import Tuple
from Drug_Generation import prop_predict

PREDICTOR_MODEL_PATH = './Drug_Generation/model/model.pt'
SAVE_MODEL_PATH = "./jak2_decorator/rl_models/model.optimized3.50"


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data :class:`~chemprop.data.scaler.StandardScaler`
             and features :class:`~chemprop.data.scaler.StandardScaler`.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler

def prop_model_load(chem_checkpoint):
    chem_model = load_checkpoint(chem_checkpoint)
    scaler, features_scaler = load_scalers(chem_checkpoint)
    #scaler = load_scalers(chem_checkpoint)
    return chem_model, scaler
def prop_model_load(chem_checkpoint):
    chem_model = load_checkpoint(chem_checkpoint)
    scaler, features_scaler = load_scalers(chem_checkpoint)
    return chem_model, scaler


class Reinforcement(object):
    def __init__(self, generator, perdictor, get_reward):

        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

    def policy_gradient(self, data, n_batch=10, gamma=0.97, grad_clipping=None, **kwargs):

        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0

        for _ in range(n_batch):
            rewards = []
            while len(rewards) == 0:
                try:
                    trajectory, scaffold, decoration = self.generator.evaluate(next(data))
                    rewards = self.get_reward(trajectory, self.predictor) #need to remove mean -done
                except:
                    rewards=[]

            discounted_rewards = rewards
            total_reward += np.sum(rewards)

            scaffold_decoration_smi_list = list(map(list, zip(scaffold, decoration)))

            ## follow the trajectory and accumulate the loss
            rl_loss, idx = self.generator.compute_loss(scaffold_decoration_smi_list, discounted_rewards, gamma)
            n_batch += (idx-1)

        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        self.generator.model.set_mode("train")
        rl_loss.backward()
    
#         if grad_clipping is not None:
#                 torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
#                                                grad_clipping)

        self.generator.optimizer.step()
        
        return total_reward, rl_loss.item()

class Generator(object):
    def __init__(self, params):
        # self.model = mm.DecoratorModel.load_from_file(params.input_model_path, mode="eval")
        self.model = mm.DecoratorModel.load_from_file("./jak2_decorator/models/model.trained.50")
        self.optimizer = torch.optim.Adam(self.model.network.parameters(), lr=1E-4)
        self.sample_model_action = ma.SampleModel(self.model, batch_size=16)
        self.nll_loss = torch.nn.NLLLoss()

        
    def evaluate(self, input_smiles):
        
        samples_df = fast_join([input_smiles], self.sample_model_action) #id,smiles,nll,scaffold,decoration_smi
        trajectory = samples_df["smiles"]
        scaffold = samples_df["clean_scaffold"]
        decoration= samples_df["decoration_smi"]
        return list(trajectory), list(scaffold), list(decoration)
    
    
    def parameters(self):
        return self.model.get_params()
    
    def compute_loss(self, scaffold_decoration_smi_list, discounted_rewards, gamma):
        #output = []
        dataloader = self.initialize_dataloader(scaffold_decoration_smi_list)
        seq_lengths = torch.ones(1)
        nlls = torch.zeros(1).cuda()
        idx=0
        for s,d in dataloader:
            discounted_reward = discounted_rewards[idx]            

            encoder_padded_seqs, hidden_states = generator.model.network.forward_encoder(*s)
            encoder_padded_seqs = encoder_padded_seqs.unsqueeze(dim=0)

            (hs_h, hs_c) = hidden_states
            hs_h = hs_h.unsqueeze(dim=1)
            hs_c = hs_c.unsqueeze(dim=1)

            for i in range(d[1].item()-1): #decoration_seq_lengths
                logits, _, attention_weights = generator.model.network.forward_decoder(
                        d[0][:,i].unsqueeze(dim=0), seq_lengths, encoder_padded_seqs, (hs_h, hs_c))
                nlls += self.nll_loss(logits.log_softmax(dim=2).reshape((1,-1)), d[0][:,i+1])*discounted_reward
                discounted_reward = discounted_reward * gamma
            
            idx+=1

        return nlls, idx
            
        
    def initialize_dataloader(self, training_set):
        dataset = mda.DecoratorDataset(training_set, vocabulary=self.model.vocabulary)
        return tud.DataLoader(dataset, batch_size=1, shuffle=False,
                              collate_fn=mda.DecoratorDataset.collate_fn, drop_last=False)

    
class CustomPredictor(object):
    def __init__(self, negative_sign = True, 
            model_dir='/home/sumin/Retro/reinvent-scaffold-decorator/Drug_Generation/model/model.pt',
            multi_objective=None):
        self.model_dir = model_dir
        self.chem_model, self.scaler = prop_model_load(self.model_dir)
        self.negative_sign = negative_sign
        self.multi_objective = multi_objective #float
        self.processed_smiles, self.invalid_smiles = [], []


    def check_smiles(self, input_smiles):
        processed_smiles = []
        invalid_smiles = []
        for sm in input_smiles:
            try:
                mol = Chem.MolFromSmiles(sm, sanitize=True)
                processed_smiles.append(sm)
            except:
                invalid_smiles.append(sm)
                continue
        return processed_smiles, invalid_smiles

    def predict(self, objects=None, **kwargs):
        self.processed_smiles, self.invalid_smiles = self.check_smiles(objects)
        prediction = np.array(prop_predict.get_result(self.processed_smiles, self.chem_model, self.scaler))
        if self.negative_sign:
            prediction = prediction * (-1)
#         if self.multi_objective:
#             mols = [Chem.MolFromSmiles(sm, sanitize=True) for sm in self.processed_smiles]
#             contraint_score = np.array([QED.qed(mol) for mol in mols])
# #            print("QED: ", contraint_score)
# #            print("BA: ", prediction)
#             multi_objective_score = contraint_score * self.multi_objective + prediction * (1 - self.multi_objective)
# #            print("mult_score: ", prediction)
#             return self.processed_smiles, prediction, multi_objective_score, self.invalid_smiles
        return self.processed_smiles, prediction, self.invalid_smiles


def plot_hist(prediction, n_to_generate, label="ba"):
    print("Mean value of predictions:", prediction.mean())
    ax = sns.kdeplot(prediction, shade=True)
    ax.set(xlabel=f'Predicted {label}', 
           title=f'Distribution of predicted {label} for generated molecules')
    plt.show()

def estimate_and_update(generator, predictor, gen_data, n_to_generate, **kwargs):
    generated = []
    buffer = []
    
    with tqdm(total=n_to_generate) as pbar:
        while True:
            pbar.set_description("Generating molecules...")
            try:
                trajectory = generator.evaluate(next(gen_data))[0][0]
                mol = Chem.MolFromSmiles(trajectory, sanitize=False)
                smiles = Chem.MolToSmiles(mol)
                qed = QED.qed(mol)
                generated.append(smiles)
                pbar.update(1)
            except:
                continue
            if len(list(np.unique(generated))[1:]) == n_to_generate:
                break

    unique_smiles = list(np.unique(generated))[1:]
#     if predictor.multi_objective:
#         smiles, prediction, multi_objective_score, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)
#         plot_hist(multi_objective_score, n_to_generate, label="multi_objective_score")
#         plot_hist(prediction, n_to_generate)
#         return smiles, prediction, multi_objective_score
#     else :
#         smiles, prediction, nan_smiles = predictor.predict(unique_smiles, get_features=get_fp)  
#     plot_hist(prediction, n_to_generate)
    smiles, prediction, nan_smiles = predictor.predict(unique_smiles)
    plot_hist(prediction, n_to_generate)
        
    return smiles, prediction


def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def get_reward_max(smiles, predictor, invalid_reward=0.0, get_features=None):
    if predictor.multi_objective:
        mol, original_prop, prop, nan_smiles = predictor.predict(smiles, get_features=get_features)
    else:
        mol, prop, nan_smiles = predictor.predict(smiles, get_features=get_features)
    if (len(nan_smiles) == 1):
        return invalid_reward
    return np.exp(prop/2)

def parse_args():
    parser = argparse.ArgumentParser(
                description="Train model with RL.")
    parser.add_argument("--input-model-path", "-i", help="Input model file", type=str, default ="jak2_decorator/models/model.trained.50" )
    parser.add_argument("--output-model-prefix-path", "-o",
                            help="Prefix to the output model (may have the epoch appended).", type=str, default ="jak2_decorator/rl_models/model.trained.50")
#     parser.add_argument("--training-set-path", "-s", help="Path to a file with (scaffold, decoration) tuples \
#             or a directory with many of these files to be used as training set.", type=str, required=True)
        # parser.add_argument("--batch_size")

    return parser.parse_args(args=[])

def getData(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                yield from choose_scaffold(row)
            except StopIteration as e:
                print(e)
            except:
                continue


with open("conditions.json.example", "r") as json_file:
    data = json.load(json_file)
    if "scaffold" in data:
        scaffold_conditions = data["scaffold"]
    if "decoration" in data:
        decoration_conditions = data["decoration"]
enumerator = usc.SliceEnumerator(usc.SLICE_SMARTS["recap"],scaffold_conditions, decoration_conditions)

def choose_scaffold(row, max_cuts=4, enumerator=enumerator):
            smiles = row[0]
            mol = uc.to_mol(smiles)
            out_rows = []
            if mol:
                for cuts in range(1, max_cuts + 1):
                    for sliced_mol in enumerator.enumerate(mol, cuts=cuts):
                        scaff_smi, dec_smis = sliced_mol.to_smiles()
                        out_rows.append(scaff_smi)
                        
            return out_rows


if __name__ =='__main__':
    matplotlib.use('Agg')
    params = parse_args()
    generator = Generator(params)
    predictor = CustomPredictor(negative_sign=True, model_dir=PREDICTOR_MODEL_PATH)
    get_reward = get_reward_max
    RL_max = Reinforcement(generator, predictor, get_reward)
    gen_data = getData("./training_sets/jak2_smiles.csv")
    
    generated_smiles, generated_scores = estimate_and_update(generator, predictor, gen_data, 1000)
    print(generated_smiles[:10], generated_scores[:10])
    
    rewards_max = []
    rl_losses_max = []
    n_iterations = 100
    n_policy = 15
    # n_iterations = 10
    # n_policy = 5
    for i in range(n_iterations):
        print("#################################################################################################")
        for j in trange(n_policy, desc='Policy gradient...'):
            cur_reward, cur_loss = RL_max.policy_gradient(gen_data, n_batch=5)
            print(f"{j}th cur_reward, cur_loss: ", cur_reward, cur_loss)
            rewards_max.append(simple_moving_average(rewards_max, cur_reward)) 
            rl_losses_max.append(simple_moving_average(rl_losses_max, cur_loss))
        plt.figure(1)
        plt.subplot(211)
        plt.plot(rewards_max)
        plt.xlabel(f'Training iteration {j}-{i}')
        plt.ylabel('Average reward')
        plt.subplot(212)
        plt.plot(rl_losses_max)
        plt.xlabel(f'Training iteration {j}-{i}')
        plt.ylabel('Loss')
        plt.show()
        plt.savefig('Average_reward_and_Loss.png')

        smiles_cur, prediction_cur = estimate_and_update(RL_max.generator, 
                                                         predictor, gen_data, 10)
        print(f'{i}th Sample trajectories:')
        for sm in smiles_cur[:5]:
            print(sm)
    
    gen_data2 = getData("./training_sets/jak2_smiles.csv")
    optimized_smiles, optimized_scores = estimate_and_update(RL_max.generator, predictor, gen_data2, 1000)
    print(optimized_smiles[:10], optimized_scores[:10])
    
    save_dict = {'generated_smiles':generated_smiles, 'generated_scores':generated_scores, 'optimized_smiles':optimized_smiles,'optimized_scores': optimized_scores,
            'avg_reward': rewards_max, 'rl_losses': rl_losses_max}
    
    with open("save_dict.pkl", "wb") as f:
        pickle.dump(save_dict,f)
    pdb.set_trace()
    sns.kdeplot(optimized_scores,label='Biased', shade=True, color='red')
    sns.kdeplot(generated_scores, label='Unbiased', shade=True, color='grey')
    plt.xlabel('values')
    plt.title("Distribution of BA")
    plt.show()
    plt.savefig('Distribution.png')
    
    RL_max.generator.model.save(SAVE_MODEL_PATH)
        
#     trajectory, scaffold, decoration = generator.evaluate(next(gen_data))
#     scaffold_decoration_smi_list = list(map(list, zip(scaffold, decoration)))
#     print(scaffold_decoration_smi_list)
#     reward = get_reward(trajectory, predictor)
#     print("reward ",reward)
#     output = generator.compute_loss(scaffold_decoration_smi_list, reward, 0.97)
#     print("output ", output)
