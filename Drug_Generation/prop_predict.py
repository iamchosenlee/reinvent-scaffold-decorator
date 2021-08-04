import numpy as np
from chemprop.train.predict import predict
from chemprop.data import MoleculeDataLoader, get_data_from_smiles
from chemprop.utils import load_checkpoint, load_scalers

def prediction(smiles_list, model, scaler):
    
    full_data = get_data_from_smiles(
        smiles=[[smiles] for smiles in smiles_list],
        skip_invalid_smiles=False,
        features_generator=None
    )
 
    test_data_loader = MoleculeDataLoader(
        dataset = full_data,
        batch_size= len(full_data),
        num_workers=8
    )

    model_preds = predict(model=model, data_loader=test_data_loader, scaler=scaler)
    for i in range(len(model_preds)):
        model_preds[i] = model_preds[i][0]

    return model_preds

def get_result(smiles_list, model, scaler):
    preds = prediction(smiles_list, model, scaler)
    return preds
