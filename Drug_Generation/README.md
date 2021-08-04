
# Drug_Generation

## For prediction 
- Example: test.py
- Input: SMILES list, checkpoint_path
- Output: predicted value list (example: [-7.886887404710157, -7.244531815293778, -7.366552358778106, -7.600420765025972])
```
import prop_predict
result = prop_predict.get_result(['O=C(C#CC1CCNCC1)N1CCN(c2cccc(C(F)(F)F)c2)CC1', 
            'NC(=O)Nc1cc(NC(=O)[C@@H](F)CN(Cc2ccccc2)Cc2ccccc2)ccc1F', 
            'O=c1oc2ccccc2n1C1CCN(Cc2ccccc2N2CCN(Cc3ccccc3)CC2)CC1', 
            'N#CCc1ccc(OCc2ccccc2NC(=O)/C=C\c2ccc(O)c(O)c2)cc1'], 
            '/home/sejeong/codes/COND/prediction/BA_model/fold_0/model_0/model.pt')
print(result)
```
<pre>


</pre>
## For chemprop (Only sejeong have to do this!) 
https://github.com/chemprop/chemprop

### How to install chemprop
```
pip install git+https://github.com/bp-kelley/descriptastorus
pip install chemprop
```
### For training
```
chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>
```
### For hyperparameter optimization 
```
chemprop_hyperopt --data_path <data_path> --dataset_type <type> --num_iters <n> --config_save_path <config_path>
chemprop_train --data_path <data_path> --dataset_type <type> --config_path <config_path>
```

