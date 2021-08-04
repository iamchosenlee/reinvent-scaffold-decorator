import time
import prop_predict
from chemprop.utils import load_checkpoint, load_scalers
start = time.time()

def prop_model_load(chem_checkpoint):
    chem_model = load_checkpoint(chem_checkpoint)
    scaler, features_scaler = load_scalers(chem_checkpoint)
    return chem_model, scaler

chem_model, scaler = prop_model_load('/home/sumin/Retro/reinvent-scaffold-decorator/Drug_Generation/model/model.pt')

result = prop_predict.get_result(['O=C(C#CC1CCNCC1)N1CCN(c2cccc(C(F)(F)F)c2)CC1', 
            'NC(=O)Nc1cc(NC(=O)[C@@H](F)CN(Cc2ccccc2)Cc2ccccc2)ccc1F', 
            'O=c1oc2ccccc2n1C1CCN(Cc2ccccc2N2CCN(Cc3ccccc3)CC2)CC1', 
            'N#CCc1ccc(OCc2ccccc2NC(=O)/C=C\c2ccc(O)c(O)c2)cc1',
            'CCN(CCOc1ccccc1C)[C@@H]1CCc2cc(Br)ccc2NC1=O',
            'Oc1c(F)cccc1Nc1ccc(Oc2ncccn2)cc1',
            'CCNCCNC(=O)c1c(C)c(/C=C2\C(=O)Nc3ccc(F)cc32)n(-c2cc(Cl)ccn2)c1C',
            'CCCN1CCCC[C@H]1C(=O)NCCCc1nc2ccccc2[nH]1',
            'NS(=O)(=O)c1cc(F)cc(NC(=O)N[C@@H]2CCC[C@@H]3CN(Cc4ccccc4)C[C@H]32)c1',
            'Cc1cc(-c2ncccc2C2(C#N)CCN(Cc3ccccc3)CC2)cnc1C#N',
            'c1cnc2c(c1)c1cnc(Nc3ccc(N4CCNCC4)cn3)nc1n2C1CCCC1',
            'CN(Cc1cccc(OC(F)(F)F)c1)Cc1cc(C#N)ccn1',
            'O=c1oc2ccccc2n1C1CCN(Cc2ccccc2N2CCN(Cc3ccccc3)CC2)CC1',
            'CCN(CCOc1ccccc1C)[C@@H]1CCc2cc(Br)ccc2NC1=O',
            'O=C(Nc1cc(Cl)ccc1O)Nc1ccccc1Br',
            'O=C(Nc1cc(C(F)F)ccc1F)NC1(CCOCc2ccccc2)CC1',
            'CC(C)(C(=O)N1CCN(c2cccc(C(F)(F)F)c2)CC1)c1cnc[nH]1',
            'C#CCN(Cc1ccccc1)[C@H]1CCc2cc(Br)ccc2NC1=O',
            'Cc1noc(C2CCN(C[C@@H](O)[C@@H](c3ccc(F)cc3F)n3cncn3)CC2)n1',
            'CNC(=O)Nc1cc(NC(=O)[C@@H](F)CN(Cc2ccccc2)Cc2ccccc2)ccc1F',
            'Cc1noc(C2CCN(C[C@@H](O)[C@@H](c3ccc(F)cc3F)n3cncn3)CC2)n1',
            'NC(=S)c1cc2c(nc1N[C@@]13COC[C@@H]1C3)CCCC2',
            'NC(=O)Nc1cc(NC(=O)[C@@H](F)CN(Cc2ccccc2)Cc2ccccc2)ccc1F',
            'O=c1[nH]c2ccccc2n1C1CCN(Cc2cccc(OC(F)F)c2)CC1',
            'Cc1[nH]c(/C=C2\C(=O)Nc3ccc(F)cc32)c(C)c1C(=O)Nc1c[nH]ccc1=O',
            'Cc1ncn(CCCCc2noc([C@@H]3C[C@H]3c3c(F)cccc3F)n2)c1C',
            'Cc1cn(-c2ccc(F)cc2)nc1C(=O)OCc1nc2ccc(Cl)nc2[nH]1',
            'Nc1cnncc1Nc1ccc2[nH]nc(N)c2c1',
            'C[C@@H](Oc1ccc2c(c1)CCC2)C(=O)N1CCC(n2c(=O)[nH]c3ccccc32)CC1',
            'C[C@@H]1CCC[C@](CNC(=O)[C@@H](F)CN(Cc2ccccc2)Cc2ccccc2)(N(C)C)C1'], chem_model, scaler
             )



print(result)
print("time :", time.time() - start)
