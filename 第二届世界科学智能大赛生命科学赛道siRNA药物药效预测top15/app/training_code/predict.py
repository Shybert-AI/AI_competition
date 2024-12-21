import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from baselinev1 import GenomicTokenizer,GenomicVocab,SiRNAModel,SiRNADataset,calculate_gc
import pickle
from glob import iglob
def predict_model(model, test_loader, device='cuda'):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    y_pred = np.array(predictions)
    return y_pred


if __name__ == "__main__":
    # Load data
    #test_data = pd.read_csv('sample_submission.csv')
    test_data = pd.read_csv('/tcdata/private_test_0808.csv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def combine_features(row):
        return ' '.join([str(row['cell_line_donor']), str(row['siRNA_concentration']), str(row['Transfection_method']), str(row['modified_siRNA_sense_seq']),str(row['modified_siRNA_antisense_seq'])])

    test_data['combined_features'] = test_data.apply(combine_features, axis=1)
    test_data['combined_features'] = test_data.apply(combine_features, axis=1)
    test_data['siRNA_sense_seq_gc'] = test_data["siRNA_sense_seq"].apply(calculate_gc)
    test_data['siRNA_antisense_seq_gc'] = test_data["siRNA_antisense_seq"].apply(calculate_gc)
    #columns = ['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list']
    columns = ['publication_id','gene_target_symbol_name','gene_target_ncbi_id','gene_target_species','siRNA_duplex_id','siRNA_sense_seq', 'siRNA_antisense_seq', 'cell_line_donor',
               'Transfection_method', 'modified_siRNA_sense_seq','modified_siRNA_antisense_seq',
               'modified_siRNA_sense_seq_list']

    # columns = ['publication_id','gene_target_symbol_name','gene_target_ncbi_id','gene_target_species','siRNA_duplex_id','siRNA_sense_seq',
    #            'siRNA_antisense_seq', 'cell_line_donor',"siRNA_concentration",'Transfection_method',"Duration_after_transfection_h",
    #            'modified_siRNA_sense_seq','modified_siRNA_antisense_seq','modified_siRNA_sense_seq_list',"modified_siRNA_antisense_seq_list"]
    # columns = ['gene_target_ncbi_id', 'gene_target_species',
    #            'siRNA_duplex_id', 'siRNA_sense_seq', 'siRNA_antisense_seq', 'cell_line_donor', "siRNA_concentration",
    #            'Transfection_method', "Duration_after_transfection_h", 'modified_siRNA_sense_seq',
    #            'modified_siRNA_antisense_seq',
    #            'modified_siRNA_sense_seq_list', "modified_siRNA_antisense_seq_list","combined_features"]

    # 81.74
    columns =  [
        #'id',
        #'publication_id',
        'gene_target_symbol_name',
        'gene_target_ncbi_id',
        'gene_target_species',
        'siRNA_duplex_id',
        'siRNA_sense_seq',
        'siRNA_antisense_seq',
        'cell_line_donor',
        'siRNA_concentration',
        #'concentration_unit',
        'Transfection_method',
        'Duration_after_transfection_h',
        'modified_siRNA_sense_seq',
        'modified_siRNA_antisense_seq',
        'modified_siRNA_sense_seq_list',
        'modified_siRNA_antisense_seq_list',
        "siRNA_sense_seq_gc",
        "siRNA_antisense_seq_gc"
        #'gene_target_seq',
        #'mRNA_remaining_pct'
    ]
    # Create vocabulary
    tokenizer = GenomicTokenizer(ngram=3, stride=3)

    with open("training_code/train_all_tokens.pkl", "rb") as f:  # 注意使用 "rb" 模式
        all_tokens = pickle.load(f)


    vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)
    model = SiRNAModel(len(vocab.itos))


    # Find max sequence length
    max_len = max(max(len(seq.split()) if ' ' in str(seq) else len(tokenizer.tokenize(str(seq)))
                      for seq in test_data[col]) for col in columns)
    max_len = 25
    # Create datasets
    test_dataset = SiRNADataset(test_data, columns, vocab, tokenizer, max_len,model_sign="test")
    test_dataset = DataLoader(test_dataset, batch_size=32)

    # 多模型融合
    model_list = [i for i in sorted(list(iglob("training_code/*.pt"))) if eval(i.split("_")[-1].split(".")[0]) > 110]
    predict_result = None
    for model_name in model_list:
        state_dict = torch.load(model_name, map_location=device)
        model.load_state_dict(state_dict)
        # model.eval()
        y = predict_model(model.to(device), test_dataset)
        print(y)

        if predict_result is None:
            predict_result = y
        else:
            predict_result += y

    predict_result = predict_result / len(model_list)
    test_data["mRNA_remaining_pct"] = predict_result

    #test_data["mRNA_remaining_pct"] = (y+y2)/2
    #test_data["mRNA_remaining_pct"] = y2
    #test_data.to_csv('submission.csv', index=None)
    test_data.to_csv("/app/submit.csv", index=None)
