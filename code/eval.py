import os


import torch
from torch.utils import data


from ner_models import NERModel, NERModel_CRF
from plot_embeddings import visualize_model_embeddings
from preprocess_dataset import PreProcessor, create_dataset
from train_utils import opts, HParams
from utils import collate_labelled_batch, compute_precision, plot_conf


print("\nEvaluating {} model saved in {}\n".format(opts["architecture"], opts["save_model_path"]))

device = opts["device"]

p = PreProcessor("train")

# Load tokenizers for input, pos, and labels. If they don't exist, create them
if os.path.exists(p.word_to_int_path):
    (vocabulary, decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.load_all_tokenizers()
else:
    p.read_data("./data")
    p.create_pos_data(test_data=None)
    (vocabulary, decode), (pos_vocabulary, int_to_pos), (label_vocabulary, int_to_label) = p.create_all_tokenizers()

# Create dev and test datasets
dev_dataset, _, encoded_dev_labels = create_dataset(dataset_type="dev", opts=opts, test_data=None)
dev_dataloader = data.DataLoader(dev_dataset, collate_fn=collate_labelled_batch, batch_size=256)

test_dataset, _, encoded_test_labels = create_dataset(dataset_type="test", opts=opts, test_data=None)
test_dataloader = data.DataLoader(test_dataset, collate_fn=collate_labelled_batch, batch_size=256)


print("Number of Epochs", opts["epochs"])

print("Loading trained model...")

# Define Hyperparameters
load_params = HParams(vocabulary, pos_vocabulary, label_vocabulary, opts)


with torch.no_grad():
    # Instatiate appropriate model
    if opts['use_crf'] == True:

        print("\n\nBiLSTM-CRF")
        ner_model = NERModel_CRF(load_params).cuda()

    elif opts['use_crf'] == False:

        print("\n\nBaseline BiLSTM")
        ner_model = NERModel(load_params).cuda()
    
    # Load model in eval mode
    state_dict = torch.load(os.path.join(opts["save_model_path"], 'state_{}.pth'.format(opts["epochs"]-1)), map_location=device)
    ner_model.load_state_dict(state_dict)    
    ner_model.to(device)
    ner_model.eval()

    print("Model loaded and set to EVAL mode") 

    # Plot word embeddings (and POS embeddings if model was trained with both)
    word_embeds = ner_model.word_embedding.weight
    if opts["use_pos_embeddings"] == True:
        pos_embeds = ner_model.word_embedding.weight
        visualize_model_embeddings("pos", pos_embeds, pos_vocabulary, opts)

    visualize_model_embeddings("word", word_embeds, vocabulary, opts)

    # Evaluate model by computing precision, confusion matrix and recall & F-score on dev or test set
    
    # precisions = compute_precision(ner_model, dev_dataloader, label_vocabulary, encoded_dev_labels, opts)
    p1 = PreProcessor("test")
    p1.read_data("./data")
    p1.create_pos_data(test_data=None)
    precisions = compute_precision(ner_model, test_dataloader, label_vocabulary, encoded_test_labels, opts, int_to_label, p1)

    per_class_precision = precisions["per_class_precision"]
    print("Micro Precision: {}\nMacro Precision: {}".format(precisions["micro_precision"], precisions["macro_precision"]))
    print("Rcall: {}\n\nF-1 score: {}\n\n".format(precisions["recall"], precisions["f1"]))
    print("Per class Precision:")
    for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
        label = int_to_label[(idx_class)] if idx_class != 0 else int_to_label[(idx_class)]
        print(label, precision)

    # Print, plot, and save the confusion matrix

    print("Confusion matrix\n", precisions["confusion_matrix"])

    file_name = "DevSet_Confusion_Matrix"
    plot_conf(file_name, precisions["confusion_matrix"], opts["save_model_path"])
