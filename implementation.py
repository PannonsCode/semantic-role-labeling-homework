import json
import random

import numpy as np
from typing import List, Tuple

from model import Model

import json
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def build_model_34(language: str, device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(language, device)


def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(language, device)



def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(language, device)


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=True):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    
    def __init__(self, language: str, device: str):

        # load the specific model for the input language
        self.language = language
        self.device = device
        self.tokenizer = None
        self.predicate_classes = None
        self.argument_classes = None
        self.modelPredIdent = None
        self.modelPredDisamb = None
        self.modelArgIdent = None
        self.modelArgClass = None

    def predict(self, sentence):

        result = {}
        file_ext = self.language+".pt"

        #Load dictionaries with classes of predicates and arguments
        if self.predicate_classes is None:
            with open("model/predicateclasses.json", "r") as input_file:
                self.predicate_classes = json.load(input_file);

        if self.argument_classes is None:
            with open("model/argumentclasses.json", "r") as input_file:   
                self.argument_classes = json.load(input_file);

        #Load the tokanizer based on Bert
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        #Load pre-trained model for predicate identification
        if self.modelPredIdent is None:
            pth_model = "model/modelPredIdent"+file_ext
            self.modelPredIdent = torch.load(pth_model, map_location=self.device)
            self.modelPredIdent.eval()

        #Load pre-trained model for predicate disambiguation
        if self.modelPredDisamb is None:
            pth_model = "model/modelPredDisamb2"+file_ext
            self.modelPredDisamb = torch.load(pth_model, map_location=self.device)
            self.modelPredDisamb.eval()

        #Load pre-trained model for argument identification
        if self.modelArgIdent is None:
            pth_model = "model/modelArgIdent"+file_ext
            self.modelArgIdent = torch.load(pth_model, map_location=self.device)
            self.modelArgIdent.eval()

        #Load pre-trained model for argument classification
        if self.modelArgClass is None:
            pth_model = "model/modelArgClass"+file_ext
            self.modelArgClass = torch.load(pth_model, map_location=self.device)
            self.modelArgClass.eval()

        #convert tokens into ids with the tokanizer and associate an attention mask
        words = sentence["words"]
        length = len(words)
        tokens_id = []
        mask = []
        for w in words:
            tokens_id.append(self.tokenizer(w)[0].ids[1])
            mask.append(1)

        '''Condition to choose if executue tasks [1+2+3+4] or [2+3+4] or [3+4]'''

        #TASK [1+2+3+4]
        if "predicates" not in sentence.keys():

            #create a dictionary with input
            #input_ids: tensor with tokes of each word
            #attention_mask: tensor with all '1' (there is no padding)
            my_input = {"input_ids": torch.unsqueeze(torch.tensor(tokens_id), 0),
                         "attention_mask": torch.unsqueeze(torch.tensor(mask), 0)}

            #predict predicates
            with torch.no_grad():
                predicates = self.modelPredIdent(my_input)
            predicates = torch.round(predicates)

            #update the input dictionary -> insert indications of predicates in a sentence
            #(a tensor with '1' if in that position there is a predicate, '0' otherwise)
            my_input.update({"predicates":torch.unsqueeze(predicates, 0)})

            #predicat classes of identified predicates
            with torch.no_grad():
                predicateDisamb = self.modelPredDisamb(**my_input)
            predicateDisamb = F.log_softmax(predicateDisamb, -1)
            predicateDisamb = torch.argmax(predicateDisamb, -1)

            #insert the predicates with its classes in the result
            new_pred_disamb = []
            for p in predicateDisamb[0][:length]:
                new_pred_disamb.append(self.predicate_classes[str(p.item())]) #convert integer into a class (string)
            result.update({"predicates":new_pred_disamb})

            #create the input for argument identification doubling the sentence if there is more than one predicate
            #the input has a sentence, a mask and a predicate tensor for each predicate)
            new_input = []
            predicates_index = []
            for i in range(length):
                if predicates[i]!=0:
                    new_pred_ref = [0 for _ in range(length)]
                    new_pred_ref[i] = 1 #the new predicates list has only one '1'
                    predicates_index.append(i) #store the index of the predicate in the sentence
                    new_input_dict = {}
                    new_input_dict.update({"input_ids": torch.unsqueeze(torch.tensor(tokens_id), 0),
                                           "attention_mask": torch.unsqueeze(torch.tensor(mask), 0),
                                           "predicates": torch.unsqueeze(torch.tensor(new_pred_ref), 0)})
                    new_input.append(new_input_dict)

            #predict arguments
            argumentCouples = []
            for i in new_input:
                with torch.no_grad():
                    out = self.modelArgIdent(**i)
                out = torch.round(out)
                argumentCouples.append(out)

            #update the input dictionary -> insert indications of arguments in a sentence
            #(a tensor with '1' if in that position there is an argument, '0' otherwise)
            for args,inp in zip(argumentCouples, new_input):
                args = torch.unsqueeze(args, 0)
                inp.update({"roles":args})

            #predict the argumet classes
            new_out = []
            for i in new_input:
                with torch.no_grad():
                    out = self.modelArgClass(**i) 
                out = F.log_softmax(out, -1)
                out = torch.argmax(out, -1)
                new_out.append(out)

            #insert the argument classes for each predicate in the result
            result.update({"roles": {}})
            for o,p in zip(new_out,predicates_index):
                roles_list = []
                for oi in o[0]:
                    roles_list.append(self.argument_classes[str(oi.item())]) #convert integer into a class (string)
                result["roles"].update({str(p):roles_list})

            return result

        else:

            #read list of predicates from the input
            predicates = sentence["predicates"]

            #TASK [2+3+4]
            if all(isinstance(i, int) for i in predicates): #check if input predicates are integers or string

                #build the input
                my_input = {"input_ids": torch.unsqueeze(torch.tensor(tokens_id), 0),
                            "attention_mask": torch.unsqueeze(torch.tensor(mask), 0),
                            "predicates":torch.unsqueeze(torch.tensor(predicates), 0)}

                #predict the classes of predicates in input
                with torch.no_grad():
                    predicateDisamb = self.modelPredDisamb(**my_input)
                predicateDisamb = F.log_softmax(predicateDisamb, -1)
                predicateDisamb = torch.argmax(predicateDisamb, -1)

                #convert integers in classes (strings) and add it to the result
                new_pred_disamb = []
                for p in predicateDisamb[0][:length]:
                    new_pred_disamb.append(self.predicate_classes[str(p.item())])
                result.update({"predicates":new_pred_disamb})

                #build input for argument identification
                new_input = []
                predicates_index = []
                for i in range(length):
                    if predicates[i]!=0:
                        new_pred_ref = [0 for _ in range(length)] #list with all zeros
                        new_pred_ref[i] = 1 #insert 1 in the position in corrispondence of a predicate in the sentence
                        predicates_index.append(i) #store the index of the predicate in a list
                        new_input_dict = {}
                        new_input_dict.update({"input_ids": torch.unsqueeze(torch.tensor(tokens_id), 0),
                                               "attention_mask": torch.unsqueeze(torch.tensor(mask), 0),
                                               "predicates": torch.unsqueeze(torch.tensor(new_pred_ref), 0)})
                        new_input.append(new_input_dict)

                #predict the arguments for each predicate in the same sentence
                argumentCouples = []
                for i in new_input:
                    #predictions
                    with torch.no_grad():
                        out = self.modelArgIdent(**i)
                    out = torch.round(out)
                    argumentCouples.append(out)

                #update the input dictionary -> insert indications of arguments in a sentence
                #(a tensor with '1' if in that position there is an argument, '0' otherwise)
                for args,inp in zip(argumentCouples, new_input):
                    args = torch.unsqueeze(args, 0)
                    inp.update({"roles":args})

                #predictions of each identified argument classes of each predicate
                new_out = []
                for i in new_input:
                    #predictions
                    with torch.no_grad():
                        out = self.modelArgClass(**i) 
                    out = F.log_softmax(out, -1)
                    out = torch.argmax(out, -1)
                    new_out.append(out)

                #insert the argument classes for each predicate in the result
                result.update({"roles": {}})
                for o,p in zip(new_out,predicates_index):
                    roles_list = []
                    for oi in o[0]:
                        roles_list.append(self.argument_classes[str(oi.item())]) #convert integers into classes (string)
                    result["roles"].update({str(p):roles_list})

                return result

            #TASK [3+4]
            else:

                #convert readed predicates in a list of '1' if it a predicate, '0' otherwise
                new_predicates = []
                for p in predicates:
                    if p!="_":
                        new_predicates.append(1)
                    else:
                        new_predicates.append(0)

                #build input for argument identification
                new_input = []
                predicates_index = []
                for i in range(length):
                    if new_predicates[i]!=0:
                        new_pred_ref = [0 for _ in range(length)]
                        new_pred_ref[i] = 1
                        predicates_index.append(i)
                        new_input_dict = {}
                        new_input_dict.update({"input_ids": torch.unsqueeze(torch.tensor(tokens_id), 0),
                                               "attention_mask": torch.unsqueeze(torch.tensor(mask), 0),
                                               "predicates": torch.unsqueeze(torch.tensor(new_pred_ref), 0)})
                        new_input.append(new_input_dict)

                #predict the arguments for each predicate in the same sentence
                argumentCouples = []
                for i in new_input:
                    #predictions
                    with torch.no_grad():
                        out = self.modelArgIdent(**i)
                    out = torch.round(out)
                    argumentCouples.append(out)

                #update the input dictionary -> insert indications of arguments in a sentence
                #(a tensor with '1' if in that position there is an argument, '0' otherwise)
                for args,inp in zip(argumentCouples, new_input):
                    args = torch.unsqueeze(args, 0)
                    inp.update({"roles":args})

                #predict the argument classes
                new_out = []
                for i in new_input:
                    #predictions
                    with torch.no_grad():
                        out = self.modelArgClass(**i) 
                    out = F.log_softmax(out, -1)
                    out = torch.argmax(out, -1)
                    new_out.append(out)

                #insert the argument classes for each predicate in the result
                result.update({"roles": {}})
                for o,p in zip(new_out,predicates_index):
                    roles_list = []
                    for oi in o[0]:
                        roles_list.append(self.argument_classes[str(oi.item())]) #convert integers into classes (string)
                    result["roles"].update({str(p):roles_list})

                return result

#class of model for predicate identification
class PredicateIdentifier(torch.nn.Module):

    def __init__(self, n_hidden, classes, language_model_name="bert-base-cased"):
        super(PredicateIdentifier, self).__init__()

        #Transformer
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)

        #Freeze Transformer
        #for param in self.transformer_model.parameters():
          #param.requires_grad = False

        #Number of features in output from Bert
        self.embedding_dim = self.transformer_model.config.hidden_size

        #Linear classifier
        self.hidden1 = torch.nn.Linear(self.embedding_dim, classes)

    def forward(self, sentence):

        out = self.transformer_model(**sentence)
        out = torch.stack(out[-4:-2], dim=0)
        out = torch.sum(out, dim = 0)

        out = self.hidden1(out)
        out = torch.sigmoid(out)

        return out.squeeze()

#class of model for predicate disambiguation
class PredicateDisambiguation(torch.nn.Module):

    def __init__(self, embedding_dim, n_hidden, classes, device, language_model_name="bert-base-cased"):
        super(PredicateDisambiguation, self).__init__()

        #Trnsformer
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)

        #Freeze Transformer
        #for param in self.transformer_model.parameters():
          #param.requires_grad = False

        #Number of features in output from the transformer
        self.embedding_dim = self.transformer_model.config.hidden_size

        #Additive feature to represent where is a predicate
        self.add_features = 1

        #BiLSTM layer
        self.lstm = nn.LSTM(self.embedding_dim+self.add_features, n_hidden, bidirectional = True, num_layers = 2, dropout=0.3, batch_first = True)

        #Linear classifier
        self.hidden1 = torch.nn.Linear(2*n_hidden, classes)

    def forward(self, 
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                predicates: torch.Tensor = None):
      
        dict_input = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }

        out = self.transformer_model(**dict_input)
        out = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)
        
        #associate a tensor of '1' in correspondence with the predicates or a '0' otherwise
        o = []
        for oi,pid in zip(out, predicates):
          new_pred_id = []
          for p in pid:
            p = p.item()
            if p!=0 and p!=-100:
              new_pred_id.append(torch.ones((self.add_features),requires_grad=True))
            else:
              new_pred_id.append(torch.zeros((self.add_features),requires_grad=True))
          new_pred_id = torch.stack(new_pred_id)
          o.append(torch.cat((oi,new_pred_id),dim=1))
        
        out = torch.stack(o)
        out, _ = self.lstm(out)
        out = self.hidden1(out)

        return out

#class of model for argument identification
class ArgumentIdentifier(torch.nn.Module):

    def __init__(self, n_hidden, classes, device, language_model_name="bert-base-cased"):
        super(ArgumentIdentifier, self).__init__()

        #Transformer
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)

        #Freeze transformer
        for param in self.transformer_model.parameters():
          param.requires_grad = False

        #Number of features in output from the transformer
        self.embedding_dim = self.transformer_model.config.hidden_size

        #Additive feature to represent where is a predicate
        self.add_features = 1

        #BiLSTM layer
        self.lstm = nn.LSTM(self.embedding_dim+self.add_features, n_hidden, bidirectional = True, num_layers = 2, dropout=0.3, batch_first = True)

        #Linear classifier
        self.hidden1 = torch.nn.Linear(2*n_hidden, classes)

    def forward(self, 
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                predicates: torch.Tensor = None):
      
        dict_input = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }

        out = self.transformer_model(**dict_input)

        out = torch.stack(out[-4:-2], dim=0)
        out = torch.sum(out, dim = 0)

        #associate a tensor of '1' in correspondence with the predicates or a '0' otherwise
        o = []
        for oi,pid in zip(out, predicates):
          new_pred_id = []
          for p in pid:
            p = p.item()
            if p!=0:
              new_pred_id.append(torch.ones(self.add_features))
            else:
              new_pred_id.append(torch.zeros(self.add_features))
          new_pred_id = torch.stack(new_pred_id)
          o.append(torch.cat((oi,new_pred_id),dim=1))
        out = torch.stack(o)

        out, _ = self.lstm(out)
        out = self.hidden1(out)
        out = torch.sigmoid(out)

        return out.squeeze()

#class of model for argument classification
class ArgumentClassification(torch.nn.Module):

    def __init__(self, embedding_dim, n_hidden, classes, device, language_model_name="bert-base-cased"):
        super(ArgumentClassification, self).__init__()

        #Transformer model
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)
        #Freeze transformer
        for param in self.transformer_model.parameters():
          param.requires_grad = False

        #Number of features in output from Bert
        self.embedding_dim = self.transformer_model.config.hidden_size

        #Additive feature to represent where is a predicate
        self.add_features = 2

        #BiLSTM layer
        self.lstm = nn.LSTM(self.embedding_dim+self.add_features, n_hidden, bidirectional = True, num_layers = 2, dropout=0.3, batch_first = True)

        #Linear classifier
        self.hidden1 = torch.nn.Linear(2*n_hidden, classes)

    def forward(self, 
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                roles: torch.Tensor = None,
                predicates: torch.Tensor = None):
      
        dict_input = {
          "input_ids": input_ids, 
          "attention_mask": attention_mask
        }

        out = self.transformer_model(**dict_input)
        out = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)

        '''
        Associate a tensor of shape (2):
        -First element: '1' in correspondence with the arguments and '0' otherwise
        -Second element: mean(features(token[i])) - mean(features(predicate)) for all i
        '''
        o = []
        for oi,rol,pred in zip(out, roles, predicates):
          index_pred = torch.argmax(pred).item()
          mean_token_pred = torch.mean(oi[index_pred,:])
          new_tensor = []
          for oii,r in zip(oi,rol):
            mean_token_word = torch.mean(oii)
            new_mean = (mean_token_word-mean_token_pred).item()
            r = r.item()
            if r!=0 and r!=-100:
              new_tensor.append(torch.tensor((new_mean,1), dtype=torch.float32))
            else:
              new_tensor.append(torch.tensor((new_mean,0), dtype=torch.float32))
          new_tensor = torch.stack(new_tensor)
          o.append(torch.cat((oi,new_tensor),dim=1))
        out = torch.stack(o)
        
        out, _ = self.lstm(out)
        out = self.hidden1(out)

        return out
