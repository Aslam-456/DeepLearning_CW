# models.py

import numpy as np
import collections
import textwrap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F


#####################
# MODELS FOR PART 1 #
#####################

device = "cuda" if torch.cuda.is_available() else "cpu"


class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

#####################
# RNN Classifier #
#####################

class RNNClassifier(ConsonantVowelClassifier, nn.Module):  # Ensure correct inheritance
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, vocab_index, num_layers=1):
        super(RNNClassifier, self).__init__()  # Initialize both parent classes
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.vocab_index = vocab_index  # Store vocab_index as an attribute

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output  # Return raw logits without softmax

    def predict(self, context):
        input_tensor = string_to_tensor(context, self.vocab_index).unsqueeze(0)  # Access vocab_index from the class
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class


# def string_to_tensor(s, vocab_index):
#     """
#     Converts a raw string to a PyTorch tensor of indices based on the vocab_index.
#     """
#     indices = [vocab_index.index_of(char) for char in s]  # Use index_of to get index of each character
#     return torch.tensor(indices, dtype=torch.long)

def string_to_tensor(s, vocab_index, max_length=20):
    """
    Converts a raw string to a PyTorch tensor of indices based on the vocab_index,
    truncating or padding to max_length.
    """
    s = s[:max_length]  # Truncate to max_length if necessary
    indices = [vocab_index.index_of(char) for char in s]
    if len(indices) < max_length:
        # Pad with a padding index (assuming 0 is the padding index)
        indices += [0] * (max_length - len(indices))
    return torch.tensor(indices, dtype=torch.long)

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    Trains an RNNClassifier on the provided training data.

    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    # Hyperparameters from args
    vocab_size = len(vocab_index)
    embed_dim = getattr(args, 'embed_dim', 128)  # Default embedding dimension
    hidden_dim = getattr(args, 'hidden_dim', 256)  # Default hidden dimension
    output_dim = 2  # Binary classification (consonant vs. vowel)
    num_layers = getattr(args, 'num_layers', 1)
    learning_rate = getattr(args, 'learning_rate', 0.001)
    num_epochs = getattr(args, 'num_epochs', 20)
    batch_size = getattr(args, 'batch_size', 32)

    # Create model, pass vocab_index during initialization
    model = RNNClassifier(vocab_size, embed_dim, hidden_dim, output_dim, vocab_index, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare training data
    train_data = [(string_to_tensor(s, vocab_index), 0) for s in train_cons_exs] + \
                 [(string_to_tensor(s, vocab_index), 1) for s in train_vowel_exs]
                 

    # Convert training data to DataLoader for batching
    train_tensors = [(s, torch.tensor(label, dtype=torch.long)) for s, label in train_data]
    
    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Unpack the batch and pad sequences
            inputs, labels = zip(*batch)
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
            labels = torch.stack(labels)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Print loss for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")



    return model

#####################
# MODELS FOR PART 2 #
#####################


# def str_to_tensor(word, vocab_index):
#     word = word[:50]
#     indices = [vocab_index.index_of(s) for s in word]
#     return torch.tensor(indices, torch.long)
    


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel,nn.Module):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)
    
    

def textChunker(text, chunksize=50):
    
    chunks = textwrap.wrap(text,chunksize)

    # chunks = [text[i:i + chunksize] for i in range(0, len(text), chunksize)]
    return chunks


def chunkiText(text, max_length = 50):
    list_of_lines = []
    while len(text) > max_length:
        line_length = text[:max_length].rfind(' ')
        list_of_lines.append(text[:line_length])
        text = text[line_length + 1:]
    list_of_lines.append(text)
    return list_of_lines


    


class RNNLanguageModel(LanguageModel,nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, vocab_index, num_layers=1):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.vocab_index = vocab_index  # Store vocab_index as an attribute
        self.num_layers = num_layers
        
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        output = self.fc(hidden[-1])
        return output 
    
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a sequence of characters following the context. 
        That is, returns log P(next_chars | context) = log P(next_char1 | context) + log P(next_char2 | context, next_char1), ...
        :param next_chars: List or tensor of next characters (target sequence).
        :param context: Initial context (previous characters).
        :return: log probability of the sequence.
        """
        # Initialize hidden state and context
        hidden = self.initHidden()  # e.g., torch.zeros(1, hidden_dim) for the first hidden state
        log_prob = 0  # Initialize log probability

        for i in range(len(next_chars)):
            # Get input tensor for the current character (use `context` for previous state)
            current_char = next_chars[i]
            context_tensor = torch.tensor([context])  # Assuming context is an integer or tensor of previous chars

            # Forward pass to get output logits
            output_logits = self.forward(context_tensor)  # Shape: [batch_size, vocab_size]
            
            # Convert logits to probabilities
            probs = F.softmax(output_logits, dim=1)  # Probabilities for each class
            
            # Get log probability of the next character
            target_char_idx = torch.tensor([current_char])  # Index of the next character in vocab
            log_prob_char = torch.log(probs[0, target_char_idx])  # Log probability of target_char
            log_prob += log_prob_char.item()  # Add to total log probability

            # Update the context (e.g., add the predicted character to context if needed)
            context = context + [current_char]  # Update context with the new character

        return log_prob



def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    
    vocab_size = len(vocab_index)
    embed_dim = getattr(args, 'embed_dim', 128)  # Default embedding dimension
    hidden_dim = getattr(args, 'hidden_dim', 256)  # Default hidden dimension
    num_layers = getattr(args, 'num_layers', 1)
    output_dim = getattr(args, 'output_dim', vocab_size)
    learning_rate = getattr(args, 'learning_rate', 0.0001)
    num_epochs = getattr(args, 'num_epochs', 20)
    batch_size = getattr(args, 'batch_size', 32)
    
    
    
    model = RNNLanguageModel(vocab_size, embed_dim, hidden_dim, output_dim, vocab_index)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    chunked_text = textChunker(train_text)
    
    # print(chunked_text[2])
        
    train_data = [(string_to_tensor(s, vocab_index, 50)) for s in chunked_text] 
    
    sequence_length = 5
    
    # train_tensors = [(b[i:i+sequence_length],b[i: i+1+sequence_length]) for b in train_data for i,_ in enumerate(b) if i<len(b)- sequence_length]
    # train_tensors = [(s, torch.tensor(label, dtype=torch.long)) for s in train_data]
    # Convert training data to DataLoader for batching
    
    train_tensors = [(b[i:i+sequence_length],b[i+1+sequence_length]) for b in train_data for i,_ in enumerate(b) if i<len(b)- sequence_length-1]

    
    
    # print(train_tensors[1])


    
    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)

    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        # last_loss = 0 
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)
            
            # labels = labels.float()
            
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            total_loss += loss.item()
            
            # if i % 1000 == 999:
            #     last_loss = running_loss / 1000 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(training_loader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.

             
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        
        # for X, Y in train_loader:
        #     if X.shape[0] != model.batch_size:
        #         continue
        #     hidden = model.init_zero_hidden(batch_size=model.batch_size)

        #     # X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)

        #     # 2. clear gradients
        #     model.zero_grad()

        #     loss = 0
        #     for c in range(X.shape[1]):
        #         out, hidden = model(X[:, c].reshape(X.shape[0],1), hidden)
        #         l = criterion(out, Y[:, c].long())
        #         loss += l

        #     # 4. Compte gradients gradients
        #     loss.backward()
            
        
    
    
    # raise Exception("Implement me")
