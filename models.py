# models.py

import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
#####################
# MODELS FOR PART 1 #
#####################

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


# class RNNClassifier(ConsonantVowelClassifier):
#     def predict(self, context):
#         raise Exception("Implement me")

# class RNNClassifier(ConsonantVowelClassifier):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.softmax = nn.Softmax(dim=1)  # Apply softmax for probability outputs
#
#     def forward(self, x):
#         # Add a batch dimension if not using batching
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#
#         embedded = self.embedding(x)
#         # LSTM expects input of shape [batch size, seq length, embed dim] due to batch_first=True
#         lstm_out, (hidden, _) = self.lstm(embedded)
#
#         # We only use the last hidden state from the final LSTM layer
#         output = self.fc(hidden[-1])
#         output = self.softmax(output)  # Apply softmax for class probabilities
#
#         return output
#
#     def predict(self, context, vocab_index):
#         """
#         Predicts whether the given context string ends in a consonant or a vowel.
#
#         :param context: A raw string of characters to be classified.
#         :param vocab_index: The Indexer of the character vocabulary.
#         :return: Predicted class (0 for consonant, 1 for vowel)
#         """
#         # Convert the context string to a tensor of indices
#         input_tensor = string_to_tensor(context, vocab_index)
#
#         # Ensure the input tensor has the appropriate batch dimension
#         input_tensor = input_tensor.unsqueeze(0)  # Adds a batch dimension to make [1, seq length]
#
#         # Set the model to evaluation mode
#         self.eval()
#
#         # No gradient computation during prediction
#         with torch.no_grad():
#             output = self.forward(input_tensor)
#
#         # Get the predicted class (0 for consonant, 1 for vowel)
#         predicted_class = torch.argmax(output, dim=1).item()
#
#         return predicted_class

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

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


# def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
#     """
#     :param args: command-line args, passed through here for your convenience
#     :param train_cons_exs: list of strings followed by consonants
#     :param train_vowel_exs: list of strings followed by vowels
#     :param dev_cons_exs: list of strings followed by consonants
#     :param dev_vowel_exs: list of strings followed by vowels
#     :param vocab_index: an Indexer of the character vocabulary (27 characters)
#     :return: an RNNClassifier instance trained on the given data
#     """
#     raise Exception("Implement me")

# def string_to_tensor(s, vocab_index):
#     """
#     Converts a raw string to a PyTorch tensor of indices based on the vocab_index.
#     """
#     indices = [vocab_index[char] for char in s]  # Convert each character to its index
#     return torch.tensor(indices, dtype=torch.long)

def string_to_tensor(s, vocab_index):
    """
    Converts a raw string to a PyTorch tensor of indices based on the vocab_index.
    """
    indices = [vocab_index.index_of(char) for char in s]  # Use index_of to get index of each character
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
    num_epochs = getattr(args, 'num_epochs', 10)
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


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")
