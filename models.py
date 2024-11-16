# models.py
import torch
import torch.nn as nn
import numpy as np
import collections
import random
import torch.optim as optim
from torch.utils.data import DataLoader

#####################
# MODELS FOR PART 1 #
#####################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables benchmark to ensure reproducibility

# Setting a seed value so that the results could be produced again
set_seed(26)

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
    def __init__(self, indexer, embed_dim, hidden_dim, output_dim, num_layers=1, dropout=0.6):
        """
        Initializes the RNNClassifier with the Indexer for vocabulary management.

        :param indexer: An Indexer instance for managing the vocabulary.
        :param embed_dim: Dimension of embeddings.
        :param hidden_dim: Dimension of the LSTM hidden state.
        :param output_dim: Number of output classes (binary: consonant vs. vowel).
        :param num_layers: Number of LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(RNNClassifier, self).__init__()  # Initialize both parent classes
        self.indexer = indexer
        self.embedding = nn.Embedding(len(indexer), embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)  # Apply dropout to embeddings
        lstm_out, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output  # Return raw logits without softmax

    def predict(self, context):
        input_tensor = string_to_tensor(context, self.indexer).unsqueeze(0)  # Use Indexer for encoding
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class


def string_to_tensor(s, indexer, max_length=20):
    """
    Converts a raw string to a PyTorch tensor of indices based on the Indexer,
    truncating or padding to max_length.

    :param s: Input string.
    :param indexer: Indexer instance for character-to-index mapping.
    :param max_length: Maximum sequence length.
    :return: A tensor of indices.
    """
    s = s[:max_length]  # Truncate to max_length if necessary
    indices = [indexer.index_of(char) for char in s]
    if len(indices) < max_length:
        # Pad with a padding index (assuming 0 is the padding index)
        indices += [0] * (max_length - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, indexer):
    """
    Trains an RNNClassifier using the provided Indexer for vocabulary.

    :param args: Command-line args for hyperparameters.
    :param train_cons_exs: List of strings followed by consonants.
    :param train_vowel_exs: List of strings followed by vowels.
    :param dev_cons_exs: List of strings followed by consonants.
    :param dev_vowel_exs: List of strings followed by vowels.
    :param indexer: An Indexer instance for vocabulary management.
    :return: A trained RNNClassifier instance.
    """
    # Hyperparameters from args
    embed_dim = getattr(args, 'embed_dim', 64)
    hidden_dim = getattr(args, 'hidden_dim', 128)
    output_dim = 2  # Binary classification (consonant vs. vowel)
    num_layers = getattr(args, 'num_layers', 1)
    dropout = getattr(args, 'dropout', 0.6)
    learning_rate = getattr(args, 'learning_rate', 0.001)
    num_epochs = getattr(args, 'num_epochs', 16)
    batch_size = getattr(args, 'batch_size', 32)

    # Create model, pass Indexer during initialization
    model = RNNClassifier(indexer, embed_dim, hidden_dim, output_dim, num_layers, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare training and validation data
    train_data = [(string_to_tensor(s, indexer), 0) for s in train_cons_exs] + \
                 [(string_to_tensor(s, indexer), 1) for s in train_vowel_exs]
    dev_data = [(string_to_tensor(s, indexer), 0) for s in dev_cons_exs] + \
               [(string_to_tensor(s, indexer), 1) for s in dev_vowel_exs]

    # Convert training data to DataLoader for batching
    train_tensors = [(s, torch.tensor(label, dtype=torch.long)) for s, label in train_data]
    dev_tensors = [(s, torch.tensor(label, dtype=torch.long)) for s, label in dev_data]

    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    dev_loader = DataLoader(dev_tensors, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = zip(*batch)
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
            labels = torch.stack(labels)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total

        # Validation accuracy
        model.eval()
        dev_correct = 0
        dev_total = 0
        with torch.no_grad():
            for dev_batch in dev_loader:
                dev_inputs, dev_labels = zip(*dev_batch)
                dev_inputs = nn.utils.rnn.pad_sequence(dev_inputs, batch_first=True)
                dev_labels = torch.stack(dev_labels)

                dev_outputs = model(dev_inputs)
                dev_predictions = torch.argmax(dev_outputs, dim=1)
                dev_correct += (dev_predictions == dev_labels).sum().item()
                dev_total += dev_labels.size(0)

        dev_accuracy = dev_correct / dev_total

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {dev_accuracy * 100:.2f}%")

        model.train()

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


class RNNLanguageModel(nn.Module):  # Inherit from nn.Module
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab_index):
        super(RNNLanguageModel, self).__init__()  # Call parent class constructor
        self.vocab_index = vocab_index
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0) # RNN layer (LSTM)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Fully connected output layer
        self.softmax = nn.LogSoftmax(dim=-1)  # Log softmax for probabilities

    def forward(self, x, hidden=None):
        # Convert indices to embeddings
        embedded = self.embedding(x)  # x: (batch_size, seq_len) -> embedded: (batch_size, seq_len, embedding_dim)
        # Process embeddings through RNN
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, seq_len, hidden_dim)
        # Pass the RNN output to the fully connected layer
        logits = self.fc(output)  # logits: (batch_size, seq_len, vocab_size)
        return self.softmax(logits), hidden  # Return log probabilities and hidden state

    def get_log_prob_single(self, next_char, context):
        """
        Log P(next_char | context): Predicts probability for one character.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Ensure context is non-empty
            if len(context) == 0:
                context = " "  # Use space as fallback

            # Convert context to indices
            context_indices = torch.tensor(
                [self.vocab_index.index_of(c) for c in context], dtype=torch.long
            ).unsqueeze(0)

            # Forward pass
            logits, _ = self.forward(context_indices)
            next_char_index = self.vocab_index.index_of(next_char)

            # Return the log probability of the next character
            return logits[0, -1, next_char_index].item()

    def get_log_prob_sequence(self, next_chars, context):
        """
        Log P(sequence | context): Predicts probabilities for multiple characters.
        """
        log_prob = 0.0
        current_context = context  # Initialize the context
        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for next_char in next_chars:
                # Ensure context is non-empty
                if len(current_context) == 0:
                    current_context = " "  # Use space as a fallback for empty context

                # Convert context to indices (numbers)
                context_indices = torch.tensor(
                    [self.vocab_index.index_of(c) for c in current_context], dtype=torch.long
                ).unsqueeze(0)

                # Forward pass to get logits
                logits, _ = self.forward(context_indices)
                next_char_index = self.vocab_index.index_of(next_char)

                # Add log probability for the next character
                log_prob += logits[0, -1, next_char_index].item()

                # Update the context by appending the predicted character
                current_context += next_char

                # Keep context size manageable (e.g., trim to last 50 chars)
                current_context = current_context[-50:]

        return log_prob


def create_chunks(text, chunk_size):
    """
    Splits the text into input-output chunks for training.

    Parameters:
        text (str): The full text to be chunked.
        chunk_size (int): The size of each input chunk.

    Returns:
        inputs (list of str): The input text chunks.
        outputs (list of str): The output text chunks (shifted by one character).
    """
    inputs = []
    outputs = []

    # Loop through the text, creating input and output chunks
    for i in range(len(text) - chunk_size):
        inputs.append(text[i:i + chunk_size])  # Input chunk (e.g., "hello worl")
        outputs.append(text[i + 1:i + chunk_size + 1])  # Output chunk (e.g., "ello world")

    return inputs, outputs


def encode_chunks(chunks, vocab_index):
    """
    Converts text chunks into tensor indices that the model can process.

    Parameters:
        chunks (list of str): Text chunks (input or output).
        vocab_index (Indexer): Maps characters to numeric indices.

    Returns:
        encoded (list of list of int): Encoded chunks as lists of indices.
    """
    encoded = []

    # Convert each character in each chunk to its corresponding index
    for chunk in chunks:
        encoded.append([vocab_index.index_of(c) for c in chunk])

    return encoded


def train_lm(args, train_text, dev_text, vocab_index):
    """
    Trains an RNN-based language model on the given training text.

    Parameters:
        args: Command-line arguments passed to the script.
        train_text (str): The training text.
        dev_text (str): The development (testing) text.
        vocab_index (Indexer): Maps characters to numeric indices.

    Returns:
        model (RNNLanguageModel): The trained language model.
    """

    print("Preparing training data...")
    # Set hyperparameters for the model
    embedding_dim = 50  # Dimension of character embeddings
    hidden_dim = 100  # Number of hidden units in the RNN
    batch_size = 50  # Number of characters per input chunk
    learning_rate = 0.001  # Learning rate for the optimizer
    num_epochs = 10  # Number of passes through the training data
    chunk_size = 50

    # Initialize the RNN language model
    model = RNNLanguageModel(len(vocab_index), embedding_dim, hidden_dim, vocab_index)

    # Define the loss function (CrossEntropyLoss) for classification
    criterion = nn.CrossEntropyLoss()

    # Use the Adam optimizer to adjust model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Step 1: Prepare training data (chunking and encoding)
    print("Preparing training data...")
    inputs, outputs = create_chunks(train_text, chunk_size=chunk_size)  # Split text into chunks
    inputs = encode_chunks(inputs, vocab_index)  # Convert input chunks to indices
    outputs = encode_chunks(outputs, vocab_index)  # Convert output chunks to indices

    # Convert to PyTorch tensors for model training
    inputs = torch.tensor(inputs, dtype=torch.long)  # Input tensor (batch_size, seq_len)
    outputs = torch.tensor(outputs, dtype=torch.long)  # Output tensor (batch_size, seq_len)

    print("Starting training...")
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):  # Train for num_epochs iterations
        epoch_loss = 0  # Track the total loss for this epoch

        # Loop through batches
        for i in range(0, len(inputs), batch_size):
            x_batch = inputs[i:i + batch_size]  # Batch of inputs
            y_batch = outputs[i:i + batch_size]  # Batch of outputs

            # Ensure the hidden state is detached at the start of each batch
            optimizer.zero_grad()  # Clear previous gradients
            logits, _ = model(x_batch)  # Forward pass
            loss = criterion(logits.view(-1, len(vocab_index)), y_batch.view(-1))  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            # Accumulate loss for monitoring
            epoch_loss += loss.item()

            # Print progress every 10% of the data
            if i % max(len(inputs) // 10, 1) == 0:  # Avoid division by zero
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

        # Print the average loss for this epoch
        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(inputs):.4f}")

    # Evaluate the model on the development set
    print("Evaluating on dev set...")
    dev_log_prob = model.get_log_prob_sequence(dev_text, "")
    print(f"Dev Log Prob: {dev_log_prob:.4f}")

    return model