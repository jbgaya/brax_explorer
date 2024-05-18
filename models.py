from torch import nn

class BaselineModel(nn.Module):
    """
    A baseline model for text classification.

    Args:
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimension of the word embeddings.
        num_class (int): The number of classes for classification.

    Attributes:
        embedding (nn.EmbeddingBag): The embedding layer for word embeddings with averaging.
        fc (nn.Linear): The fully connected layer for classification.
    """

    def __init__(self, vocab_size, embed_dim, num_class):
        super(BaselineModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        """
        Forward pass of the model.

        Args:
            text (torch.Tensor): The input text tokenized tensor.
            offsets (torch.Tensor): The offsets tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)