import torch.nn as nn

from bhabana.models import DMNPlus


class MemoryRegressor(nn.Module):
    def __init__(self, vocab_size, n_output_neurons, hidden_size=128,
                 memory_dims=128, num_hop=3, pretrained_word_vectors=None,
                 trainable_embeddings=True, output_activation=None):
        super(MemoryRegressor, self).__init__()
        self.model = DMNPlus(hidden_size=hidden_size, vocab_size=vocab_size,
                             n_output_neurons=n_output_neurons,
                             memory_dims=memory_dims, num_hop=num_hop,
                             pretrained_word_vectors=pretrained_word_vectors,
                             trainable=trainable_embeddings,
                             output_activation=output_activation)

    def get_embedding_weights(self):
        return self.model.word_embedding.weight.clone().data.cpu()

    def forward(self, data):
        output, attention = self.model(data["text"], data["batch_size"])
        return {"out": output, "attention": attention}