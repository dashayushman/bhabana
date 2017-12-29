import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
from bhabana.utils import generic_utils as gu


def position_encoding(embedded_sentence):
    '''
    embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
    l.size() -> (#sentence, #embedding)
    output.size() -> (#batch, #sentence, #embedding)
    '''
    _, slen, elen = embedded_sentence.size()

    l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
    l = torch.FloatTensor(l)
    l = l.unsqueeze(0) # for #batch
    #l = l.unsqueeze(1) # for #sen
    l = l.expand_as(embedded_sentence)
    weighted = embedded_sentence * Variable(l.cuda())
    return weighted # sum with tokens


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.U.state_dict()['weight'])

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = F.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = F.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h


class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size)).cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal(self.z1.state_dict()['weight'])
        init.xavier_normal(self.z2.state_dict()['weight'])
        init.xavier_normal(self.next_mem.state_dict()['weight'])

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)

        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''

        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem, G


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions, word_embedding):
        '''
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions


class InputModule(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, word_embedding, batch_size):
        '''
        contexts.size() -> (#batch, #sentence, #token)
        word_embedding() -> (#batch, #sentence x #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        '''
        #batch_num, sen_num, token_num = contexts.size()

        #contexts = contexts.view(batch_num, -1)
        contexts = word_embedding(contexts)

        #contexts = contexts.view(batch_num, sen_num, token_num, -1)
        contexts = position_encoding(contexts)
        contexts = self.dropout(contexts)

        h0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
        facts, hdn = self.gru(contexts, h0)
        facts = facts[:, :, :self.hidden_size] + facts[:, :, self.hidden_size:]
        return facts


class AnswerModule(nn.Module):
    def __init__(self, n_classes, hidden_size, activation=None):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, n_classes)
        init.xavier_normal(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)
        self.activation = gu.get_activation_by_name(activation)
        if self.activation is not None:
            self.activation = self.activation()

    def forward(self, M, questions):
        M = self.dropout(M)
        concat = torch.cat([M, questions], dim=2).squeeze(1)
        z = self.z(concat)
        if self.activation is not None:
            z = self.activation(z)
        return z


class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, n_output_neurons,
                 memory_dims=128, num_hop=3, pretrained_word_vectors=None,
                 trainable=True, output_activation=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.memory_dims = memory_dims
        w2v_dims = pretrained_word_vectors.shape[-1] if \
            pretrained_word_vectors is not None else hidden_size
        self.word_embedding = nn.Embedding(vocab_size, w2v_dims, padding_idx=0)
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3**0.5), b=3**0.5)
        #self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.input_module = InputModule(w2v_dims, hidden_size)
        self.questions = nn.Parameter(torch.randn(1, 1, memory_dims),
                                            requires_grad=True)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(n_output_neurons, hidden_size,
                                          output_activation)
        self.init_weights(pretrained_word_vectors, trainable)

    def init_weights(self, pretrained_word_vectors, trainable=False):
        if pretrained_word_vectors is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(pretrained_word_vectors).float())
        if not trainable:
            self.word_embedding.weight.requires_grad = False

    def forward(self, contexts, batch_size):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        '''
        attention_scores = []
        facts = self.input_module(contexts, self.word_embedding, batch_size)
        questions = self.questions.expand(batch_size, 1, self.memory_dims)
        M = self.questions.expand(batch_size, 1, self.memory_dims)
        for hop in range(self.num_hop):
            M, attention = self.memory(facts, questions, M)
            attention_scores.append(attention)
        preds = self.answer_module(M, questions)
        return preds, attention_scores