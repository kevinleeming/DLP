from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import pickle



"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function
4. Gaussian score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, conversion words, Gaussian score, generation words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
========================================================================================"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
latent_size = 32
embedding_size = 8
#The number of vocabulary
vocab_size = 29
teacher_forcing_ratio = 1.0
#empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.01
isMonotonic = True

MAX_LENGTH = 25
#---------------Data---------------#
tense2index = {'sp': 0, 'tp': 1, 'pg': 2, 'p': 3}

def prepareTrainData():
    with open('./train.txt', 'r') as reader:
        lines = reader.read().split('\n')
    data = []
    for line in lines:
        for i, word in enumerate(line.split(' ')):
            data.append(((word, i), (word, i)))
    return data

def prepareTestData():
    with open('./test.txt', 'r') as reader:
        lines = reader.read().split('\n')
    with open('./test_tense.txt', 'r') as reader:
        t_lines = reader.read().split('\n')
    data = []
    for i in range(len(lines)):
        words = lines[i].split(' ')
        tenses = t_lines[i].split(' ')
        data.append(((words[0], tense2index[tenses[0]]), (words[1], tense2index[tenses[1]])))
    return data

def tensorFromWord(word):
    index_list = [ord(w)-ord('a')+2 for w in word]
    index_list.append(EOS_token)
    return torch.tensor(index_list, dtype=torch.long, device=device).view(-1, 1)

def tensorFromPair(pair):
    input_tensor = tensorFromWord(pair[0][0])
    target_tensor = tensorFromWord(pair[1][0])
    input_c = torch.tensor(pair[0][1], dtype=torch.long, device=device).view(1, -1)
    target_c = torch.tensor(pair[1][1], dtype=torch.long, device=device).view(1, -1)
    return (input_tensor, target_tensor), (input_c, target_c)

def index2chr(index):
    if index >= 2:
        return chr(index-2+ord('a'))
    else:
        return ''

################################
#Example inputs of compute_bleu
################################
#The target word
reference = 'accessed'
#The word generated by your model
output = 'access'

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""

def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = './train.txt'#should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)

        self.h_mu = nn.Linear(hidden_size, latent_size)
        self.h_logvar = nn.Linear(hidden_size, latent_size)
        self.c_mu = nn.Linear(hidden_size, latent_size)
        self.c_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hidden, cell) = self.LSTM(output, (hidden, cell))
        return output, hidden, cell, self.h_mu(hidden), self.h_logvar(hidden), self.c_mu(cell), self.c_logvar(cell)

    def initHidden(self, input_c):
        hidden = torch.zeros(1, 1, self.hidden_size-input_c.size(2), device=device)
        hidden = torch.cat((hidden, input_c), 2)
        return hidden

    #def initCell(self):
        #return torch.zeros(1, 1, self.hidden_size, device=device)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.LSTM(output, (hidden, cell))
        output = self.out(output[0])
        return output, hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#VAE
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(VAE, self).__init__()

        self.embedding = nn.Embedding(5, embedding_size)
        self.encoder = EncoderRNN(input_size, hidden_size, latent_size)
        self.fc1 = nn.Linear(latent_size+embedding_size, hidden_size)
        self.fc2 = nn.Linear(latent_size+embedding_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_size)
    def forward(self, input_tensor, input_c, target_c):
        input_c = self.embedding(input_c)
        encoder_hidden = self.encoder.initHidden(input_c)
        encoder_cell = self.encoder.initHidden(input_c)

        input_length = input_tensor.size(0)
        for ei in range(input_length):
            encoder_output, encoder_hidden, encoder_cell, h_mu, h_logvar, c_mu, c_logvar = self.encoder(input_tensor[ei], encoder_hidden, encoder_cell)
        
        return self.generate(h_mu, c_mu, target_c)

    def generate(self, h_latent, c_latent, target_c):
        decoder_input = torch.tensor([[SOS_token]], device=device)
        target_c = self.embedding(target_c)
        decoder_hidden = torch.cat((h_latent, target_c), 2)
        decoder_hidden = self.fc1(decoder_hidden)
        #decoder_cell = encoder_cell
        decoder_cell = torch.cat((c_latent, target_c), 2)
        decoder_cell = self.fc2(decoder_cell)

        pred_list = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            pred_list.append(index2chr(topi))
            if decoder_input.item() == EOS_token:
                break
        pred = ''.join(pred_list)
        
        return pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

def loss_function(y_pred, y, h_mu, h_logvar, c_mu, c_logvar):
    BCE = F.cross_entropy(y_pred, y)
    h_KLD = -0.5 * torch.sum(1 + h_logvar - h_mu.pow(2) - h_logvar.exp())
    c_KLD = -0.5 * torch.sum(1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
    return BCE, h_KLD, c_KLD

def train(input_tensor, target_tensor, input_c, target_c, model, optimizer, criterion, KLD_weight, max_length=MAX_LENGTH):
    model.train()
    input_c = model.embedding(input_c)
    encoder_hidden = model.encoder.initHidden(input_c)
    encoder_cell = model.encoder.initHidden(input_c)

    optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    cross_loss = 0
    KLD_loss = 0

    #----------sequence to sequence part for encoder----------#
    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_cell, h_mu, h_logvar, c_mu, c_logvar = model.encoder(input_tensor[ei], encoder_hidden, encoder_cell)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    target_c = model.embedding(target_c)
    h_latent = model.reparameterize(h_mu, h_logvar)
    decoder_hidden = torch.cat((h_latent, target_c), 2)
    decoder_hidden = model.fc1(decoder_hidden)
    #decoder_cell = encoder_cell
    c_latent = model.reparameterize(c_mu, c_logvar)
    decoder_cell = torch.cat((c_latent, target_c), 2)
    decoder_cell = model.fc2(decoder_cell)


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = model.decoder(decoder_input, decoder_hidden, decoder_cell)
            _cross_loss, h_KLD_loss, c_KLD_loss = criterion(decoder_output, target_tensor[di], h_mu, h_logvar, c_mu, c_logvar)
            KLD_loss = (h_KLD_loss+c_KLD_loss)/2
            cross_loss += _cross_loss
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = model.decoder(decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            _cross_loss, h_KLD_loss, c_KLD_loss = criterion(decoder_output, target_tensor[di], h_mu, h_logvar, c_mu, c_logvar)
            KLD_loss = (h_KLD_loss+c_KLD_loss)/2
            cross_loss += _cross_loss
            if decoder_input.item() == EOS_token:
                break

    loss = cross_loss + KLD_weight * KLD_loss
    loss.backward()

    optimizer.step()

    return loss.item()/target_length, cross_loss.item()/target_length, KLD_loss/target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def KL_annealing(iter, isMonotonic):
    # KLD_weight
    if isMonotonic:
        return min(1.0, iter/50000)
    else:
        return min(1.0, (iter % 20000)/10000)

def evaluate(model, isPrint):
    model.eval()
    with torch.no_grad():
        testing_pairs = []
        testing_cs = []
        for i in range(len(test_data)):
            pairs = tensorFromPair(test_data[i])
            testing_pairs.append(pairs[0])
            testing_cs.append(pairs[1])
        bleu_total = 0
        for i in range(len(test_data)):
            x = test_data[i][0][0]
            target = test_data[i][1][0]
            y_pred = model(testing_pairs[i][0], testing_cs[i][0], testing_cs[i][1])
            bleu_total += compute_bleu(y_pred, target)
            if isPrint:
                print('==============================')
                print(f'input:\t{x}')
                print(f'target:\t{target}')
                print(f'pred:\t{y_pred}')
        if isPrint:
            print(f'Average BLEU-4 score: {bleu_total/len(test_data)}')

        # gaussian score
        words_list = []
        for i in range(100):
            h_latent = torch.randn(1,1,latent_size, device=device)
            c_latent = torch.randn(1,1,latent_size, device=device)
            word = []
            for _t in range(4):
                word.append(model.generate(h_latent, c_latent, torch.tensor(_t, dtype=torch.long, device=device).view(1, -1)))
            words_list.append(word)
        gau_score = Gaussian_score(words_list)
        if isPrint:
            print('==============================')
            for i, word in enumerate(words_list):
                print(word)
                if i >= 4:
                    break
            print(f'Gaussian score: {gau_score}')
    return bleu_total/len(test_data), gau_score

def trainIters(model, n_iters, print_every=1000, plot_every=100, learning_rate=0.05):
    start = time.time()
    cross_loss_list, KLD_loss_list, bleu_list, gau_list = [], [], [], []
    print_loss_total = 0  # Reset every print_every
    plot_cross_total = 0  # Reset every plot_every
    plot_KLD_total = 0

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # your own dataloader
    training_pairs = []
    training_cs = []
    for i in range(n_iters):
        pairs = tensorFromPair(random.choice(train_data))
        training_pairs.append(pairs[0])
        training_cs.append(pairs[1])

    # Train
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        training_c = training_cs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        input_c = training_c[0]
        target_c = training_c[1]

        loss, cross_loss, KLD_loss = train(input_tensor, target_tensor, input_c, target_c,
                     model, optimizer, loss_function, KL_annealing(iter, isMonotonic))
        print_loss_total += loss
        plot_cross_total += cross_loss
        plot_KLD_total += KLD_loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_cross_avg = plot_cross_total / plot_every
            plot_cross_total = 0
            plot_KLD_avg = plot_KLD_total / plot_every
            plot_KLD_total = 0
            cross_loss_list.append(plot_cross_avg)
            KLD_loss_list.append(plot_KLD_avg)
            if iter % n_iters == 0:
                bleu, gau_score = evaluate(model, True)
            else:
                bleu, gau_score = evaluate(model, False)
            bleu_list.append(bleu)
            gau_list.append(gau_score)
            if bleu >= 0.7 and gau_score >= 0.2:
                torch.save({'bleu': bleu, 'gau': gau_score, 'state_dict': model.state_dict()}, f'./model/model_{gau_score}_{bleu:.2f}.tar')
                evaluate(model, True)
                print(f'***************Best: {bleu} {gau_score}')

    with open('curve.pickle', 'wb') as writer:
        pickle.dump({
                'Iter': n_iters,
                'plot_every': plot_every,
                'cross_loss': cross_loss_list,
                'KLD_loss': KLD_loss_list,
                'test_bleu': bleu_list,
                'test_gau': gau_list,
            }, writer)
    x_list = [i for i in range(1, n_iters+1, plot_every)]
    plt.title('Loss')
    plt.plot(x_list, cross_loss_list, label='CrossEntropy')
    plt.plot(x_list, KLD_loss_list, label='KL loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./Loss.png')
    plt.clf()

    plt.title('Score')
    plt.plot(x_list, bleu_list, label='BLEU-4 score')
    plt.plot(x_list, gau_list, label='Gaussian score')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'./Score.png')
    plt.clf()




train_data = prepareTrainData()
test_data = prepareTestData()
model = VAE(vocab_size, hidden_size, latent_size, vocab_size).to(device)
trainIters(model, 100000, print_every=5000, plot_every=100, learning_rate=LR)