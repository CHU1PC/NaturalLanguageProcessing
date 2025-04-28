import glob
import os
import random
import string
import unicodedata

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFD", input_str)
    return ''.join([c for c in nfkd_form if unicodedata.category(c) != 'Mn'])


def data_maker():
    path = r"D:\program\programming\python\study\pytorch3~\data2\names\*txt"
    all_letters = string.ascii_letters + " .,;'-"

    category_lines = {}
    all_categories = []
    for filepath in glob.glob(path):
        category = os.path.splitext(os.path.basename(filepath))[0]
        all_categories.append(category)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = []
            for line in f.readlines():
                line = remove_accents(line.strip())
                # all_lettersに含まれない文字を除去
                line = ''.join([c for c in line if c in all_letters])
                if line:  # 空行は除外
                    lines.append(line)
            category_lines[category] = lines
    return category_lines, all_categories


# 単一バッチ
###############################################################################
# class GRUNet(nn.Module):                                                    #
#     def __init__(self, input_size, n_categories, hidden_size, output_size): #
#         super().__init__()                                                  #
#         self.hidden_size = hidden_size                                      #
#         self.gru = nn.GRUCell(input_size + n_categories, hidden_size)       #
#         self.i2o = nn.Linear(hidden_size, output_size)                      #
#         self.dropout = nn.Dropout(p=0.1)                                    #
#         self.logsoftmax = nn.LogSoftmax(dim=1)                              #
#                                                                             #
#     def forward(self, category, input, hidden):                             #
#         combined = torch.cat((category, input), 1)                          #
#         hidden = self.gru(combined, hidden)                                 #
#         output = self.i2o(hidden)                                           #
#         output = self.dropout(output)                                       #
#         output = self.logsoftmax(output)                                    #
#         return output, hidden                                               #
#                                                                             #
#     def initHidden(self):                                                   #
#         return torch.zeros(1, self.hidden_size, device=device)              #
###############################################################################

# 複数バッチ
class GRUNet(nn.Module):
    def __init__(self, input_size, n_categories, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size + n_categories, hidden_size, num_layers=2)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # バッチ用のforward
    def forward(self, category, input):
        # input: [seq_len, batch_size, input_size]
        # category: [batch_size, n_categories]
        batch_size = input.size(1)
        category = category.unsqueeze(0).expand(input.size(0), batch_size, category.size(1))
        combined = torch.cat((category, input), 2)
        output, hidden = self.gru(combined)
        output = self.i2o(output)
        output = self.dropout(output)
        output = self.logsoftmax(output)
        return output, hidden

    def initHidden(self, batch_size=1):
        # num_layers, batch_size, hidden_size
        return torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, device=device)

    # 単一処理用のforward
    def forward_step(self, category, input, hidden):
        # category: [1, n_categories], input: [1, 1, n_letters], hidden: [1, 1, hidden_size]
        category = category.unsqueeze(0)  # [1, 1, n_categories]
        combined = torch.cat((category, input), 2)  # [1, 1, n_categories + n_letters]
        output, hidden = self.gru(combined, hidden)
        output = self.i2o(output)
        output = self.dropout(output)
        output = self.logsoftmax(output)
        return output.squeeze(0), hidden


def make_batch(category_lines, all_categories, all_letters, batch_size):
    categories = []
    inputs = []
    targets = []
    for _ in range(batch_size):
        category, line = random_training_pair(category_lines, all_categories)
        categories.append(categoryTensor(category, all_categories).squeeze(0))
        input_tensor = inputTensor(line, all_letters)
        target_tensor = targetTensor(line, all_letters)
        inputs.append(input_tensor)
        targets.append(target_tensor)
    # パディングしてテンソル化
    seq_lengths = [inp.size(0) for inp in inputs]
    max_len = max(seq_lengths)
    input_batch = torch.zeros(max_len, batch_size, len(all_letters) + 1, device=device)
    target_batch = torch.full((max_len, batch_size), fill_value=-100, dtype=torch.long, device=device)
    for i in range(batch_size):
        input_batch[:inputs[i].size(0), i, :] = inputs[i].squeeze(1)
        target_batch[:targets[i].size(0), i] = targets[i]
    category_batch = torch.stack(categories)
    return category_batch, input_batch, target_batch, seq_lengths


def categoryTensor(category, all_category):
    # 0xカテゴリー数の零tensorを作る
    tensor = torch.zeros(1, len(all_category), device=device)
    tensor[0, all_category.index(category)] = 1
    return tensor


def inputTensor(line, all_letters):
    n_letters = len(all_letters) + 1  # EOS分
    tensor = torch.zeros(len(line), 1, n_letters, device=device)
    for li, letter in enumerate(line):
        if letter in all_letters:
            tensor[li][0][all_letters.index(letter)] = 1
    return tensor


def targetTensor(line, all_letters):
    n_letters = len(all_letters) + 1
    letter_indexes = [all_letters.index(letter) for letter in line[1:] if letter in all_letters]
    letter_indexes += [n_letters - 1]  # EOS
    return torch.tensor(letter_indexes, dtype=torch.long)


def random_training_pair(category_lines, all_categories):
    category = random.choice(all_categories)
    line = random.choice(category_lines[category]).strip()
    return category, line


def train(rnn, optimizer, criterion, category_batch, input_batch, target_batch, seq_lengths):
    optimizer.zero_grad()
    output, hidden = rnn(category_batch, input_batch)
    # 出力: [seq_len, batch_size, output_size]
    output = output.view(-1, output.size(2))
    target = target_batch.view(-1)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


# Sample from a category and starting letter
def sample(rnn, n_letters, all_letters, category, all_categories, start_letter='A', max_length=20):
    with torch.no_grad():
        category_tensor = categoryTensor(category, all_categories)
        input = inputTensor(start_letter, all_letters)
        hidden = rnn.initHidden()
        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn.forward_step(category_tensor, input[0].unsqueeze(0), hidden)
            # 確率分布からサンプリング
            probs = torch.exp(output)
            topi = torch.multinomial(probs, 1)[0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter, all_letters)
        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(rnn, n_letters, all_letters, category, all_categories, start_letters='ABC'):  # noqa
    for start_letter in start_letters:
        print(sample(rnn, n_letters, all_letters, category, all_categories, start_letter))  # noqa


def randomChoice(list):
    return list[random.randint(0, len(list) - 1)]


def randomTrainingPair(all_categories, category_lines):
    category = randomChoice(all_categories)  # all_categoriesからランダムでcategoryをとる
    line = randomChoice(category_lines[category])
    return category, line  # 国名, 苗字を返す


def main():
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1  # Plus EOS marker
    category_lines, all_categories = data_maker()
    n_category = len(all_categories)
    hidden_size = 1024
    model = GRUNet(n_letters, n_category, hidden_size, n_letters).to(device)
    criterion = nn.NLLLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習ループ
    n_iters = 50000
    print_every = 2000
    batch_size = 16
    for iter in range(1, n_iters + 1):
        category_batch, input_batch, target_batch, seq_lengths = make_batch(
            category_lines, all_categories, all_letters, batch_size)
        loss = \
            train(model, optimizer, criterion, category_batch, input_batch, target_batch, seq_lengths)
        if iter % print_every == 0:
            print(f"Iter {iter} Loss: {loss:.4f}")

    # サンプリング
    print("\n=== サンプル生成 ===")
    samples(model, n_letters, all_letters, random.choice(all_categories),
            all_categories, start_letters="ABC")  # noqa


if __name__ == "__main__":
    main()
