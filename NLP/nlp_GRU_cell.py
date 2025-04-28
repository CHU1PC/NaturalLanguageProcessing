import glob
import os
import random
import string
import unicodedata
from io import open

import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


class GRUNet(nn.Module):
    def __init__(self, input_size, n_categories, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRUCell(input_size + n_categories, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        combined = torch.cat((category, input), 1)
        # RNNだとhiddenとoutputを一回ずつ作ってまたcombineしないといけない
        hidden = self.gru(combined, hidden)  # このGRUCellが一回で新しい隠れ状態を作れちゃう
        output = self.i2o(hidden)
        output = self.dropout(output)
        output = self.logsoftmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)


def unicodeToAscii(s):  # 変な記号文字などをなくす
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):  # \nごとにlinesに代入
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]


def randomChoice(list):
    return list[random.randint(0, len(list) - 1)]


def randomTrainingPair(all_categories, category_lines):
    category = randomChoice(all_categories)  # all_categoriesからランダムでcategoryをとる
    line = randomChoice(category_lines[category])
    return category, line  # 国名, 苗字を返す


def categoryTensor(category, all_categories):
    """
    カテゴリ名をワンホットベクトルに変換する
    例(
        all_categories = ['English', 'French', 'German'] のとき
        categoryTensor('French') の返り値は
        [[0, 1, 0]] となります。
    )
    """
    n_categories = len(all_categories)
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories, device=device)
    tensor[0][li] = 1
    return tensor


def inputTensor(line):
    # 文字数, batch size, 文字種類数
    tensor = torch.zeros(len(line), 1, n_letters, device=device)
    for li in range(len(line)):  # 文字に対してfor loop を回してる
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1  # 該当部分だけ1に変更
    return tensor


def targetTensor(line):
    # 次の文字を予測するためにrangeが(1, len(line))なっている
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # n_letters - 1はEOSである
    return torch.LongTensor(letter_indexes).to(device)


def train(rnn, optimizer, criterion, category_tensor, input_line_tensor, target_line_tensor):  # noqa
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    outputs = []  # 前回は分類だからいらないけど今回は予想のため過去のデータがいる

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        outputs.append(output)

    # [シーケンス長, n_letters] になるようにまとめる
    outputs = torch.cat(outputs, dim=0)  # type: ignore
    # target_line_tensorのshapeは[シーケンス長]
    loss = criterion(outputs, target_line_tensor)

    loss.backward()
    optimizer.step()

    return outputs, loss.item()


# Sample from a category and starting letter
def sample(rnn, category, all_categories, start_letter='A', max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category, all_categories)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(rnn, category, all_categories, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(rnn, category, all_categories, start_letter))


def main():
    category_lines = {}
    all_categories = []
    for filename in glob.glob(
        "D:/program/programming/python/study/pytorch3~/data2/names/*.txt"
    ):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)

    hidden_size = 128
    rnn = GRUNet(n_letters, n_categories, hidden_size, n_letters).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.005)

    # 学習ループ
    n_iters = 10000
    print_every = 1000
    for iter in range(1, n_iters + 1):
        category, line = randomTrainingPair(all_categories, category_lines)
        category_tensor = categoryTensor(category, all_categories)
        input_line_tensor = inputTensor(line)
        target_line_tensor = targetTensor(line)

        outputs, loss = \
            train(rnn, optimizer, criterion,
                  category_tensor, input_line_tensor, target_line_tensor)

        if iter % print_every == 0:
            print(f"Iter {iter} Loss: {loss:.4f}")

    # サンプリング
    print("\n=== サンプル生成 ===")
    samples(rnn, random.choice(all_categories), all_categories, start_letters="ABC")  # noqa


if __name__ == "__main__":
    main()
