import math
import random
import re
import string
import time
import unicodedata
from io import open

import matplotlib.pyplot as plt
# plt.switch_backend('agg') # コメントアウト (Colab等で問題になる場合)
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


# --- データ前処理 ---
###############################################################################
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOSとEOSの分

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word # index2wordにも追加
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # Note: r" /1" は句読点の前にスペースを入れ、句読点を "1" に置換します。
    # もしスペース挿入のみが目的なら r" \1" が適切かもしれません。
    s = re.sub(r"([.!?])", r" \1", s) # 句読点の前にスペース
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # アルファベットと句読点以外をスペースに
    return s.strip() # 余分なスペースを削除


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    filepath = (
        f"D:/program/programming/python/study/pytorch3~/data/"
        f"{lang1}-{lang2}.txt"
    )
    try:
        with open(filepath, encoding="utf-8") as fd:
            lines = fd.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, []

    pairs = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            pair = stripped_line.split("\t")
            if len(pair) == 2:
                normalized_pair = [normalizeString(s) for s in pair]
                pairs.append(normalized_pair)
            else:
                print(f"Skipping invalid line: {stripped_line}")

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    # p[0] と p[1] の両方が MAX_LENGTH 未満かチェック
    # かつ、p[1] (reverse=True の場合は英語側) が eng_prefixes で始まるかチェック
    return len(p[0].split(" ")) < MAX_LENGTH and \
        len(p[1].split(" ")) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    if input_lang is None: # ファイル読み込み失敗時
        return None, None, []
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    if not pairs:
        print("Warning: No pairs left after filtering!")
        return input_lang, output_lang, pairs # 空でも返す

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(f"{input_lang.name}: {input_lang.n_words}")
    print(f"{output_lang.name}: {output_lang.n_words}")
    return input_lang, output_lang, pairs
###############################################################################


# --- モデル定義 ---
###############################################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        # GRUへの入力は embedded のまま
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(
        self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Attention計算用の層
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # Attention適用後のベクトルをGRU入力用に変換する層
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 最終出力層
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # 1. Embedding と Dropout
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 2. Attention Weight の計算
        # embedded[0] と hidden[0] を連結し、attn層 -> softmax で重みを計算
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # 3. コンテキストベクトルの計算
        # attn_weights と encoder_outputs で重み付き和を計算 (バッチ行列積)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # 4. Decoder RNN への入力準備
        # embedded[0] と attn_applied[0] (コンテキストベクトル) を連結
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # attn_combine 層で hidden_size に変換し、GRU入力用に次元調整
        output = self.attn_combine(output).unsqueeze(0)

        # 5. ReLU と GRU
        output = F.relu(output)
        output, hidden = self.gru(output, hidden) # GRU実行

        # 6. 最終出力
        # GRUの出力を out 層 -> log_softmax で対数確率に変換
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
###############################################################################


# --- データ変換ユーティリティ ---
###############################################################################
def indexesFromSentence(lang, sentence):
    # 未知語処理を追加 (例: <UNK> トークンを使うなど)
    # ここでは単純に辞書にない単語は無視する
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
###############################################################################


# --- 学習 ---
###############################################################################
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Encoder の各ステップの出力を保存するテンソル
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # --- Encoder ---
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        # encoder_outputs に Encoder の出力を格納
        # encoder_output[0, 0] は (1, 1, hidden_size) -> (hidden_size)
        if ei < max_length: # max_length を超えないように
             encoder_outputs[ei] = encoder_output[0, 0]

    # --- Decoder ---
    decoder_input = torch.tensor([[SOS_token]], device=device) # 最初の入力は SOS
    decoder_hidden = encoder_hidden # Encoder の最後の隠れ状態を初期状態に

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: 次の入力としてターゲットを与える
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
            if decoder_input.item() == EOS_token and di == target_length - 1: # 最後のターゲットがEOSなら終了
                break

    else:
        # Teacher forcing なし: 自身の予測を次の入力として使用
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1) # 最も確率の高い単語を取得
            decoder_input = topi.squeeze().detach()  # detach() で計算グラフから切り離す

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token: # EOS が生成されたら終了
                break

    loss.backward() # 誤差逆伝播

    encoder_optimizer.step() # Encoder パラメータ更新
    decoder_optimizer.step() # Decoder パラメータ更新

    return loss.item() / target_length # ターゲット長で割った平均損失


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent) if percent > 0 else 0 # ゼロ除算回避
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, input_lang, output_lang, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    if not pairs:
        print("Error: Training pairs list is empty. Cannot start training.")
        return

    for iter_num in range(1, n_iters + 1):
        # イテレーションごとにペアをランダム選択＆テンソル化
        training_pair_sentences = random.choice(pairs)
        training_pair = tensorsFromPair(training_pair_sentences, input_lang, output_lang)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # 入力テンソルかターゲットテンソルが空の場合はスキップ
        if input_tensor.size(0) == 0 or target_tensor.size(0) == 0:
             print(f"Warning: Skipping iteration {iter_num} due to empty tensor from pair: {training_pair_sentences}")
             continue


        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter_num % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            percent_complete = iter_num / n_iters
            print('%s (%d %d%%) %.4f' % (timeSince(start, percent_complete),
                                         iter_num, percent_complete * 100, print_loss_avg))

        if iter_num % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
###############################################################################


# --- 評価 ---
###############################################################################
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.xlabel("Iterations (x{})".format(plot_every)) # X軸ラベル追加 (plot_everyが必要)
    plt.ylabel("Loss") # Y軸ラベル追加
    plt.title("Training Loss") # タイトル追加
    plt.show() # プロットを表示して待機


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad(): # 勾配計算を無効化
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        if input_length == 0: # 入力が空の場合
             return ["Error: Input sentence resulted in empty tensor."], None

        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # --- Encoder ---
        for ei in range(input_length):
            # max_length を超える入力は無視
            if ei >= max_length:
                break
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        # --- Decoder ---
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length) # Attentionを保存

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Attention weights を保存 (di行目に格納)
            decoder_attentions[di] = decoder_attention.data

            # 最も確率の高い単語を取得
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                # indexから単語に変換
                word = output_lang.index2word.get(topi.item(), '<UNK>') # 未知語対応
                decoded_words.append(word)

            decoder_input = topi.squeeze().detach() # 次の入力とする

        # Attention は生成した単語数分だけ返す (di+1 まで)
        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=10):
    if not pairs:
        print("Error: Pairs list is empty. Cannot evaluate.")
        return
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0]) # 入力文 (例: フランス語)
        print('=', pair[1]) # 正解文 (例: 英語)
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence) # モデルの出力文
        print('')
###############################################################################


# --- メイン処理 ---
###############################################################################
if __name__ == '__main__':
    # データ準備
    # 例: フランス語 -> 英語 の翻訳 (reverse=True)
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    # データが準備できた場合のみ実行
    if input_lang and output_lang and pairs:
        print("Random pair example:")
        print(random.choice(pairs))

        # モデル初期化
        hidden_size = 256
        encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

        # 学習実行
        # グローバル変数 plot_every を showPlot で使うために定義
        plot_every = 100
        trainIters(encoder1, attn_decoder1, input_lang, output_lang, pairs, 75000, print_every=5000, plot_every=plot_every)

        # 学習結果の評価
        print("\nEvaluating randomly chosen sentences from training data:")
        evaluateRandomly(encoder1, attn_decoder1, input_lang, output_lang, pairs)

        # (オプション) 特定の文で評価
        # print("\nEvaluating specific sentence:")
        # test_sentence = "je suis trop froid ." # 例: フランス語
        # output_words, attentions = evaluate(encoder1, attn_decoder1, test_sentence, input_lang, output_lang)
        # print('>', test_sentence)
        # print('<', ' '.join(output_words))
        # plt.matshow(attentions.numpy()) # Attentionの可視化
        # plt.show()
    else:
        print("Failed to prepare data. Exiting.")
###############################################################################