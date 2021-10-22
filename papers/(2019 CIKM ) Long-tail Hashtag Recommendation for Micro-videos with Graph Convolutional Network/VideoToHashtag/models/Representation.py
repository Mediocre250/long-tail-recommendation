import torch
from torch import nn
from torch.nn import functional as F


class RepresentationModel(nn.Module):
    def __init__(self, image_in, image_h, audio_in, audio_h, text_in, text_h, common_size):
        super(RepresentationModel, self).__init__()
        self.ilstm = ImageLSTM(image_in, image_h)
        self.alstm = AudioLSTM(audio_in, audio_h)
        self.tlstm = TextLSTM(text_in, text_h)

        # map into the common space
        self.ilinear = MLP(image_h, common_size)
        self.alinear = MLP(audio_h, common_size)
        self.tlinear = MLP(text_h, common_size)

    def forward(self, image, audio, text, audio_len, text_len):
        # lstm + attention
        h_image = self.ilstm(image)  # batch*image_h
        h_audio = self.alstm(audio, audio_len)  # batch*audio_h
        h_text = self.tlstm(text, text_len)  # batch*text_h

        # map into the common space
        e_image = self.ilinear(h_image)  # batch*common_size
        e_audio = self.alinear(h_audio)  # batch*common_size
        e_text = self.tlinear(h_text)  # batch*common_size

        # return three kind of features
        return e_image, e_audio, e_text


# Image LSTM
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ImageLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionLayer = AttentionNet(self.hidden_size)

    def forward(self, x):
        out, _ = self.ilstm(x)  # x:[batch*seq*feature], out:[batch*seq*feature]
        attention = self.attentionLayer(out)  # attention:[batch*seq*1]
        h = torch.mul(attention, out)  # h:[batch*seq*feature]
        sum_h = torch.sum(h, 1)  # sum_h:[batch*feature]
        return sum_h

# Audio LSTM
class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(AudioLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionLayer = AttentionNet(self.hidden_size)

    def forward(self, x, audio_len):

        X = nn.utils.rnn.pack_padded_sequence(x, audio_len, batch_first=True)
        X, _ = self.alstm(X)
        out, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # x:[batch*seq*feature], out:[batch*seq*feature]

        attention = self.attentionLayer(out)  # attention:[batch*seq*1]
        h = torch.mul(attention, out)  # h:[batch*seq*feature]
        sum_h = torch.sum(h, 1)  # sum_h:[batch*feature]
        return sum_h

# Text LSTM
class TextLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(TextLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tlstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionLayer = AttentionNet(self.hidden_size)

    def forward(self, x, text_len):

        X = nn.utils.rnn.pack_padded_sequence(x, text_len, batch_first=True)
        X, _ = self.tlstm(X)  # x:[batch*seq*feature], out:[batch*seq*feature]
        out, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        attention = self.attentionLayer(out)  # attention:[batch*seq*1]
        h = torch.mul(attention, out)  # h:[batch*seq*feature]
        sum_h = torch.sum(h, 1)  # sum_h:[batch*feature]
        return sum_h

# attention layer
class AttentionNet(nn.Module):
    def __init__(self, input_size):
        super(AttentionNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, 1)
        )

    def forward(self, x):
        out = self.linear(x)  # batch*seq*1
        out = F.softmax(out, dim=1)  # batch*seq*1
        return out  # batch*seq*1


# multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
