import torch
from torch import nn

class InteractionModel(nn.Module):
    def __init__(self, input_size,  user_num, embed_size):
        super(InteractionModel, self).__init__()
        self.predict = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, 1)
        )

        # embedding layer to catch the feature of user
        self.user_embedding=nn.Embedding(user_num,embed_size)

    def forward(self, hashtag_matrix, image_data, audio_data, text_data, hashtag_id, user_id):
        # image_data, audio_data, text_data: batch*common_size
        # we concatenate the data

        topic_data = hashtag_matrix[hashtag_id]

        # User Embedding
        user_data = self.user_embedding(user_id)

        if len(topic_data.data.size()) == 2:
            # for train

            x = torch.cat(
                (torch.mul(image_data, topic_data) + torch.mul(image_data, user_data) + torch.mul(topic_data, user_data),
                 torch.mul(audio_data, topic_data) + torch.mul(audio_data, user_data) + torch.mul(topic_data, user_data),
                 torch.mul(text_data, topic_data) + torch.mul(text_data, user_data) + torch.mul(topic_data, user_data)),
                dim=1)
        else:
            # for evaluation
            x = torch.cat(
                (torch.mul(image_data, topic_data) + torch.mul(image_data, user_data) + torch.mul(topic_data, user_data),
                 torch.mul(audio_data, topic_data) + torch.mul(audio_data, user_data) + torch.mul(topic_data, user_data),
                 torch.mul(text_data, topic_data) + torch.mul(text_data, user_data) + torch.mul(topic_data, user_data)),
                dim=2)

        # predict a score
        out = torch.sigmoid(self.predict(x))
        return out
