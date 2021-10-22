import time
import torch.utils.data as Data
import torch
import numpy as np


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        self.hashtag_num = 15751

        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('train_batch_size') is None:
            self.state['train_batch_size'] = 2048

        if self._state('validate_batch_size') is None:
            self.state['validate_batch_size'] = 8

        if self._state('test_batch_size') is None:
            self.state['test_batch_size'] = 8

        if self._state('epochs') is None:
            self.state['epochs'] = 30

        if self._state('topk') is None:
            self.state['topk'] = [5,10]

        self.image_seq = 40  # image samples
        self.audio_seq = 60  # audio samples
        self.text_seq = 6  # text samples

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def learning(self, state, helper, representaion_model, propagation_model, interaction_model,
                    train_dataset, validate_dataset, test_dataset,
                    criterion,optimizer_interaction,optimizer_propagation,optimizer_representation):
        for epoch in range(0, state['epochs']):

            # train model
            self.training_model(epoch, state, helper, representaion_model, propagation_model, interaction_model, train_dataset,
                    criterion, optimizer_interaction, optimizer_propagation, optimizer_representation)

            #  evaluate model
            self.evaluation_model(epoch, state, helper, representaion_model, propagation_model, interaction_model,
                                  validate_dataset)

            #  test model
            self.testing_model(epoch, state, helper, representaion_model, propagation_model, interaction_model,
                                  test_dataset)

    def my_collate_fn(self, batch_data):
        image_feature_list = []
        audio_feature_list = []
        text_feature_list = []
        audio_length = []
        text_length = []
        positive_hashtag = []
        negative_hashtag = []
        user_id = []

        for example in batch_data:
            image_feature_single, audio_feature_single, text_feature_single, positive_hashtag_single, negative_hashtag_single, user_id_single = example

            # positive hashtag, negative hashtag, user id
            positive_hashtag.append(positive_hashtag_single)
            negative_hashtag.append(negative_hashtag_single)
            user_id.append(user_id_single)

            # image feature
            image_feature_list.append(torch.from_numpy(np.abs(image_feature_single).astype(np.float32)))

            # audio feature process
            audio_pre = np.abs(audio_feature_single).astype(np.float32)
            length_a = audio_pre.shape[0]
            if length_a >= self.audio_seq:
                audio_length.append(self.audio_seq)
                audio_after = audio_pre[:self.audio_seq]
            else:
                audio_length.append(length_a)
                audio_after = np.zeros((self.audio_seq, 128)).astype(np.float32)
                for i in range(0, length_a):
                    audio_after[i] = audio_pre[i]
            audio_feature_list.append(torch.from_numpy(audio_after))

            # text featture process
            text_pre = np.abs(text_feature_single).astype(np.float32)
            length_t = text_pre.shape[0]
            if length_t >= self.text_seq:
                text_length.append(self.text_seq)
                text_after = text_pre[:self.text_seq]
            else:
                text_length.append(length_t)
                text_after = np.zeros((self.text_seq, 300)).astype(np.float32)
                for i in range(0, length_t):
                    text_after[i] = text_pre[i]
            text_feature_list.append(torch.from_numpy(text_after))

        # audio feature
        sorted_index_a = sorted(range(len(audio_length)), key=lambda k: audio_length[k], reverse=True)
        audio_feature_sorted = []
        for i in sorted_index_a:
            audio_feature_sorted.append(audio_feature_list[i])
        audio_length_sorted = sorted(audio_length, reverse=True)

        # text feature
        sorted_index_t = sorted(range(len(text_length)), key=lambda k: text_length[k], reverse=True)
        text_feature_sorted = []
        for i in sorted_index_t:
            text_feature_sorted.append(text_feature_list[i])
        text_length_sorted = sorted(text_length, reverse=True)

        img_feature = torch.stack(image_feature_list, dim=0)
        audio_feature = torch.stack(audio_feature_sorted, dim=0)
        text_feature = torch.stack(text_feature_sorted, dim=0)

        return img_feature, audio_feature, text_feature, audio_length_sorted, text_length_sorted, positive_hashtag, negative_hashtag, user_id

    def training_model(self, epoch, state, helper, representaion_model, propagation_model, interaction_model, train_dataset,
                    criterion, optimizer_interaction, optimizer_propagation, optimizer_representation):

        # define optimizer
        print("Training model---")

        representaion_model.train()
        propagation_model.train()
        interaction_model.train()

        print("Epoch: " + str(epoch + 1) + "--------")

        total_loss = 0
        begin_time = time.time()

        data_loader = Data.DataLoader(dataset=train_dataset,
                                      batch_size=state['train_batch_size'],
                                      collate_fn=self.my_collate_fn)

        for i, data in enumerate(data_loader):
            image_data, audio_data, text_data, audio_len, text_len, positive_hashtag, negative_hashtag, user_id = data
            image_data = helper.to_var(image_data, state['use_gpu'])  # batch*seq*feature
            audio_data = helper.to_var(audio_data, state['use_gpu'])
            text_data = helper.to_var(text_data, state['use_gpu'])

            # get the common space feature
            image_data, audio_data, text_data = representaion_model(image_data, audio_data, text_data, audio_len, text_len)

            hashtag_matrix = propagation_model()

            positive_hashtag = helper.to_var(torch.LongTensor(positive_hashtag), state['use_gpu'])
            negative_hashtag = helper.to_var(torch.LongTensor(negative_hashtag), state['use_gpu'])
            user_id = helper.to_var(torch.LongTensor(user_id), state['use_gpu'])

            optimizer_representation.zero_grad()
            optimizer_propagation.zero_grad()
            optimizer_interaction.zero_grad()

            pos_prediction = interaction_model(hashtag_matrix, image_data, audio_data, text_data, positive_hashtag, user_id)
            pos_target = helper.to_var(torch.from_numpy(np.ones((pos_prediction.detach().size())).astype(np.float32)),
                                      state['use_gpu'])
            neg_prediction = interaction_model(hashtag_matrix, image_data, audio_data, text_data, negative_hashtag, user_id)
            neg_target = helper.to_var(torch.from_numpy(np.zeros(neg_prediction.detach().size()).astype(np.float32)),
                                      state['use_gpu'])

            # cross entropy loss
            loss = criterion(pos_prediction, pos_target) + criterion(neg_prediction, neg_target)
            total_loss += loss.item()
            loss.backward()

            # optimizer.step()
            optimizer_interaction.step()
            optimizer_propagation.step()
            optimizer_representation.step()

            # end an batch
            print('the loss of batch %d is: %.4f ' % (i, loss))
            # with open('batch-loss.txt', 'a', encoding="utf-8") as f:
            #     f.write('the loss of model Epoch[%d / %d], batch %d: is %.4f , time: %d s' % (
            #         epoch, config.tagging_epoch, i, loss, time.time() - begin_time) + '\n')

        # end an epoch
        print('the loss of model Epoch[%d / %d]: is %.4f , time: %d s' % (
            epoch, state['epochs'], total_loss, time.time() - begin_time))

        with open('epoch-loss.txt', 'a', encoding="utf-8") as f:
            f.write('the loss of model Epoch[%d / %d]: is %.4f , time: %d s' % (
                epoch, state['epochs'], total_loss, time.time() - begin_time) + '\n')

        # save model
        torch.save(representaion_model.state_dict(), 'checkpoint/representationmodel/representation_%d_EPOCHS_loss%.4f.pkl' % (epoch, total_loss))
        torch.save(propagation_model.state_dict(), 'checkpoint/propagationmodel/propagation_model_%d_EPOCHS_loss%.4f.pkl' % (epoch, total_loss))
        torch.save(interaction_model.state_dict(), 'checkpoint/interactionmodel/interaction_%d_EPOCHS_loss%.4f.pkl' % (epoch, total_loss))

        print("Done the train of tagging model")



    def evaluation_model(self, epoch, state, helper, representaion_model, propagation_model, interaction_model,
                                  validate_dataset):
        begin_time = time.time()

        representaion_model.eval()
        propagation_model.eval()
        interaction_model.eval()

        print("Evaluation Model---")


        for j in state['topk']:

            topk = j
            Recall_s = []
            NDCG_s = []

            print("This is topk=" + str(topk) + " result")

            data_loader = Data.DataLoader(dataset=validate_dataset,
                                          batch_size=state['train_batch_size'],
                                          collate_fn=self.my_collate_fn)

            # evaluate
            for i, data in enumerate(data_loader):
                image_data, audio_data, text_data, audio_len, text_len, all_hashtag, true_label, user_id = data

                image_data = helper.to_var(image_data, state['use_gpu'])
                audio_data = helper.to_var(audio_data, state['use_gpu'])  # batch*seq*feature
                text_data = helper.to_var(text_data, state['use_gpu'])

                # get the common space feature
                image_data, audio_data, text_data = representaion_model(image_data, audio_data, text_data, audio_len, text_len)  # batch*common_size

                hashtag_matrix = propagation_model()

                image_data, audio_data, text_data = helper.expend(image_data, state['hashtag_num']), \
                                         helper.expend(audio_data, state['hashtag_num']), \
                                         helper.expend(text_data, state['hashtag_num'])  # batch*hashtag_num*common_size

                # leave one out measure
                all_hashtag = torch.LongTensor(all_hashtag)
                all_hashtag = helper.to_var(all_hashtag, state['use_gpu'])

                # user
                u_id = helper.to_var(torch.LongTensor(user_id), state['use_gpu'])
                u_id = helper.expend_u(u_id, state['hashtag_num'])

                # loop the data
                prediction = interaction_model(hashtag_matrix, image_data, audio_data, text_data, all_hashtag, user_id)
                prediction = torch.squeeze(prediction, 2)  # batch*hashtag_num

                # torch.topk
                temp_count = 0
                for true_list in true_label:
                    Recall_s.append(
                        helper.count_Recall(prediction.detach()[temp_count], all_hashtag[temp_count], true_list, topk))
                    NDCG_s.append(helper.count_ndcg(prediction.detach()[temp_count], all_hashtag[temp_count], true_list, topk))
                    temp_count += 1

                    # print('Batch ' + str(i))
                    # print("Recall is:", np.mean(Recall_s))
                    # print("NDCG is:", np.mean(NDCG_s))

            # count the Recall and NDCG
            ndcg = np.mean(NDCG_s)
            Recall = np.mean(Recall_s)
            print('time consume %d s' % (time.time() - begin_time))
            print("Recall is:", Recall)
            print("NDCG is:", ndcg)

            with open('evaluation.txt', 'a', encoding="utf-8") as f:
                f.write("Epoch: " + str(epoch + 1) + "--------")
                f.write("Topk= " + str(topk) + ", Recall is: " + str(Recall) + '\n')
                f.write("Topk= " + str(topk) + ", NDCG is: " + str(ndcg) + '\n')

    def testing_model(self, epoch, state, helper, representaion_model, propagation_model, interaction_model,
                         test_dataset):
        begin_time = time.time()

        representaion_model.eval()
        propagation_model.eval()
        interaction_model.eval()

        print("Testing Model---")

        for j in state['topk']:

            topk = j
            Recall_s = []
            NDCG_s = []

            print("This is topk=" + str(topk) + " result")

            data_loader = Data.DataLoader(dataset=test_dataset,
                                          batch_size=state['train_batch_size'],
                                          collate_fn=self.my_collate_fn)

            # test
            for i, data in enumerate(data_loader):
                image_data, audio_data, text_data, audio_len, text_len, all_hashtag, true_label, user_id = data

                image_data = helper.to_var(image_data, state['use_gpu'])
                audio_data = helper.to_var(audio_data, state['use_gpu'])  # batch*seq*feature
                text_data = helper.to_var(text_data, state['use_gpu'])

                # get the common space feature
                image_data, audio_data, text_data = representaion_model(image_data, audio_data, text_data, audio_len,
                                                                        text_len)  # batch*common_size

                hashtag_matrix = propagation_model()

                image_data, audio_data, text_data = helper.expend(image_data, state['hashtag_num']), \
                                                    helper.expend(audio_data, state['hashtag_num']), \
                                                    helper.expend(text_data,
                                                                  state['hashtag_num'])  # batch*hashtag_num*common_size

                # leave one out measure
                all_hashtag = torch.LongTensor(all_hashtag)
                all_hashtag = helper.to_var(all_hashtag, state['use_gpu'])

                # user
                u_id = helper.to_var(torch.LongTensor(user_id), state['use_gpu'])
                u_id = helper.expend_u(u_id, state['hashtag_num'])

                # loop the data
                prediction = interaction_model(hashtag_matrix, image_data, audio_data, text_data, all_hashtag, user_id)
                prediction = torch.squeeze(prediction, 2)  # batch*hashtag_num

                # torch.topk
                temp_count = 0
                for true_list in true_label:
                    Recall_s.append(
                        helper.count_Recall(prediction.detach()[temp_count], all_hashtag[temp_count], true_list, topk))
                    NDCG_s.append(
                        helper.count_ndcg(prediction.detach()[temp_count], all_hashtag[temp_count], true_list, topk))
                    temp_count += 1

                    # print('Batch ' + str(i))
                    # print("Recall is:", np.mean(Recall_s))
                    # print("NDCG is:", np.mean(NDCG_s))

            # count the Recall and NDCG
            ndcg = np.mean(NDCG_s)
            Recall = np.mean(Recall_s)
            print('time consume %d s' % (time.time() - begin_time))
            print("Recall is:", Recall)
            print("NDCG is:", ndcg)

            with open('testing.txt', 'a', encoding="utf-8") as f:
                f.write("Epoch: " + str(epoch + 1) + "--------")
                f.write("Topk= " + str(topk) + ", Recall is: " + str(Recall) + '\n')
                f.write("Topk= " + str(topk) + ", NDCG is: " + str(ndcg) + '\n')