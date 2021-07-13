import os
import torch
import numpy as np


def neighbor_similarity(config, user_embeddings):
    anchors = np.load(os.path.join(config.processed_path, 'experiments', 'anchors_step2.npy'))[:32]
    neighbors = np.load(os.path.join(config.processed_path, 'experiments', 'all_neighbors_step2.npy'))[:32]
    print('shape of anchors and neighbors:', anchors.shape, neighbors.shape)
    anchors = torch.tensor(anchors, dtype=torch.long).cuda()
    # (n,)
    neighbors = torch.tensor(neighbors, dtype=torch.long).cuda()
    # (n, k)
    anchor_embedding = user_embeddings[anchors]
    # (n, 64)
    neighbor_embeddings = user_embeddings[neighbors]
    # (n, k, 64)
    # similarity = torch.cosine_similarity(anchor_embedding.unsqueeze(dim=-2), neighbor_embeddings, dim=-1).mean(dim=0)
    similarity = torch.cosine_similarity(anchor_embedding.unsqueeze(dim=-2), neighbor_embeddings, dim=-1).sum(dim=0)
    # (n, k)
    # similarity = len(similarity) * similarity

    print('-----------------------------')
    # print('\t'.join(str(round(e, 3)) for e in torch.softmax(similarity, dim=-1).tolist()))
    print('\t'.join(str(round(e, 3)) for e in (similarity / similarity.sum()).tolist()))
    print('-----------------------------')

    # def cosine_similarity(a, b):
    #     norm_a = torch.sqrt(torch.sum(a * a, -1)).unsqueeze(dim=-1)
    #     norm_b = torch.sqrt(torch.sum(b * b, -1)).unsqueeze(dim=-1)
    #     dot = a @ b.t()
    #     score = dot / (norm_a @ norm_b.t() + 1e-8)
    #     return score
    #
    # all_similarity = cosine_similarity(user_embeddings, user_embeddings).mean()
    # print(similarity.mean(dim=0) / all_similarity)


def word_similarity(config, word_dict, model, top_k=100):
    # word_embeddings = word_embeddings[[word_dict['awesome'], word_dict['bad']]]
    # word_embeddings = word_embeddings[[word_dict['awesome'], word_dict['bad']]]
    words = np.load(os.path.join(config.processed_path, 'experiments', 'words.npy'))
    word_map = dict(zip(range(len(words)), words))
    # words = torch.tensor(words).cuda()
    user_embeddings: torch.Tensor = model.graph.nodes['entity'].data['e'][:, :model.entity_embedding_size]
    # word_embeddings = model.word_embedding_layer.weight[words]
    word_embeddings = model.word_embedding_layer.weight
    # word_embeddings = word_embeddings[words].unsqueeze(dim=-2)
    # doc_embeddings = model.doc_embedding(word_embeddings)

    anchor = user_embeddings[torch.tensor([338], dtype=torch.long).cuda()]
    similarity = torch.cosine_similarity(anchor.unsqueeze(dim=0), word_embeddings, dim=-1)
    top_similarities, top_words = similarity.topk(top_k, -1, True)

    reversed_word_dict = {v: k for k, v in word_dict.items()}
    for word, sim in zip(top_words[0], top_similarities[0]):
        if word.item() in words:
            print(reversed_word_dict[word.item()], round(sim.item(), 3), sep='\t')
        # elif np.random.random() < 0.1:
        #     print('not')
        #     print(reversed_word_dict[word.item()], round(sim.item(), 3), sep='\t')
        # print(reversed_word_dict[word_map[word.item()]], round(sim.item(), 3), sep='\t')
        # if word.item() in words:
        #     print('--------------------------')
    print('-------------------')
    similarity = torch.cosine_similarity(anchor, word_embeddings[[word_dict['pedal'],
                                                                  word_dict['iRiffport'],
                                                                  word_dict['chorus'],
                                                                  word_dict['screamer'],
                                                                  ]], dim=-1)
    print(similarity)
    top_words = [word_map[word.item()] for word in top_words[0]]
    return top_similarities[0], top_words
