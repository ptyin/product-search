import torch
from torch import nn


class Model(nn.Module):
    def __init__(self,
                 visual_size, text_size, embed_size,
                 user_size, mode, dropout, is_training):
        super(Model, self).__init__()
        """ 
        Important Args:
        visual_size: for end_to_end is 4096, for others is not
        text_size: for end_to_end is 512, for others is not
        mode: could be 'end', vis', 'text', 'double'
        """
        self.visual_size = visual_size
        self.text_size = text_size
        self.embed_size = embed_size
        self.user_size = user_size
        self.mode = mode
        self.is_training = is_training
        self.elu = nn.ELU()

        # Custom weights initialization.
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        # visual fully connected layers
        def visual_fc():
            return nn.Sequential(
                nn.Linear(visual_size, embed_size),
                nn.ELU(),
                nn.Dropout(p=dropout),
                nn.Linear(embed_size, embed_size),
                nn.ELU())

        # textual fully connected layers
        def textual_fc():
            return nn.Sequential(
                nn.Linear(text_size, embed_size),
                nn.ELU(),
                nn.Dropout(p=dropout),
                nn.Linear(embed_size, embed_size),
                nn.ELU())

        if self.mode == 'double':
            self.visual_fc = visual_fc()
            self.textual_fc = textual_fc()
            self.visual_fc.apply(init_weights)
            self.textual_fc.apply(init_weights)
        elif self.mode == 'text':
            self.textual_fc = textual_fc()
        elif self.mode == 'vis':
            self.visual_fc = visual_fc()
        else:
            raise NotImplementedError

        # user and query embedding
        self.user_embed = nn.Embedding(self.user_size, embed_size)
        nn.init.xavier_uniform_(self.user_embed.weight)
        # nn.init.normal_(self.user_embed.weight, 0, 0.1)

        self.query_embed = nn.Sequential(
            nn.Linear(text_size, embed_size),
            nn.ELU())
        self.query_embed.apply(init_weights)

        # for embed user and item in the same space
        self.translation = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ELU())
        self.translation.apply(init_weights)

        # item fully connected layers
        if self.mode in ['end', 'double']:
            self.item_fc = nn.Sequential(
                nn.Linear(2 * embed_size, embed_size),
                # nn.ELU(),
                # nn.Dropout(p=dropout),
                # nn.Linear(embed_size, embed_size),
                nn.ELU())
        else:
            self.item_fc = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ELU(),
                nn.Linear(embed_size, embed_size),
                nn.ELU())
        self.item_fc.apply(init_weights)

    def forward(self, user, query, pos_text,
                neg_text, test_first, pos_vis=None, neg_vis=None):

        # positive features attention and concatenation
        if self.mode == 'vis':
            pos_vis = self.visual_fc(pos_vis)
            pos_concat = pos_vis
        elif self.mode == 'text':
            pos_text = self.textual_fc(pos_text)
            pos_concat = pos_text
        else:
            pos_vis = self.visual_fc(pos_vis)
            pos_text = self.textual_fc(pos_text)
            pos_concat = torch.cat((pos_vis, pos_text), dim=-1)
        # pos_concat = pos_vis * pos_text
        pos_item = self.item_fc(pos_concat)
        pos_item = self.translation(pos_item)

        if test_first:
            return pos_item

        user = self.elu(self.user_embed(user))
        user = self.translation(user)
        query = self.translation(self.query_embed(query))
        item_predict = user + query

        if self.is_training:
            # Negative features attention and concatenation.
            if self.mode == 'vis':
                neg_vis = self.visual_fc(neg_vis)
                neg_concat = neg_vis
            elif self.mode == 'text':
                neg_text = self.textual_fc(neg_text)
                neg_concat = neg_text
            else:
                neg_vis = self.visual_fc(neg_vis)
                neg_text = self.textual_fc(neg_text)
                neg_concat = torch.cat((neg_vis, neg_text), dim=-1)
            # neg_concat = neg_vis * neg_text
            neg_items = self.item_fc(neg_concat)
            neg_items = self.translation(neg_items)

            return item_predict, pos_item, neg_items
        else:
            return item_predict, pos_item
