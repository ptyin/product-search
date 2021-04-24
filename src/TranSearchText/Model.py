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
        # nn.init.xavier_uniform_(self.user_embed.weight)
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

        def __get_item(vis, text):
            if self.mode == 'vis':
                vis = self.visual_fc(vis)
                concat = vis
            elif self.mode == 'text':
                text = self.textual_fc(text)
                concat = text
            else:
                vis = self.visual_fc(vis)
                text = self.textual_fc(text)
                concat = torch.cat((vis, text), dim=-1)
            item = self.item_fc(concat)
            item = self.translation(item)
            return item

        def __get_pred(u, q):
            u = self.elu(self.user_embed(u))
            u = self.translation(u)
            q = self.translation(self.query_embed(q))
            pred = u + q
            return pred

        if self.is_training:
            # Negative features attention and concatenation.
            item_predict = __get_pred(user, query)
            pos_item = __get_item(pos_vis, pos_text)
            neg_items = __get_item(neg_vis, neg_text)

            return item_predict, pos_item, neg_items
        else:
            if test_first:
                # positive features attention and concatenation
                return __get_item(pos_vis, pos_text)
            else:
                item_predict = __get_pred(user, query)
                # return item_predict, pos_item
                return item_predict
