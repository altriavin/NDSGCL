import torch.nn as nn
import faiss

from util import *


class ICLLDA(object):
    def __init__(self, conf, training_set, test_set, i):
        super(ICLLDA, self).__init__(conf, training_set, test_set)
        self.config = conf
        self.emb_size = int(self.config['embbedding.size'])
        args = OptionConf(self.config['NCL'])
        self.n_layers = int(args['-n_layer'])
        self.ssl_temp = float(args['-tau'])
        self.ssl_reg = float(args['-ssl_reg'])
        self.hyper_layers = int(args['-hyper_layers'])
        self.alpha = float(args['-alpha'])
        self.proto_reg = float(args['-proto_reg'])
        self.k = int(args['-num_clusters'])
        self.data = Interaction(conf, training_set, test_set)
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.lRate = float(self.config['learnRate'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        self.batch_size = int(self.config['batch_size'])
        self.reg = float(self.config['reg.lambda'])
        self.lncRNA_centroids = None
        self.lncRNA_2cluster = None
        self.drug_centroids = None
        self.drug_2cluster = None
        self.i = i

    def e_step(self):
        lncRNA_embeddings = self.model.embedding_dict['lncRNA_emb'].detach().cpu().numpy()
        drug_embeddings = self.model.embedding_dict['drug_emb'].detach().cpu().numpy()
        self.lncRNA_centroids, self.lncRNA_2cluster = self.run_kmeans(lncRNA_embeddings)
        self.drug_centroids, self.drug_2cluster = self.run_kmeans(drug_embeddings)

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, lncRNA_idx, drug_idx):
        lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        lncRNA2cluster = self.lncRNA_2cluster[lncRNA_idx]
        lncRNA2centroids = self.lncRNA_centroids[lncRNA2cluster]
        proto_nce_loss_lncRNA = InfoNCE(lncRNA_emb[lncRNA_idx],lncRNA2centroids,self.ssl_temp)
        drug2cluster = self.drug_2cluster[drug_idx]
        drug2centroids = self.drug_centroids[drug2cluster]
        proto_nce_loss_drug = InfoNCE(drug_emb[drug_idx],drug2centroids,self.ssl_temp)
        proto_nce_loss = self.proto_reg * (proto_nce_loss_lncRNA + proto_nce_loss_drug)
        return proto_nce_loss

    def ssl_layer_loss(self, context_emb, initial_emb, lncRNA, drug):
        context_lncRNA_emb_all, context_drug_emb_all = torch.split(context_emb, [self.data.lncRNA_num, self.data.drug_num])
        initial_lncRNA_emb_all, initial_drug_emb_all = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        context_lncRNA_emb = context_lncRNA_emb_all[lncRNA]
        initial_lncRNA_emb = initial_lncRNA_emb_all[lncRNA]
        norm_lncRNA_emb1 = F.normalize(context_lncRNA_emb)
        norm_lncRNA_emb2 = F.normalize(initial_lncRNA_emb)
        norm_all_lncRNA_emb = F.normalize(initial_lncRNA_emb_all)
        pos_score_lncRNA = torch.mul(norm_lncRNA_emb1, norm_lncRNA_emb2).sum(dim=1)
        ttl_score_lncRNA = torch.matmul(norm_lncRNA_emb1, norm_all_lncRNA_emb.transpose(0, 1))
        pos_score_lncRNA = torch.exp(pos_score_lncRNA / self.ssl_temp)
        ttl_score_lncRNA = torch.exp(ttl_score_lncRNA / self.ssl_temp).sum(dim=1)
        ssl_loss_lncRNA = -torch.log(pos_score_lncRNA / ttl_score_lncRNA).sum()

        context_drug_emb = context_drug_emb_all[drug]
        initial_drug_emb = initial_drug_emb_all[drug]
        norm_drug_emb1 = F.normalize(context_drug_emb)
        norm_drug_emb2 = F.normalize(initial_drug_emb)
        norm_all_drug_emb = F.normalize(initial_drug_emb_all)
        pos_score_drug = torch.mul(norm_drug_emb1, norm_drug_emb2).sum(dim=1)
        ttl_score_drug = torch.matmul(norm_drug_emb1, norm_all_drug_emb.transpose(0, 1))
        pos_score_drug = torch.exp(pos_score_drug / self.ssl_temp)
        ttl_score_drug = torch.exp(ttl_score_drug / self.ssl_temp).sum(dim=1)
        ssl_loss_drug = -torch.log(pos_score_drug / ttl_score_drug).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_lncRNA + self.alpha * ssl_loss_drug)
        return ssl_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):

            self.e_step()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                lncRNA_idx, pos_idx, neg_idx = batch
                model.train()
                rec_lncRNA_emb, rec_drug_emb, emb_list = model()
                lncRNA_emb, pos_drug_emb, neg_drug_emb = rec_lncRNA_emb[lncRNA_idx], rec_drug_emb[pos_idx], rec_drug_emb[neg_idx]
                rec_loss = bpr_loss(lncRNA_emb, pos_drug_emb, neg_drug_emb)
                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers*2]
                ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,lncRNA_idx,pos_idx)

                proto_loss = self.ProtoNCE_loss(initial_emb, lncRNA_idx, pos_idx)
                batch_loss = rec_loss + l2_reg_loss(self.reg, lncRNA_emb, pos_drug_emb) + ssl_loss + proto_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.drug(), 'ssl_loss', ssl_loss.drug(), 'proto_loss', proto_loss.drug())
            model.eval()
            with torch.no_grad():
                self.lncRNA_emb, self.drug_emb, _ = model()

    def predict(self, u):
        u = self.data.get_lncRNA_id(u)
        score = torch.matmul(self.lncRNA_emb[u], self.drug_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'lncRNA_emb': nn.Parameter(initializer(torch.empty(self.data.lncRNA_num, self.latent_size))),
            'drug_emb': nn.Parameter(initializer(torch.empty(self.data.drug_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['lncRNA_emb'], self.embedding_dict['drug_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        lncRNA_all_embeddings = lgcn_all_embeddings[:self.data.lncRNA_num]
        drug_all_embeddings = lgcn_all_embeddings[self.data.lncRNA_num:]
        return lncRNA_all_embeddings, drug_all_embeddings, all_embeddings