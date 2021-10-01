import torch
import torch.nn as nn

class LMLoss(nn.Module):
    def __init__(self, reduce='none', ignore_idx=0):
        super(LMLoss, self).__init__()
        self.reduce = reduce
        self.lw = nn.CrossEntropyLoss(
            reduce=reduce,
            ignore_index=ignore_idx
        )

        self.lp = ContrastiveWeightedMarginLoss(reduce=reduce)
        
    def forward(
            self,
            input, target,
            pcontext, ppositves, pnegatives,
            acontext, apositves, anegatives,
            margin, temperature=0.1
    ):
        denom = input.size(0)

        nt_loss = self.lw(input.view(-1 , input.size(-1)), target.view(-1))
        np_loss = self.lp(pcontext, ppositves, pnegatives, margin)
        na_loss = self.lp(acontext, apositves, anegatives, margin)
        loss = nt_loss + temperature * (np_loss, na_loss)

        if self.reduce == 'none':
            return loss
        elif self.reduce == 'sum':
            return loss.sum()
        else:
            return loss.sum() / denom

class ContrastiveWeightedMarginLoss(nn.Module):
    def __init__(self, reduce='none'):
        super(ContrastiveWeightedMarginLoss, self).__init__()
        self.reduce = reduce
        self.sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, contexts, positives, negatives, margin=0.5):
        pos_sim = self.sim(contexts, positives)
        neg_sim = self.sim(contexts, negatives)
        logits = pos_sim - neg_sim + torch.log(margin + 1)
        logits = logits.view(-1, logits.size(-1))

        target = logits.new_zeros(contexts.size(0)*contexts.size(1), dtype=torch.float)

        denom = contexts.size(0)
        loss = max(logits, target)

        if self.reduce == 'none':
             return loss
        elif self.reduce == 'sum':
            return loss.sum()
        else:
            return loss.sum()/denom
