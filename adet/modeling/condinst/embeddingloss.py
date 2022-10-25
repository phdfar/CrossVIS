
import torch
from torch import nn
from torch.nn import functional as F
from ._lovasz import LovaszHingeLoss


def build_emd():
    return EMDHead()


class EMDHead(nn.Module):
    def __init__(self):
        super(EMDHead, self).__init__()
        self.register_buffer('_iter', torch.zeros([1]))
        
    
    def compute_prob_map(self, embedding_map, instance_embeddings):
        """
        Compute the fg/bg probability per instance
        :param embedding_map: tensor(T, H, W, E)
        :param instance_embeddings: tensor(N, E)
        :param instance_bandwidth: tensor(N, E - N_FREE_DIMS)
        :return: tensor(T, H, W)
        """
        embedding_center = instance_embeddings.mean(dim=0, keepdim=True)[None, None, :]
        mean_bandwidth = torch.var(instance_embeddings,dim=0)
        mean_bandwidth = mean_bandwidth[None, None, :]
        probs = torch.exp(-0.5 * torch.sum(
            torch.pow(embedding_map - embedding_center, 2) * mean_bandwidth, dim=-1))
        
        return probs
    

    def __call__(self,mask_feats_1,mask_feats_2,gt_final):
        import random
        index = random.randint(1,555)
        

        if self.training:
            self._iter += 1


            x = mask_feats_1.unsqueeze(2)
            y = mask_feats_2.unsqueeze(2)
            embedding_map = torch.cat((x,y),dim=2)
            embedding_map = embedding_map.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
            #sig = torch.nn.Sigmoid()
            #embedding_map = torch.nn.functional.relu(embedding_map)
            #embedding_map = sig(embedding_map)
            lovasz_hinge_loss = LovaszHingeLoss()
            total_instances = 0;
            lovasz_loss = 0.
            losses = {}
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
            #print('start batch')
            for idx, (embeddings_per_seq, masks) in enumerate(zip(embedding_map,gt_final)):

              #masks = targets_per_seq.clone()

              if masks.numel() == 0:
                continue
              nonzero_mask_pts = masks.nonzero(as_tuple=False)
              if nonzero_mask_pts.shape[0] == 0:
                print("[ WARN] No valid mask points exist in sample.")
                continue
            
              _, instance_pt_counts = nonzero_mask_pts[:, 0].unique(sorted=True, return_counts=True)
              instance_id_sort_idx = nonzero_mask_pts[:, 0].argsort()
              nonzero_mask_pts = nonzero_mask_pts[instance_id_sort_idx]
              nonzero_mask_pts = nonzero_mask_pts.split(tuple(instance_pt_counts.tolist()))
              nonzero_mask_pts = tuple([nonzero_mask_pts[i].unbind(1)[1:] for i in range(len(nonzero_mask_pts))])
            
              instance_embeddings = [
                            embeddings_per_seq[nonzero_mask_pts[n]]
                            for n in range(len(nonzero_mask_pts))
                        ]  # list(tensor[I, E])
            
              total_instances += len(nonzero_mask_pts)

              for n in range(len(nonzero_mask_pts)):
                probs_map = self.compute_prob_map(embeddings_per_seq, instance_embeddings[n])
                logits_map = (probs_map * 2.) - 1.
                instance_target = masks[n].flatten()

                if instance_target.sum(dtype=torch.long) == 0:
                  continue
               
                g = lovasz_hinge_loss(logits_map.flatten(), instance_target)
                if torch.isnan(g)==False:
                  lovasz_loss = lovasz_loss + g

            if total_instances == 0:
              lovasz_loss = (embedding_map.sum()) * 0
            else:
                # compute weighted sum of lovasz and variance losses based on number of instances per batch sample
                lovasz_loss = lovasz_loss / total_instances
           

            return lovasz_loss

