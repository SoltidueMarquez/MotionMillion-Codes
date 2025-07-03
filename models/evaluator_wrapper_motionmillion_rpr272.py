
import torch
from os.path import join as pjoin
import numpy as np

class EvaluatorModelWrapper272RPR(object):

    def __init__(self, args, device):

        from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
        from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
        
        modelpath = 'distilbert-base-uncased'

        self.textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=512)
        self.motionencoder = ActorAgnosticEncoder(nfeats=272, vae = True, num_layers=4, max_len=300, latent_dim=512)
        self.device = device
                
        ckpt = torch.load('./checkpoints/evaluator/epoch=199.ckpt', map_location='cpu')
        
        # load textencoder
        textencoder_ckpt = {}
        for k, v in ckpt['state_dict'].items():
            if k.split(".")[0] == "textencoder":
                name = k.replace("textencoder.", "")
                textencoder_ckpt[name] = v
        self.textencoder.load_state_dict(textencoder_ckpt, strict=True)

        # load motionencoder
        motionencoder_ckpt = {}
        for k, v in ckpt['state_dict'].items():
            if k.split(".")[0] == "motionencoder":
                name = k.replace("motionencoder.", "")
                motionencoder_ckpt[name] = v
        self.motionencoder.load_state_dict(motionencoder_ckpt, strict=True)

        self.textencoder.to(device)
        self.motionencoder.to(device)

        self.textencoder.eval()
        self.motionencoder.eval()

        if args.dataname == 'motionmillion': # or args.dataname == 'motionmillion_30fps_600iterFSQ' or args.dataname == 'motionmillion_30fps_600iterFSQ_1024' or args.dataname == 'motionmillion_30fps_600iterFSQ_4096' or args.dataname == 'motionmillion_30fps_600iterFSQ_16384' or args.dataname == 'motionmillion_60fps':
            self.data_root = './dataset/MotionMillion'
                
        self.mean = np.load(pjoin(self.data_root, 'mean_std', 'vector_272', 'mean.npy'))
        self.std = np.load(pjoin(self.data_root, 'mean_std', 'vector_272', 'std.npy'))

    def normalize_motion(self, motions):
        with torch.no_grad():
            motions = (motions - torch.from_numpy(self.mean).to(self.device)) / torch.from_numpy(self.std).to(self.device)
        return motions
    
    def get_co_embeddings(self, texts, motions, m_lens):
        with torch.no_grad():
            et = self.textencoder(texts).loc
            motions = self.normalize_motion(motions)
            em = self.motionencoder(motions.float(), m_lens).loc
        return et, em

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = self.normalize_motion(motions)
            em = self.motionencoder(motions, m_lens).loc
        return em
