import torch
import torch.nn as nn
import numpy as np
import math
from utils.rotation import quat_fk
from models import Animation
from models import AnimationStructure
import scipy.linalg as linalg

from models.Quaternions_old import Quaternions

class ForwardKinematics:
    def __init__(self, args, edges):
        self.topology = [-1] * (len(edges) + 1)
        self.rotation_map = []
        for i, edge in enumerate(edges):
            self.topology[edge[1]] = edge[0]
            self.rotation_map.append(edge[1])

        self.world = args.fk_world
        self.pos_repr = args.pos_repr
        self.quater = args.rotation == 'quaternion'

    def forward_from_raw(self, raw, offset, world=None, quater=None):
        if world is None: world = self.world
        if quater is None: quater = self.quater
        if self.pos_repr == '3d':
            position = raw[:, -3:, :]
            rotation = raw[:, :-3, :]
        elif self.pos_repr == '4d':
            raise Exception('Not support')
        if quater:
            rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1]))
            identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0], -1, 3, rotation.shape[-1]))
            identity = torch.zeros((3, ), dtype=torch.float, device=raw.device)
        identity = identity.reshape((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape)
        for i, j in enumerate(self.rotation_map):
            rotation_final[:, j, :, :] = rotation[:, i, :, :]
        return self.forward(rotation_final, position, offset, world=world, quater=quater)

    '''
    rotation should have shape batch_size * Joint_num * (3/4) * Time
    position should have shape batch_size * 3 * Time
    offset should have shape batch_size * Joint_num * 3
    output have shape batch_size * Time * Joint_num * 3
    '''
    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False, world=True):
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        result = torch.empty(rotation.shape[:-1] + (3, ), device=position.device)


        norm = torch.norm(rotation, dim=-1, keepdim=True)
        #norm[norm < 1e-10] = 1
        rotation = rotation / norm


        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            result[..., i, :] = torch.matmul(transform[..., i, :, :], offset[..., i, :, :]).squeeze()
            if world: result[..., i, :] += result[..., pi, :]
        return result

    def from_local_to_world(self, res: torch.Tensor):
        res = res.clone()
        for i, pi in enumerate(self.topology):
            if pi == 0 or pi == -1:
                continue
            res[..., i, :] += res[..., pi, :]
        return res

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m


class InverseKinematics:
    def __init__(self, rotations: torch.Tensor, positions: torch.Tensor, offset, parents, constrains):
        self.rotations = rotations
        self.rotations.requires_grad_(True)
        self.position = positions
        self.position.requires_grad_(True)

        self.parents = parents
        self.offset = offset
        self.constrains = constrains

        self.optimizer = torch.optim.Adam([self.position, self.rotations], lr=1e-3, betas=(0.9, 0.999))
        self.crit = nn.MSELoss()

    def step(self):
        self.optimizer.zero_grad()
        glb = self.forward(self.rotations, self.position, self.offset, order='', quater=True, world=True)
        loss = self.crit(glb, self.constrains)
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()

    def tloss(self, time):
        return self.crit(self.glb[time, :], self.constrains[time, :])

    def all_loss(self):
        res = [self.tloss(t).detach().numpy() for t in range(self.constrains.shape[0])]
        return np.array(res)

    '''
        rotation should have shape batch_size * Joint_num * (3/4) * Time
        position should have shape batch_size * 3 * Time
        offset should have shape batch_size * Joint_num * 3
        output have shape batch_size * Time * Joint_num * 3
    '''

    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False,
                world=True):
        '''
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        '''
        result = torch.empty(rotation.shape[:-1] + (3,), device=position.device)

        norm = torch.norm(rotation, dim=-1, keepdim=True)
        rotation = rotation / norm

        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.parents):
            if pi == -1:
                assert i == 0
                continue

            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            if world: result[..., i, :] += result[..., pi, :]
        return result

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m

class InverseKinematics_humdog:
    def __init__(self, rot, pos, offset, parents, constraints):
        self.rot = rot
        self.pos = pos
        self.rot.requires_grad_(True)
        self.pos.requires_grad_(True)

        self.parents = parents
        self.offset = offset
        self.constraints = constraints

        self.optimizer = torch.optim.Adam([self.pos, self.rot], lr=1e-3, betas=(0.9, 0.999))
        self.crit = nn.MSELoss()

    def step(self):
        self.optimizer.zero_grad()

        glb = self.forward(self.rot, self.pos, self.offset, self.parents)
        loss = self.crit(glb, self.constraints)
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()

    def forward(self, rot, pos, offset, parents):
        offset = offset.reshape(1, offset.shape[-2], offset.shape[-1]).repeat(rot.shape[0], 1, 1)
        offset[:, 0, :] = pos
        norm = torch.norm(rot, dim=-1, keepdim=True)
        rot = rot / norm
        _, x = quat_fk(rot, offset, parents)
        return x
    


class JacobianInverseKinematics:
    """
    Jacobian Based Full Body IK Solver
    
    This is a full body IK solver which
    uses the dampened least squares inverse
    jacobian method.
    
    It should remain fairly stable and effective
    even for joint positions which are out of
    reach and it can also take any number of targets
    to treat as end effectors.
    
    Parameters
    ----------
    
    animation : Animation
        animation to solve inverse problem on

    targets : {int : (F, 3) ndarray}
        Dictionary of target positions for each
        frame F, mapping joint index to 
        a target position
    
    references : (F, 3)
        Optional list of J joint position
        references for which the result
        should bias toward
        
    iterations : int
        Optional number of iterations to
        compute. More iterations results in
        better accuracy but takes longer to 
        compute. Default is 10.
        
    recalculate : bool
        Optional if to recalcuate jacobian
        each iteration. Gives better accuracy
        but slower to compute. Defaults to True
        
    damping : float
        Optional damping constant. Higher
        damping increases stability but
        requires more iterations to converge.
        Defaults to 5.0
        
    secondary : float
        Force, or bias toward secondary target.
        Defaults to 0.25
        
    silent : bool
        Optional if to suppress output
        defaults to False
    """
    
    def __init__(self, animation, targets,
        references=None, iterations=10,
        recalculate=True, damping=2.0,
        secondary=0.25, translate=False,
        silent=False, weights=None,
        weights_translate=None):
        
        self.animation = animation
        self.targets = targets
        self.references = references
        
        self.iterations  = iterations
        self.recalculate = recalculate
        self.damping   = damping
        self.secondary   = secondary
        self.translate   = translate
        self.silent      = silent
        self.weights     = weights
        self.weights_translate = weights_translate
        
    def cross(self, a, b):
        o = np.empty(b.shape)
        o[...,0] = a[...,1]*b[...,2] - a[...,2]*b[...,1]
        o[...,1] = a[...,2]*b[...,0] - a[...,0]*b[...,2]
        o[...,2] = a[...,0]*b[...,1] - a[...,1]*b[...,0]
        return o
        
    def jacobian(self, x, fp, fr, ts, dsc, tdsc):
        
        """ Find parent rotations """
        prs = fr[:,self.animation.parents]
        prs[:,0] = Quaternions.id((1))
        
        """ Find global positions of target joints """
        tps = fp[:,np.array(list(ts.keys()))]
        
        """ Get partial rotations """
        qys = Quaternions.from_angle_axis(x[:,1:prs.shape[1]*3:3], np.array([[[0,1,0]]]))
        qzs = Quaternions.from_angle_axis(x[:,2:prs.shape[1]*3:3], np.array([[[0,0,1]]]))
        
        """ Find axis of rotations """
        es = np.empty((len(x),fr.shape[1]*3, 3))
        es[:,0::3] = ((prs * qzs) * qys) * np.array([[[1,0,0]]])
        es[:,1::3] = ((prs * qzs) * np.array([[[0,1,0]]]))
        es[:,2::3] = ((prs * np.array([[[0,0,1]]])))
        
        """ Construct Jacobian """
        j = fp.repeat(3, axis=1)
        j = dsc[np.newaxis,:,:,np.newaxis] * (tps[:,np.newaxis,:] - j[:,:,np.newaxis])
        j = self.cross(es[:,:,np.newaxis,:], j)
        j = np.swapaxes(j.reshape((len(x), fr.shape[1]*3, len(ts)*3)), 1, 2)
                
        if self.translate:
            
            es = np.empty((len(x),fr.shape[1]*3, 3))
            es[:,0::3] = prs * np.array([[[1,0,0]]])
            es[:,1::3] = prs * np.array([[[0,1,0]]])
            es[:,2::3] = prs * np.array([[[0,0,1]]])
            
            jt = tdsc[np.newaxis,:,:,np.newaxis] * es[:,:,np.newaxis,:].repeat(tps.shape[1], axis=2)
            jt = np.swapaxes(jt.reshape((len(x), fr.shape[1]*3, len(ts)*3)), 1, 2)
            
            j = np.concatenate([j, jt], axis=-1)
        
        return j
        
    #@profile(immediate=True)
    def __call__(self, descendants=None, gamma=1.0):
        
        self.descendants = descendants
        
        """ Calculate Masses """
        if self.weights is None:
            self.weights = np.ones(self.animation.shape[1])
            
        if self.weights_translate is None:
            self.weights_translate = np.ones(self.animation.shape[1])
        
        """ Calculate Descendants """
        if self.descendants is None:
            self.descendants = AnimationStructure.descendants_mask(self.animation.parents)
        
        self.tdescendants = np.eye(self.animation.shape[1]) + self.descendants
        
        self.first_descendants = self.descendants[:,np.array(list(self.targets.keys()))].repeat(3, axis=0).astype(int)
        self.first_tdescendants = self.tdescendants[:,np.array(list(self.targets.keys()))].repeat(3, axis=0).astype(int)
        
        """ Calculate End Effectors """
        self.endeff = np.array(list(self.targets.values()))
        #from IPython import embed; embed();
        self.endeff = np.swapaxes(self.endeff, 0, 1) 
        
        if not self.references is None:
            self.second_descendants = self.descendants.repeat(3, axis=0).astype(int)
            self.second_tdescendants = self.tdescendants.repeat(3, axis=0).astype(int)
            self.second_targets = dict([(i, self.references[:,i]) for i in xrange(self.references.shape[1])])
        
        nf = len(self.animation)
        nj = self.animation.shape[1]
        
        if not self.silent:
            gp = Animation.positions_global(self.animation)
            gp = gp[:,np.array(list(self.targets.keys()))]            
            error = np.mean(np.sqrt(np.sum((self.endeff - gp)**2.0, axis=2)))
            print('[JacobianInverseKinematics] Start | Error: %f' % error)
        
        for i in range(self.iterations):

            """ Get Global Rotations & Positions """
            gt = Animation.transforms_global(self.animation)
            gp = gt[:,:,:,3]
            gp = gp[:,:,:3] / gp[:,:,3,np.newaxis]
            gr = Quaternions.from_transforms(gt)
            
            x = self.animation.rotations.euler().reshape(nf, -1)
            w = self.weights.repeat(3)
            
            if self.translate:
                x = np.hstack([x, self.animation.positions.reshape(nf, -1)])
                w = np.hstack([w, self.weights_translate.repeat(3)])
            
            """ Generate Jacobian """
            if self.recalculate or i == 0:
                j = self.jacobian(x, gp, gr, self.targets, self.first_descendants, self.first_tdescendants)
            
            """ Update Variables """            
            l = self.damping * (1.0 / (w + 0.001))
            d = (l*l) * np.eye(x.shape[1])
            e = gamma * (self.endeff.reshape(nf,-1) - gp[:,np.array(list(self.targets.keys()))].reshape(nf, -1))
            
            x += np.array(list(map(lambda jf, ef:
                linalg.lu_solve(linalg.lu_factor(jf.T.dot(jf) + d), jf.T.dot(ef)), j, e)))
            
            """ Generate Secondary Jacobian """
            if self.references is not None:
                
                ns = np.array(list(map(lambda jf:
                    np.eye(x.shape[1]) - linalg.solve(jf.T.dot(jf) + d, jf.T.dot(jf)), j)))
                    
                if self.recalculate or i == 0:
                    j2 = self.jacobian(x, gp, gr, self.second_targets, self.second_descendants, self.second_tdescendants)
                        
                e2 = self.secondary * (self.references.reshape(nf, -1) - gp.reshape(nf, -1))
                
                x += np.array(list(map(lambda nsf, j2f, e2f:
                    nsf.dot(linalg.lu_solve(linalg.lu_factor(j2f.T.dot(j2f) + d), j2f.T.dot(e2f))), ns, j2, e2)))

            """ Set Back Rotations / Translations """
            self.animation.rotations = Quaternions.from_euler(
                x[:,:nj*3].reshape((nf, nj, 3)), order='xyz', world=True)
                
            if self.translate:
                self.animation.positions = x[:,nj*3:].reshape((nf,nj, 3))
                
            """ Generate Error """
            
            if not self.silent:
                gp = Animation.positions_global(self.animation)
                gp = gp[:,np.array(list(self.targets.keys()))]
                error = np.mean(np.sum((self.endeff - gp)**2.0, axis=2)**0.5)
                print('[JacobianInverseKinematics] Iteration %i | Error: %f' % (i+1, error))