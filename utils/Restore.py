import os
import torch
from config import settings

def load(args, model, filename = 'checkpoint.pth.tar'):
    savedir = os.path.join(settings.SNAPSHOT_DIR, args.dataset, args.arch, args.seed)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    print('Loaded weights from %s'%(snapshot))


def save_model(args, model, is_best=False, filename='checkpoint.pth.tar'):
    state = {'state_dict': model.state_dict()}
    savedir = os.path.join(settings.SNAPSHOT_DIR, args.dataset, args.arch, args.seed)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, 'Sess'+str(args.sess)+filename)
    torch.save(state, savepath)
    # if is_best:
    #     shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))
