import torch


checkpoint = []
def deepCopy(tbl):
    copy = []
    for k, v in enumerate(tbl):
        if type(v) == 'table':
            copy[k] = deepCopy(v)
        else:
            copy[k] = v
    if torch.typename(tbl):
        # torch.setmetatable(copy, torch.typename(tbl))
        pass
    return copy


def latest(opt):
   if opt.resume == 'none':
      return None

   latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath):
      return None

   print('=> Loading checkpoint ' + latestPath)
   latest = torch.load(latestPath)
   optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState


def save(epoch, model, optimState, isBestModel, opt):
    if torch.typename(model) == 'nn.DataParallelTable':
        # model = model
        pass

    modelFile = 'model_' + epoch + '.t7'
    optimFile = 'optimState_' + epoch + '.t7'

    torch.save(paths.concat(opt.save, modelFile), model)
    torch.save(paths.concat(opt.save, optimFile), optimState)
    torch.save(paths.concat(opt.save, 'latest.t7'), {
        epoch = epoch,modelFile = modelFile,optimFile = optimFile,})

    if isBestModel:
    torch.save(paths.concat(opt.save, 'model_best.t7'), model)