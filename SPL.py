from myutils.options import args
import myutils.common as utils
import time
from myutils.common import *
from myutils.conv_type import *
import mymodels
from load_dataset import *
from ssl.vat import VAT
from ssl.pseudo_label import PL
from ssl.pimodel import PiModel
visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]
checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
path = os.path.join(args.job_dir, 'logger' + now + '.log')
logger = utils.get_logger(path)

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# Data
print('==> Loading Data..')
args.l_batch_size = args.l_batch_size // 2
args.ul_batch_size = args.ul_batch_size // 2
args.test_batch_size = args.test_batch_size // 2
data_loaders = get_dataloaders(dataset='CIFAR10', n_labels=args.n_labels, n_unlabels=args.n_unlabels, n_valid=args.n_valid,
                               l_batch_size=args.l_batch_size, ul_batch_size=args.ul_batch_size,
                               test_batch_size=args.test_batch_size,
                               tot_class=args.n_class, ratio=args.ratio)
label_loader = data_loaders['labeled']
unlabel_loader = data_loaders['unlabeled']
test_loader = data_loaders['test']
val_loader = data_loaders['valid1']
val2_loader = data_loaders['valid2']

def train_first(model, optimizer, val_loader, args, epoch):
    model.train()
    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')

    for batch, (inputs, targets, _) in enumerate(val_loader):
        inputs, targets = inputs.float().to(device), targets.long().to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

def train_second(model2, optimizer2, label_loader, unlabel_loader, args, epoch, ssl_obj):
    losses2 = utils.AverageMeter(':.4e')
    accurary2 = utils.AverageMeter(':6.3f')
    start_time = time.time()
    iter_u = iter(unlabel_loader)
    for batch, (inputs_l, targets_l, _) in enumerate(label_loader):
        try:
            inputs_u, target_u, index_u = next(iter_u)
        except StopIteration:
            iter_u = iter(unlabel_loader)
            inputs_u, target_u, index_u = next(iter_u)

        l_images, l_labels = inputs_l.to(device).float(), targets_l.to(device).long()
        u_images, u_labels = inputs_u.to(device).float(), target_u.to(device).long()
        model2.train()
        labels = torch.cat([l_labels, u_labels], 0)
        ground_label = torch.cat([l_labels, u_labels], 0)
        labels[-len(u_labels):] = -1  # unlabeled mask
        unlabeled_mask = (labels == -1).float()
        images = torch.cat([l_images, u_images], 0)
        out = model2(images)
        ssl_loss = ssl_obj(images, out.detach(), model2, unlabeled_mask)
        cls_loss = F.cross_entropy(out, labels, reduction='none', ignore_index=-1).mean()
        loss = cls_loss + ssl_loss * args.consis_coef
        optimizer2.zero_grad()
        loss.backward()
        losses2.update(loss.item(), images.size(0))
        optimizer2.step()

        prec2 = utils.accuracy(out, ground_label)
        accurary2.update(prec2[0], images.size(0))

def validate(model, testLoader, epoch):
    global best_acc
    model.eval()

    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(testLoader):
            inputs, targets = inputs.float().to(device), targets.long().to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Epoch[{}]\t Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s'
                .format(epoch, float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg


def generate_pr_cfg(model, prune_rate):
    cfg_len = {
        'vgg': 17,
        'resnet32': 32,
    }

    pr_cfg = []
    if args.layerwise == 'l1':
        weights = []
        for name, module in model.named_modules():
            if hasattr(module, "set_prune_rate") and name != 'fc' and name != 'classifier':
                conv_weight = module.weight.data.detach().cpu()
                weights.append(conv_weight.view(-1))
        all_weights = torch.cat(weights, 0)
        preserve_num = int(all_weights.size(0) * (1 - prune_rate))
        preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
        threshold = preserve_weight[preserve_num - 1]

        for weight in weights:
            pr_cfg.append(torch.sum(torch.lt(torch.abs(weight), threshold)).item() / weight.size(0))
        pr_cfg.append(0)
    elif args.layerwise == 'uniform':
        pr_cfg = [prune_rate] * cfg_len[args.arch]
        pr_cfg[-1] = 0

    get_prune_rate(model, pr_cfg)

    return pr_cfg


def get_prune_rate(model, pr_cfg):
    all_params = 0
    prune_params = 0

    i = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_prune_rate"):
            w = module.weight.data.detach().cpu()
            params = w.size(0) * w.size(1) * w.size(2) * w.size(3)
            all_params = all_params + params
            prune_params += int(params * pr_cfg[i])
            i += 1

    logger.info('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % (
    (all_params - prune_params) / 1000000, all_params / 1000000, 100. * prune_params / all_params))


def main():
    start_epoch = 0
    best_acc = 0.0
    test_acc = 0.0

    if args.alg == "VAT":  # virtual adversarial training
        ssl_obj = VAT(6,  1e-67, 1)
    if args.alg == "PI":  # PI Model
        ssl_obj = PiModel()
    if args.alg == "PL":  # pseudo label
        ssl_obj = PL(0.95)

    prune_rate = args.prune_rate
    model, pr_cfg = get_model(args, logger, prune_rate)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    final_test_acc = 0
    for t in range(args.iteration):
        optimizer_m = get_optimizer(args, model, args.lr_m)
        optimizer_w = get_optimizer(args, model, args.lr_w)
        scheduler_m = torch.optim.lr_scheduler.StepLR(optimizer_m, step_size=args.step_size_m, gamma=0.1)
        scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=args.step_size_w, gamma=0.1)
        print("train iteration:", t)
        for epoch in range(start_epoch, args.epochs_m):

            freeze_model_weights(model)
            melt_model_masks(model)
            train_first(model, optimizer_m, val_loader, args, epoch)
            acc = validate(model, val2_loader, epoch)
            scheduler_m.step()
            if best_acc < acc:
                best_acc = acc
                test_acc = validate(model, test_loader, epoch)
                final_test_acc = test_acc
                save_pth_path = "./save_model/" + args.model_name + '.pth'
                torch.save(model.state_dict(), os.path.join(save_pth_path))

        for epoch in range(args.epochs_w):
            save_pth_path = "./save_model/" + args.model_name + '.pth'
            model.load_state_dict(
                torch.load(os.path.join(save_pth_path), map_location=device), strict=False)
            melt_model_weights(model)
            freeze_model_masks(model)
            train_second(model, optimizer_w, label_loader, unlabel_loader, args, epoch, ssl_obj)
            acc = validate(model, val2_loader, epoch)
            scheduler_w.step()
            if best_acc < acc:
                best_acc = acc
                test_acc = validate(model, test_loader, epoch)
                final_test_acc = test_acc
                save_pth_path = "./save_model/" + args.model_name + '.pth'
                torch.save(model.state_dict(), os.path.join(save_pth_path))

        save_pth_path = "./save_model/" + args.model_name + '.pth'
        model.load_state_dict(
            torch.load(os.path.join(save_pth_path), map_location=device), strict=False)
        print("iteration:", str(t), "final_test_acc:", final_test_acc)

    logger.info('Best accurary: {:.3f}'.format(float(test_acc)))


def resume(args, model, optimizer):
    if os.path.exists(args.job_dir + '/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")
        checkpoint = torch.load(args.job_dir + '/checkpoint/model_last.pt')
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")
        return start_epoch, best_acc
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")


def get_model(args, logger, prune_rate):
    pr_cfg = []
    print("=> Creating model '{}'".format(args.arch))
    model = mymodels.__dict__[args.arch]().to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.pretrained_model), map_location=device), strict=False)
    pr_cfg = generate_pr_cfg(model, prune_rate)
    set_model_prune_rate(model, pr_cfg, logger)
    if args.freeze_weights:
        freeze_model_weights(model)
    model = model.to(device)
    return model, pr_cfg


def get_optimizer(args, model, lr):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            lr,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
    return optimizer


if __name__ == '__main__':
    main()