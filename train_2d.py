import os
import argparse
import time
import yaml
from collections import OrderedDict
from datetime import datetime

import torch
import torch.distributed as dist

import cv2
import numpy as np
from einops import rearrange

from bspnet.data.dataset import BSP2dDataset
from bspnet.data.loader import fast_collate, PrefetchLoader
from bspnet.module.factory import factory as model_factory
from bspnet.module.bsp_2d.loss import BSP2dLoss
from bspnet.utils.checkpoint import CheckpointSaver
from bspnet.utils.metrics import update_summary, AverageMeter
from bspnet.utils.scheduler import cosine_scheduler


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset options
    parser.add_argument("dir", type=str)

    # experiment options
    parser.add_argument("--exp", type=str, default="baseline")
    parser.add_argument("--eval-metric", type=str, default="loss")

    # model options
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-planes", type=int, default=256)
    parser.add_argument("--num-primitives", type=int, default=64)
    parser.add_argument("--phase", type=int, default=0)

    # training options
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # misc
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.distributed = False    # TODO;
    args.world_size = 1

    # prepare dataset
    train_dataset = BSP2dDataset(
        os.path.join(args.dir, "train.hdf5")
    )
    valid_dataset = BSP2dDataset(
        os.path.join(args.dir, "eval.hdf5")
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=fast_collate,
        num_workers=args.num_workers,
        shuffle=True,
    )
    train_loader = PrefetchLoader(train_loader)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        collate_fn=fast_collate,
    )
    valid_loader = PrefetchLoader(valid_loader)

    # prepare model
    model = model_factory()
    model.cuda()

    # TODO;
    model_ema = None

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # scheduler
    lr_scheduler = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.num_epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    # loss
    loss_fn = BSP2dLoss(model)

    # saver
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    # if utils.is_primary(args):
    if True:    # TODO; DDP
        output_dir = os.path.join('output/bsp2d', f"{args.exp}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(output_dir)
        # decreasing = True if eval_metric == 'loss' else False
        decreasing = True
        saver = CheckpointSaver(
            model,
            optimizer,
            args,
            model_ema,
            checkpoint_prefix='checkpoint',
            checkpoint_dir=output_dir,
            decreasing=decreasing,
            max_history=5,
        )
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(args.num_epochs):
            if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                lr_scheduler,
                loss_fn,
                args,
                model_ema,
            )

            # TODO;
            # if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            #     if utils.is_primary(args):
            #         _logger.info("Distributing BatchNorm running means and vars")
            #     utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(
                epoch,
                model,
                valid_loader,
                loss_fn,
                args,
                output_dir,
                is_ema=False,
            )

            if model_ema is not None:
                # TODO;
                # if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                #     utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(
                    epoch,
                    model_ema,
                    valid_loader,
                    loss_fn,
                    args,
                    output_dir,
                    is_ema=True,
                )
                eval_metrics = ema_eval_metrics

            if output_dir is not None:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        lr_scheduler,
        loss_fn,
        args,
        model_ema=None,
):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    num_batches_per_epoch = len(loader)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch
    for batch_idx, batch in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        it = len(loader) * epoch + batch_idx
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[it]
            # no weight decay schedule

        # step
        optimizer.zero_grad()

        pred = model(batch)

        # reshape pred to 2d image
        c, h, w = batch.shape[1:]
        pred = rearrange(pred, "b (c h w) -> b c h w", c=c, h=h, w=w)

        loss = loss_fn(pred, batch)     # Autoencoder
        loss.backward()

        if not args.distributed:
            losses_m.update(loss.item(), batch.size(0))

        optimizer.step()

        # update EMA
        ema_beta = 0.5 ** (1 / max(num_updates, 1e-8))
        if model_ema is not None:
            for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(model_ema.buffers(), model.buffers()):
                b_ema.copy_(b)

        torch.cuda.synchronize()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:

            if args.distributed:

                def reduce_tensor(tensor, n):
                    rt = tensor.clone()
                    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
                    rt /= n
                    return rt

                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), batch.size(0))

            # TODO; consider DDP
            print(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g}) '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=batch.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=batch.size(0) * args.world_size / batch_time_m.avg,
                    data_time=data_time_m,
                )
            )

        end = time.time()
        # end for

    return OrderedDict([
        ('loss', losses_m.avg),
    ])


def validate(
        epoch,
        model,
        loader,
        loss_fn,
        args,
        output_dir,
        is_ema=False,
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    log_suffix = " (EMA)" if is_ema else ""
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            last_batch = batch_idx == last_idx

            pred = model(batch)

            # reshape pred to 2d image
            c, h, w = batch.shape[1:]
            pred = rearrange(pred, "b (c h w) -> b c h w", c=c, h=h, w=w)

            loss = loss_fn(pred, batch)

            if args.distributed:
                def reduce_tensor(tensor, n):
                    rt = tensor.clone()
                    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
                    rt /= n
                    return rt

                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), batch.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

            # save output samples
            if batch_idx == 0:
                preds = []
                targets = []

                for sample_idx in range(16):
                    preds.append(pred[sample_idx])
                    targets.append(batch[sample_idx])

                preds = torch.stack(preds, 0)
                targets = torch.stack(targets, 0)

                to_save = torch.stack([preds, targets], 0)
                to_save = to_save.detach().cpu().numpy()

                to_save = rearrange(to_save, "x b c h w -> (b h) (x w) c")
                to_save = np.clip(to_save, 0, 1)
                to_save = to_save * 255.
                to_save = to_save.astype(np.uint8)

                filename = f'result_epoch{epoch:03d}.jpg'
                cv2.imwrite(os.path.join(output_dir, filename), to_save)

            if last_batch or batch_idx % args.log_interval == 0:
                # TODO; consider DDP
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
                        log_name, batch_idx, last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                    )
                )

    return OrderedDict([
        ('loss', losses_m.avg),
    ])


if __name__ == '__main__':
    main()

        
