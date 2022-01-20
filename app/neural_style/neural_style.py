import boto3 #追加
import argparse
import os
import sys
import time
import re
import uuid
from urllib.parse import unquote_plus

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

# ローカル
import utils
from transformer_net import TransformerNet
from vgg import Vgg16
# DOCKER
# from neural_style import utils
# from neural_style.transformer_net import TransformerNet
# from neural_style.vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + str(
        int(args.content_weight)) + "_" + str(int(args.style_weight)) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    img_base64 = utils.save_image(args.output_image, output[0])
    return img_base64


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


# if __name__ == "__main__":
#     main()

class args:
    def __init__(self, content_image, model, output_image, cuda):
        # -は_に変換する
        self.content_image = content_image
        self.content_scale = None
        self.model = model
        self.output_image = output_image
        self.cuda = int(cuda)
        self.export_onnx = None

# args_class = args("app/images/content-images/train/amber.jpg",
# "app/saved_models/mosaic.pth", "app/images/output-images/アンバー4.jpg", 0)
# print("直打ち", args_class.cuda)

# stylize(args_class)

    # start = time.time()
    # print("スタート。引数は、", sys.argv[1:])
    #
    # args_list = sys.argv[1:]
    # args_class = args(*args_list)
    # def test():
    #     args_class = args(*args_list)
    #     stylize(args_class)
    #     t = time.time() - start
    #     print("終了。所要時間は、", t)
    # test()



# 入力DICT、event['Records']の中身はリスト。それをクラスに変換する
# 通常のローカルテスト用
# def handler(event, context):
#     print("レコード", event['Params'])
#     start = time.time()
#     args_list = event['Params']
#     print("スタート。引数は、", args_list)
#
#     args_class = args(*args_list)
#     stylize(args_class)
#
#     t = time.time() - start
#     print("終了。所要時間は、", t)

# S3用
# def handler(event, context):
#     # print("レコード", event['Params'])
#     start = time.time()
#     # model, output_image, cuda = event['Params']
#     # print("スタート。引数は、", model, output_image, cuda)
#     print("スタート")
#     # アップロードされた画像をダウンロード
#     # bucket = event['Records']['s3']['bucket']['name']
#     # key = event['Records']['s3']['object']['key']
#     s3_client = boto3.client('s3')
#     #バケット名、key名を修正する
#     for record in event['Records']:
#         bucket = record['s3']['bucket']['name']
#         print("バケット名", bucket)
#         key = unquote_plus(record['s3']['object']['key'])
#         print("KEY名", key)
#         tmpkey = key.replace('/', '')
#         print("tmpkey名", tmpkey)
#         # コンテンツパスを修正
#         # download_path = "images/content-images/train/s3img.jpg"
#         download_path = '/tmp/{}{}'.format(uuid.uuid4(), tmpkey)
#         print("ダウンロードパスは", download_path)
#         # スタイルパスを修正
#         model = "saved_models/candy.pth"
#         output_image = '/tmp/resized-{}'.format(tmpkey)
#         print("アウトプットDIR", output_image)
#         s3_client.download_file(bucket, key, download_path)
#         print("ダウンロードできた")
#         args_class = args(content_image=download_path, model=model, output_image=output_image, cuda=0)
#         print("推論開始！")
#         img_base64 = stylize(args_class)
#         print("出力完了")
        # s3_client.upload_file(output_image, '{}-resized'.format(bucket), key)
        # print("アップロード") resizedバケットにアップロードしないので不要


def stylize_for_apigateway(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_image = utils.base64_to_pil(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    img_base64 = utils.save_image(args.output_image, output[0])
    return img_base64

# api gateway用
def handler(event, context):
    start = time.time()
    print("スタート")
    base64_image = event["image"]
    # スタイルパスlambda
    # model = "saved_models/candy.pth"
    # パスlocal
    model = "app/saved_models/candy.pth"
    args_class = args(content_image=base64_image, model=model, output_image=None, cuda=0)
    print("推論開始！")
    # lambda
    # img_base64 = stylize_for_apigateway(args_class)
    # local
    img_base64 = stylize(args_class)
    print("出力完了", img_base64)
    # s3_client.upload_file(output_image, '{}-resized'.format(bucket), key)
    # print("アップロード") resizedバケットにアップロードしないので不要




    t = time.time() - start
    print("終了。所要時間は、", t)
    return {
    'statusCode': 200,
    'headers': {
  "Access-Control-Allow-Headers": 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
  "Access-Control-Allow-Origin": '*',
  "Access-Control-Allow-Methods": "OPTIONS,POST"

    },
    'body': img_base64
}


def main2(event, context):
    handler(event, context)

if __name__ == "__main__":
    print(sys.argv)
    event = {"image": sys.argv[1]}
    context = 1
    main2(event, context)

# handler(event={"image":
# "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"
# },
# context=None)
