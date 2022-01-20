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
    print("0")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("1")
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    print("2")
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    print("3")
    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()
    print("4")
    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    print("5")
    style = utils.load_image(args.style_image, size=args.style_size)
    print("6")
    style = style_transform(style)
    print("7")
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]
    print("8")
    for e in range(args.epochs):
        transformer.train()
        print("9")
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        print("10")
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            print("11")
            x = x.to(device)
            y = transformer(x)
            print("12")
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            print("13")
            features_y = vgg(y)
            features_x = vgg(x)
            print("14")
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight
            print("15")
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            print("16")
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            print("17")
            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
                print("18")
            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                print("19")
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                print("20")
                torch.save(transformer.state_dict(), ckpt_model_path)
                print("21")
                transformer.to(device).train()

    # save model
    print("22")
    transformer.eval().cpu()
    print("23")
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + str(
        int(args.content_weight)) + "_" + str(int(args.style_weight)) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    print("24")
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


# def main():
#     main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
#     subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
#
#     train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
#     train_arg_parser.add_argument("--epochs", type=int, default=2,
#                                   help="number of training epochs, default is 2")
#     train_arg_parser.add_argument("--batch-size", type=int, default=4,
#                                   help="batch size for training, default is 4")
#     train_arg_parser.add_argument("--dataset", type=str, required=True,
#                                   help="path to training dataset, the path should point to a folder "
#                                        "containing another folder with all the training images")
#     train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
#                                   help="path to style-image")
#     train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
#                                   help="path to folder where trained model will be saved.")
#     train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
#                                   help="path to folder where checkpoints of trained models will be saved")
#     train_arg_parser.add_argument("--image-size", type=int, default=256,
#                                   help="size of training images, default is 256 X 256")
#     train_arg_parser.add_argument("--style-size", type=int, default=None,
#                                   help="size of style-image, default is the original size of style image")
#     train_arg_parser.add_argument("--cuda", type=int, required=True,
#                                   help="set it to 1 for running on GPU, 0 for CPU")
#     train_arg_parser.add_argument("--seed", type=int, default=42,
#                                   help="random seed for training")
#     train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
#                                   help="weight for content-loss, default is 1e5")
#     train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
#                                   help="weight for style-loss, default is 1e10")
#     train_arg_parser.add_argument("--lr", type=float, default=1e-3,
#                                   help="learning rate, default is 1e-3")
#     train_arg_parser.add_argument("--log-interval", type=int, default=500,
#                                   help="number of images after which the training loss is logged, default is 500")
#     train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
#                                   help="number of batches after which a checkpoint of the trained model will be created")
#
#     eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
#     eval_arg_parser.add_argument("--content-image", type=str, required=True,
#                                  help="path to content image you want to stylize")
#     eval_arg_parser.add_argument("--content-scale", type=float, default=None,
#                                  help="factor for scaling down the content image")
#     eval_arg_parser.add_argument("--output-image", type=str, required=True,
#                                  help="path for saving the output image")
#     eval_arg_parser.add_argument("--model", type=str, required=True,
#                                  help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
#     eval_arg_parser.add_argument("--cuda", type=int, required=True,
#                                  help="set it to 1 for running on GPU, 0 for CPU")
#     eval_arg_parser.add_argument("--export_onnx", type=str,
#                                  help="export ONNX model to a given file")
#
#     args = main_arg_parser.parse_args()
#
#     if args.subcommand is None:
#         print("ERROR: specify either train or eval")
#         sys.exit(1)
#     if args.cuda and not torch.cuda.is_available():
#         print("ERROR: cuda is not available, try running on CPU")
#         sys.exit(1)
#
#     if args.subcommand == "train":
#         check_paths(args)
#         train(args)
#     else:
#         stylize(args)

class args:
    def __init__(self, content_image, model, output_image, cuda):
        # -は_に変換する
        self.content_image = content_image
        self.content_scale = None
        self.model = model
        self.output_image = output_image
        self.cuda = int(cuda)
        self.export_onnx = None

def stylize_for_apigateway(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_image = utils.base64_to_pil(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
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




class args_train:
    # def __init__(self, epochs=2, batch_size=4, dataset, style_image, save_model_dir,
    # checkpoint_model_dir=None, image_size=256, style_size=None, cuda=0, seed=42,
    # content_weight=1e5, style_weight=1e10, lr=1e-3, log_interval=500, checkpoint_interval=2000):
    def __init__(self, epochs, batch_size, dataset, style_image, save_model_dir,
    checkpoint_model_dir, image_size, style_size, cuda, seed,
    content_weight, style_weight, lr, log_interval, checkpoint_interval):

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.dataset = dataset #訓練データセットのフォルダ（その直下に複数のフォルダある構成になるので注意）
        self.style_image = style_image #スタイルの画像ファイル
        self.save_model_dir = save_model_dir #例えばapp/model
        self.checkpoint_model_dir = checkpoint_model_dir #中間モデルの出力
        self.image_size = int(image_size) #訓練データのサイズ。デフォルトは256*256
        self.style_size = int(style_size) #スタイル画像のサイズ。デフォルトはオリジナル
        self.cuda = int(cuda)
        self.seed = int(seed)
        self.content_weight = float(content_weight)
        self.style_weight = float(style_weight)
        self.lr = float(lr)
        self.log_interval = int(log_interval)
        self.checkpoint_interval = int(checkpoint_interval)


if __name__ == "__main__":
    print(sys.argv)
    (epochs, batch_size, dataset, style_image, save_model_dir, checkpoint_model_dir, image_size, style_size,
    cuda, seed, content_weight, style_weight, lr, log_interval, checkpoint_interval) = sys.argv[1:]

    args_train_class = args_train(epochs, batch_size, dataset, style_image, save_model_dir,
    checkpoint_model_dir, image_size, style_size, cuda, seed, content_weight,
    style_weight, lr, log_interval, checkpoint_interval)
    train(args_train_class)
