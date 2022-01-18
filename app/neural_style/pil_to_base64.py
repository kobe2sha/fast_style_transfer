import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image


def pil_to_base64(img, format="jpeg"):
    buffer = BytesIO()
    img.save(buffer, format)
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")

    return img_str


# 画像を読み込む。
img = Image.open('/content/69ACA424-10D4-4876-9CDB-BD8D332A3DB6.jpg')

# base64 文字列 (jpeg) に変換する。
img_base64 = pil_to_base64(img, format="jpeg")

# インライン画像として表示して確認する。
img_tag = f'<img src="data:image/jpeg;base64,{img_base64}">'
display(HTML(img_tag))

# base64 文字列 (png) に変換する。
img_base64 = pil_to_base64(img, format="png")

# インライン画像として表示して確認する。
img_tag = f'<img src="data:image/png;base64,{img_base64}">'
display(HTML(img_tag))
