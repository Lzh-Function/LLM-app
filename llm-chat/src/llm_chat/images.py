"""画像添付の検証と OpenAI 互換形式への変換。"""

import base64
import binascii
import io
import re
import warnings

from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
DATA_URL = re.compile(r"^data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)$")


class ImageAttachment(BaseModel):
    name: str = Field(default="画像", max_length=255)
    data_url: str = Field(max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3) + 64)

    @field_validator("data_url")
    @classmethod
    def validate_image(cls, value: str) -> str:
        match = DATA_URL.fullmatch(value)
        if not match:
            raise ValueError("PNG・JPEG・WebP の画像データを指定してください")
        try:
            data = base64.b64decode(match[2], validate=True)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("画像は1枚5MiB以下にしてください")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    expected = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[match[1]]
                    if image.format != expected:
                        raise ValueError("画像形式とデータが一致しません")
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("画像は2000万画素以下にしてください")
                    image.verify()
        except (binascii.Error, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError("画像データを読み込めません") from exc
        return value


class ChatMessage(BaseModel):
    role: str
    content: str
    images: list[ImageAttachment] = Field(default_factory=list, max_length=MAX_IMAGES)

    @model_validator(mode="after")
    def validate_role(self):
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("未対応のメッセージ種別です")
        if self.images and self.role != "user":
            raise ValueError("画像はユーザーメッセージに添付してください")
        return self

    def backend_message(self) -> dict:
        content = self.content
        if self.images:
            content = [
                {"type": "image_url", "image_url": {"url": image.data_url}}
                for image in self.images
            ]
            content.append({"type": "text", "text": self.content or "この画像について説明してください。"})
        return {"role": self.role, "content": content}
