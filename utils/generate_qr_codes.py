import io
import base64
import zipfile
import qrcode

def make_qrcode_base64_img(url):
    img = qrcode.make(url,
                      error_correction=qrcode.constants.ERROR_CORRECT_H,
                      box_size=4)
    buffer = io.BytesIO()
    img.save(buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def make_qrcodes_zipfile(tokens):
    def zip_directory(bytes_io):
        with zipfile.ZipFile(bytes_io, mode="w") as zf:
            for token in tokens:
                url = f'nrelopenpath://login_token?token={token}'
                qrcode = make_qrcode_base64_img(url)
                image_data = base64.b64decode(qrcode)
                zf.writestr(token + '.png', image_data)
    return zip_directory
