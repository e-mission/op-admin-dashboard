import io
import base64
import zipfile
import qrcode

def make_qrcode_base64_img(token):
    url = f'nrelopenpath://login_token?token={token}'
    img = qrcode.make(url,
                      error_correction=qrcode.constants.ERROR_CORRECT_H,
                      box_size=4)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def make_qrcodes_zipfile(tokens):
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for token in tokens:
            qrcode_b64 = make_qrcode_base64_img(token)
            image_data = base64.b64decode(qrcode_b64)
            zf.writestr(f"{token}.png", image_data)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
