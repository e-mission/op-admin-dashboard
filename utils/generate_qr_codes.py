import secrets
import argparse
import base64
import os
import qrcode
# import PIL as pil
import io

def readRandomTokens(filename):
    tokens = []
    with open(filename) as fp:
        tokens = [t.strip() for t in fp.readlines()]
    return tokens

def saveAsQRCode(token):
    qrcode_data = f"nrelopenpath://login_token?token={token}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(qrcode_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="generate_login_qr_codes")

    parser.add_argument("token_file_name")
    parser.add_argument("qr_code_dir")
    args = parser.parse_args()

    tokens = readRandomTokens(args.token_file_name)
    for t in tokens[0:10]:
        print(t)
    os.makedirs(args.qr_code_dir, exist_ok=True)
    for t in tokens:
        saveAsQRCode(args.qr_code_dir, t)    
